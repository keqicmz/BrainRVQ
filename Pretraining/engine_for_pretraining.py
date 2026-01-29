import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from einops import rearrange
from contextlib import nullcontext

def compute_information_content(patches, fs=200):
    B, N, A, T = patches.shape
    patches_flat = patches.reshape(B, N * A, T)
    device = patches.device
    
    fft = torch.fft.rfft(patches_flat, dim=-1)
    power = torch.abs(fft) ** 2
    freqs = torch.fft.rfftfreq(T, d=1.0/fs).to(device)
    
    total = power.sum(dim=-1) + 1e-8
    
    neural_mask = (freqs >= 4) & (freqs < 30)   # theta + alpha + beta
    artifact_mask = (freqs < 2) | (freqs >= 45) 
    
    neural_power = power[..., neural_mask].sum(dim=-1)
    artifact_power = power[..., artifact_mask].sum(dim=-1)
    
    neural_ratio = neural_power / total
    clean_score = 1 - (artifact_power / total)
    
    var = patches_flat.var(dim=-1) + 1e-8
    
    dx = torch.diff(patches_flat, dim=-1)
    var_dx = dx.var(dim=-1) + 1e-8
    
    ddx = torch.diff(dx, dim=-1)
    var_ddx = ddx.var(dim=-1)
    
    activity = torch.log(var)
    mobility = torch.sqrt(var_dx / var)
    
    complexity = torch.sqrt(var_ddx / var_dx) / (mobility + 1e-8)
    
    abs_diff = dx.abs()
    diff_of_diff = torch.diff(abs_diff, dim=-1).abs()
    irregularity = diff_of_diff.mean(dim=-1) / (abs_diff.mean(dim=-1) + 1e-8)
    
    def normalize(x):
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + 1e-8)
    
    info_scores = (
        0.30 * normalize(neural_ratio) + 
        0.25 * normalize(clean_score) +  
        0.20 * normalize(complexity) +   
        0.15 * normalize(irregularity) +  
        0.10 * normalize(mobility) 
    )
    
    return info_scores

def limit_mask_continuous_length_vectorized(mask, max_len=5):
    if mask.dtype != torch.bool:
        mask = mask.bool()

    B, L = mask.shape
    device = mask.device

    padded_mask = torch.cat([torch.zeros(B, 1, device=device, dtype=torch.bool), mask], dim=1)
    indices = torch.arange(L + 1, device=device).unsqueeze(0).expand(B, -1)
    zero_locations = indices * (~padded_mask).long()
    last_zero_idx = zero_locations.cummax(dim=1).values
    consecutive_counts = (indices - last_zero_idx)[:, 1:]

    limit = max_len + 1
    to_unmask = (consecutive_counts % limit == 0) & mask
    new_mask = mask.masked_fill(to_unmask, False)

    return new_mask


def apply_random_punching(mask, punch_ratio):
    if punch_ratio <= 0:
        return mask

    rand_probs = torch.rand_like(mask, dtype=torch.float32)
    to_unmask = mask & (rand_probs < punch_ratio)
    mask = mask.masked_fill(to_unmask, False)

    return mask

def get_curriculum_info_weight(epoch, total_epochs):
    progress = epoch / max(total_epochs, 1)
    weight = 0.2 + 0.5 * progress
    return weight

def generate_random_mask(samples, mask_ratio, device, punch_ratio=0.0, max_len=5):
    B, N, A, T = samples.shape
    L = N * A
    num_mask = int(L * mask_ratio)

    noise = torch.rand(B, L, device=device)
    _, top_indices = torch.topk(noise, k=num_mask, dim=-1)

    mask = torch.zeros([B, L], device=device, dtype=torch.bool)
    mask.scatter_(1, top_indices, True)

    if punch_ratio > 0:
        mask = apply_random_punching(mask, punch_ratio)

    mask = limit_mask_continuous_length_vectorized(mask, max_len=max_len)

    return mask


def generate_importance_guided_mask(samples, mask_ratio, device, 
                                    info_weight=0.5, temperature=0.8,
                                    punch_ratio=0.0, max_len=5):
    B, N, A, T = samples.shape
    L = N * A
    num_mask = int(L * mask_ratio)
    
    if num_mask == 0:
        return torch.zeros([B, L], device=device, dtype=torch.bool)
    
    with torch.no_grad():
        info_scores = compute_information_content(samples)  # (B, N*A)
        random_scores = torch.rand(B, L, device=device)
        combined_scores = info_weight * info_scores + (1 - info_weight) * random_scores

        mean = combined_scores.mean(dim=-1, keepdim=True)
        std = combined_scores.std(dim=-1, keepdim=True) + 1e-6
        norm_scores = (combined_scores - mean) / std
        
        logits = norm_scores / temperature
        probs = F.softmax(logits, dim=-1)
        
        ids_mask = torch.multinomial(probs, num_samples=num_mask, replacement=False)
        
        mask = torch.zeros((B, L), device=device, dtype=torch.bool)
        mask.scatter_(1, ids_mask, True)

        if punch_ratio > 0:
            mask = apply_random_punching(mask, punch_ratio)

        mask = limit_mask_continuous_length_vectorized(mask, max_len=max_len)
    
    return mask


def train_one_epoch(model: torch.nn.Module, vqnsp: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, ch_names_list=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()
    
    mask_ratio = getattr(args, 'mask_ratio', 0.5) 
    temperature = getattr(args, 'temperature', 0.8) 
    use_importance_mask = getattr(args, 'use_importance_mask', True)
    max_continuous_mask = getattr(args, 'max_continuous_mask', 5)
    punch_ratio = getattr(args, 'punch_ratio', 0.1)
    
    info_weight = get_curriculum_info_weight(epoch, args.epochs)

    if utils.is_main_process() and epoch == 0:
        print(f"[Mask Config] mask_ratio={mask_ratio}, info_weight={info_weight}, temperature={temperature}, "
              f"use_importance_mask={use_importance_mask}, punch_ratio={punch_ratio}, max_len={max_continuous_mask}")

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(data_loader) == 0:
            continue
        input_chans = utils.get_input_chans(ch_names)
        layer_weights = [1.0, 0.5, 0.25]
        
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            # Assign learning rate & weight decay for each step
            it = start_steps + step + step_loader
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples = batch
            samples = samples.float().to(device, non_blocking=True) / 100
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            
            if use_importance_mask:
                bool_masked_pos = generate_importance_guided_mask(
                    samples, 
                    mask_ratio=mask_ratio,
                    device=device,
                    info_weight=info_weight,
                    temperature=temperature,
                    punch_ratio=punch_ratio,
                    max_len=max_continuous_mask
                )
            else:
                bool_masked_pos = generate_random_mask(
                    samples,
                    mask_ratio=mask_ratio,
                    device=device,
                    punch_ratio=punch_ratio,
                    max_len=max_continuous_mask
                )

            # ==================== Get RVQ token indices ====================
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    input_ids_t, input_ids_f = vqnsp.get_codebook_indices(samples, input_chans)
                    n_layers = input_ids_t.shape[0]
                
                labels_list_t = [input_ids_t[i] for i in range(n_layers)]
                labels_list_f = [input_ids_f[i] for i in range(n_layers)]
                
                labels_masked_t = [input_ids_t[i][bool_masked_pos] for i in range(n_layers)]
                labels_unmasked_t = [input_ids_t[i][~bool_masked_pos] for i in range(n_layers)]
                labels_masked_f = [input_ids_f[i][bool_masked_pos] for i in range(n_layers)]
                labels_unmasked_f = [input_ids_f[i][~bool_masked_pos] for i in range(n_layers)]

            # ==================== Forward pass with teacher forcing ====================
            my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            with my_context():
                with torch.cuda.amp.autocast():
                    x_rec_list_t, x_rec_sym_list_t, x_rec_list_f, x_rec_sym_list_f = model(
                        samples, input_chans, bool_masked_pos=bool_masked_pos,
                        labels_t=labels_list_t, labels_f=labels_list_f
                    )

                    # ==================== Compute losses ====================
                    loss_rec_t = sum(
                        layer_weights[i] * loss_fn(x_rec_list_t[i], labels_masked_t[i]) 
                        for i in range(n_layers)
                    )
                    loss_rec_sym_t = sum(
                        layer_weights[i] * loss_fn(x_rec_sym_list_t[i], labels_unmasked_t[i]) 
                        for i in range(n_layers)
                    )

                    loss_rec_f = sum(
                        layer_weights[i] * loss_fn(x_rec_list_f[i], labels_masked_f[i]) 
                        for i in range(n_layers)
                    )
                    loss_rec_sym_f = sum(
                        layer_weights[i] * loss_fn(x_rec_sym_list_f[i], labels_unmasked_f[i]) 
                        for i in range(n_layers)
                    )

                    loss = loss_rec_t + loss_rec_sym_t + loss_rec_f + loss_rec_sym_f

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                sys.exit(1)

            # Backward pass
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= args.gradient_accumulation_steps
            grad_norm = loss_scaler(
                loss, optimizer, clip_grad=max_norm,
                parameters=model.parameters(), create_graph=is_second_order, 
                update_grad=(step + 1) % args.gradient_accumulation_steps == 0
            )
            loss_scale_value = loss_scaler.state_dict()["scale"]
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            
            # ==================== Compute per-layer accuracies ====================
            mlm_acc_list_t = [
                (x_rec_list_t[i].max(-1)[1] == labels_masked_t[i]).float().mean().item()
                for i in range(n_layers)
            ]
            mlm_acc_sym_list_t = [
                (x_rec_sym_list_t[i].max(-1)[1] == labels_unmasked_t[i]).float().mean().item()
                for i in range(n_layers)
            ]
            mlm_acc_list_f = [
                (x_rec_list_f[i].max(-1)[1] == labels_masked_f[i]).float().mean().item()
                for i in range(n_layers)
            ]
            mlm_acc_sym_list_f = [
                (x_rec_sym_list_f[i].max(-1)[1] == labels_unmasked_f[i]).float().mean().item()
                for i in range(n_layers)
            ]
            
            # Log per-layer accuracies
            for i in range(n_layers):
                metric_logger.update(**{f'mlm_acc_t_L{i}': mlm_acc_list_t[i]})
                metric_logger.update(**{f'mlm_acc_sym_t_L{i}': mlm_acc_sym_list_t[i]})
                metric_logger.update(**{f'mlm_acc_f_L{i}': mlm_acc_list_f[i]})
                metric_logger.update(**{f'mlm_acc_sym_f_L{i}': mlm_acc_sym_list_f[i]})
                
                if log_writer is not None:
                    log_writer.update(**{f'mlm_acc_t_L{i}': mlm_acc_list_t[i]}, head="loss")
                    log_writer.update(**{f'mlm_acc_sym_t_L{i}': mlm_acc_sym_list_t[i]}, head="loss")
                    log_writer.update(**{f'mlm_acc_f_L{i}': mlm_acc_list_f[i]}, head="loss")
                    log_writer.update(**{f'mlm_acc_sym_f_L{i}': mlm_acc_sym_list_f[i]}, head="loss")
            
            # Overall accuracies
            mlm_acc_t = sum(mlm_acc_list_t) / len(mlm_acc_list_t)
            mlm_acc_sym_t = sum(mlm_acc_sym_list_t) / len(mlm_acc_sym_list_t)
            mlm_acc_f = sum(mlm_acc_list_f) / len(mlm_acc_list_f)
            mlm_acc_sym_f = sum(mlm_acc_sym_list_f) / len(mlm_acc_sym_list_f)

            metric_logger.update(mlm_acc_t=mlm_acc_t)
            metric_logger.update(mlm_acc_sym_t=mlm_acc_sym_t)
            metric_logger.update(loss_rec_t=loss_rec_t.item())
            metric_logger.update(mlm_acc_f=mlm_acc_f)
            metric_logger.update(mlm_acc_sym_f=mlm_acc_sym_f)
            metric_logger.update(loss_rec_f=loss_rec_f.item())

            if log_writer is not None:
                log_writer.update(mlm_acc_t=mlm_acc_t, head="loss")
                log_writer.update(mlm_acc_sym_t=mlm_acc_sym_t, head="loss")
                log_writer.update(loss_rec_t=loss_rec_t.item(), head="loss")
                log_writer.update(mlm_acc_f=mlm_acc_f, head="loss")
                log_writer.update(mlm_acc_sym_f=mlm_acc_sym_f, head="loss")
                log_writer.update(loss_rec_f=loss_rec_f.item(), head="loss")
                
            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")
                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        
        step_loader += step
    
    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}