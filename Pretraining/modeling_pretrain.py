import math
import torch
import torch.nn as nn
from functools import partial

from modeling_finetune import Block, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange
import numpy as np
import torch.nn.functional as F


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class TemporalConv(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x


class NeuralTransformerForMaskedEEGModeling(nn.Module):
    def __init__(self, EEG_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 30, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, input_chans, bool_masked_pos):
        batch_size, c, time_window, _ = x.size()
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.forward_features(x, input_chans=input_chans, bool_masked_pos=bool_masked_pos)
        if return_all_patch_tokens:
            return x
        x = x[:, 1:]
        if return_patch_tokens:
            return x
        if return_all_tokens:
            return self.lm_head(x)
        else:
            return self.lm_head(x[bool_masked_pos])


class NeuralTransformerForMEM_Autoregressive(nn.Module):
    def __init__(self, EEG_size=6000, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Base encoder (student)
        self.student = NeuralTransformerForMaskedEEGModeling(
            EEG_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
            num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, 
            drop_path_rate, norm_layer, init_values, attn_head_dim,
            use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std
        )
        
        # RVQ layer configurations
        self.n_layers = 3
        self.rvq_vocab_sizes = [8192, 8192, 8192]  # [2048*4, 4096*2, 4096*2] in your original
        
        # ==================== Time domain (quantize_t) ====================
        # Token embeddings for each RVQ layer (used to condition next layer)
        self.token_embed_t = nn.ModuleList([
            nn.Embedding(vocab_size_i, embed_dim) 
            for vocab_size_i in self.rvq_vocab_sizes
        ])
        
        # Prediction heads for each RVQ layer
        self.lm_heads_t = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size_i) 
            for vocab_size_i in self.rvq_vocab_sizes
        ])
        
        # Fusion layers: combine encoder features with previous layer embeddings
        # Option 1: Simple addition (no extra params)
        # Option 2: Learnable fusion with LayerNorm
        self.fusion_norm_t = nn.ModuleList([
            norm_layer(embed_dim) if i > 0 else nn.Identity()
            for i in range(self.n_layers)
        ])
        
        # ==================== Frequency domain (quantize_f) ====================
        self.token_embed_f = nn.ModuleList([
            nn.Embedding(vocab_size_i, embed_dim) 
            for vocab_size_i in self.rvq_vocab_sizes
        ])
        
        self.lm_heads_f = nn.ModuleList([
            nn.Linear(embed_dim, vocab_size_i) 
            for vocab_size_i in self.rvq_vocab_sizes
        ])
        
        self.fusion_norm_f = nn.ModuleList([
            norm_layer(embed_dim) if i > 0 else nn.Identity()
            for i in range(self.n_layers)
        ])
        
        # Initialize weights
        self.init_std = init_std
        self._init_rvq_weights()
    
    def _init_rvq_weights(self):
        """Initialize RVQ-specific weights"""
        for module_list in [self.lm_heads_t, self.lm_heads_f, 
                           self.token_embed_t, self.token_embed_f]:
            for module in module_list:
                if isinstance(module, nn.Linear):
                    trunc_normal_(module.weight, std=self.init_std)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.Embedding):
                    trunc_normal_(module.weight, std=self.init_std)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'student.cls_token', 'student.pos_embed', 'student.time_embed'}
    
    def forward_autoregressive(self, encoder_features, mask, lm_heads, token_embeds, 
                                fusion_norms, labels_list=None):
        B, N, D = encoder_features.shape
        device = encoder_features.device
        
        # Get masked features: [num_masked, D]
        masked_features = encoder_features[mask]  # [M, D] where M = sum(mask)
        num_masked = masked_features.shape[0]
        
        # Get unmasked features for symmetric loss
        # Note: We process masked and unmasked separately
        
        logits_list = []
        cumulative_embed = torch.zeros(num_masked, D, device=device)
        
        for layer_idx in range(self.n_layers):
            # Fuse encoder features with cumulative embeddings from previous layers
            if layer_idx == 0:
                current_features = masked_features
            else:
                # Add cumulative embedding and normalize
                current_features = fusion_norms[layer_idx](masked_features + cumulative_embed)
            
            # Predict logits for this layer
            logits = lm_heads[layer_idx](current_features)  # [M, vocab_size]
            logits_list.append(logits)
            
            # Get token embeddings for next layer (teacher forcing during training)
            if labels_list is not None and self.training:
                # Teacher forcing: use ground truth tokens
                tokens = labels_list[layer_idx]  # [M]
            else:
                # Inference: use predicted tokens
                tokens = logits.argmax(dim=-1)  # [M]
            
            # Accumulate embeddings
            token_embed = token_embeds[layer_idx](tokens)  # [M, D]
            cumulative_embed = cumulative_embed + token_embed
        
        return logits_list
    
    def forward(self, x, input_chans=None, bool_masked_pos=None, 
                labels_t=None, labels_f=None):
        # Encode with masked positions
        x_masked = self.student(x, input_chans, bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls = x_masked[:, 1:]  # Remove CLS token
        
        # Encode with inverted mask (symmetric)
        x_masked_sym = self.student(x, input_chans, ~bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls_sym = x_masked_sym[:, 1:]
        
        # Prepare labels for teacher forcing
        labels_masked_t = None
        labels_unmasked_t = None
        labels_masked_f = None
        labels_unmasked_f = None
        
        if labels_t is not None:
            # labels_t is a list of [B, seq_len] tensors
            labels_masked_t = [lab[bool_masked_pos] for lab in labels_t]
            labels_unmasked_t = [lab[~bool_masked_pos] for lab in labels_t]
        
        if labels_f is not None:
            labels_masked_f = [lab[bool_masked_pos] for lab in labels_f]
            labels_unmasked_f = [lab[~bool_masked_pos] for lab in labels_f]
        
        # ==================== Time domain predictions ====================
        x_rec_list_t = self.forward_autoregressive(
            x_masked_no_cls, bool_masked_pos,
            self.lm_heads_t, self.token_embed_t, self.fusion_norm_t,
            labels_masked_t
        )
        
        x_rec_sym_list_t = self.forward_autoregressive(
            x_masked_no_cls_sym, ~bool_masked_pos,
            self.lm_heads_t, self.token_embed_t, self.fusion_norm_t,
            labels_unmasked_t
        )
        
        # ==================== Frequency domain predictions ====================
        x_rec_list_f = self.forward_autoregressive(
            x_masked_no_cls, bool_masked_pos,
            self.lm_heads_f, self.token_embed_f, self.fusion_norm_f,
            labels_masked_f
        )
        
        x_rec_sym_list_f = self.forward_autoregressive(
            x_masked_no_cls_sym, ~bool_masked_pos,
            self.lm_heads_f, self.token_embed_f, self.fusion_norm_f,
            labels_unmasked_f
        )
        
        return x_rec_list_t, x_rec_sym_list_t, x_rec_list_f, x_rec_sym_list_f


# Alias for backward compatibility
NeuralTransformerForMEM = NeuralTransformerForMEM_Autoregressive


@register_model
def brainrvq_base_patch200_1600_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = NeuralTransformerForMEM_Autoregressive(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, 
        qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model