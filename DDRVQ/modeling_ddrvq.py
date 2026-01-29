import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model

from modeling_finetune import NeuralTransformer
from rvq import NormEMAVectorQuantizer, ResidualVQ

class DDRVQ(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192, 
                 embed_dim=32,
                 decay=0.99,
                 quantize_kmeans_init=True,
                 decoder_out_dim=200,
                 smooth_l1_loss = False,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config['in_chans'] != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim

        # encoder & decode params
        print('Final encoder config', encoder_config)
        self.encoder = NeuralTransformer(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder_t = NeuralTransformer(**decoder_config)

        print('Final decoder config', decoder_config)
        self.decoder_f = NeuralTransformer(**decoder_config)
        
        self.quantize_t = ResidualVQ(n_embed=[2048*4, 8192, 8192], embedding_dim=64, beta=1, kmeans_init=quantize_kmeans_init, decay=decay)
        self.quantize_f = ResidualVQ(n_embed=[2048*4, 8192, 8192], embedding_dim=64, beta=1, kmeans_init=quantize_kmeans_init, decay=decay)

        self.patch_size = encoder_config['patch_size']
        self.token_shape = (62, encoder_config['EEG_size'] // self.patch_size)

        self.decoder_out_dim = decoder_out_dim

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim) # for quantize
        )
        self.decode_task_layer_time = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_amp = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.decode_task_layer_angle = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )

        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer_time.apply(self._init_weights)
        self.decode_task_layer_amp.apply(self._init_weights)
        self.decode_task_layer_angle.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed', 
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device
        
    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, **kwargs):
        quantize, embed_ind, loss = self.encode(data, input_chans=input_chans)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data
        output['quantize'] = rearrange(quantize, 'b d a c -> b (a c) d')

        return output

    def encode(self, x, input_chans=None):
        batch_size, n, a, t = x.shape
        encoder_features = self.encoder(x, input_chans, return_patch_tokens=True)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = n, N // n

        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        quantize_t, loss_t, embed_ind_t = self.quantize_t(to_quantizer_features)
        quantize_f, loss_f, embed_ind_f = self.quantize_f(to_quantizer_features)

        return quantize_t, embed_ind_t, loss_t, quantize_f, embed_ind_f, loss_f
    
    def decode(self, quantize_t, quantize_f, input_chans=None, **kwargs):
        decoder_features_t = self.decoder_t(quantize_t, input_chans, return_patch_tokens=True)
        decoder_features_f = self.decoder_f(quantize_f, input_chans, return_patch_tokens=True)

        rec_time = self.decode_task_layer_time(decoder_features_t)
        rec_amp = self.decode_task_layer_amp(decoder_features_f)
        rec_angle = self.decode_task_layer_angle(decoder_features_f)
        return rec_time, rec_amp, rec_angle
    
    def get_codebook_indices(self, x, input_chans=None, **kwargs):
        quantize_t, embed_ind_t, loss_t, quantize_f, embed_ind_f, loss_f = self.encode(x, input_chans)
        return torch.stack([ind.view(x.shape[0], -1) for ind in embed_ind_t], dim=0), torch.stack([ind.view(x.shape[0], -1) for ind in embed_ind_f], dim=0)

    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss
    
    def calculate_rec_time_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x
    
    def std_norm_time(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + 1e-6)

    def forward(self, x, input_chans=None, **kwargs):
        """
        x: shape [B, N, T]
        """
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        x_time_target = self.std_norm_time(x)
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)
        angle = torch.angle(x_fft)
        angle = self.std_norm(angle)

        quantize_t, embed_ind_t, emb_loss_t, quantize_f, embed_ind_f, emb_loss_f = self.encode(x, input_chans)
        
        rec_time, rec_amp, rec_angle = self.decode(quantize_t, quantize_f, input_chans)
        rec_loss = self.calculate_rec_loss(rec_amp, amplitude)
        rec_angle_loss = self.calculate_rec_loss(rec_angle, angle)
        rec_time_loss = self.calculate_rec_time_loss(rec_time, x_time_target)

        loss = emb_loss_t + emb_loss_f + rec_loss + rec_angle_loss + rec_time_loss

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/quant_loss_t'] = emb_loss_t.detach().mean()
        log[f'{split}/quant_loss_f'] = emb_loss_f.detach().mean()

        log[f'{split}/rec_loss'] = rec_loss.detach().mean()
        log[f'{split}/rec_angle_loss'] = rec_angle_loss.detach().mean()
        log[f'{split}/rec_time_loss'] = rec_time_loss.detach().mean()

        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log

def get_model_default_params():
    return dict(EEG_size=1600, patch_size=200, in_chans=1, num_classes=1000, embed_dim=200, depth=12, num_heads=10,  
                             mlp_ratio=4., qkv_bias=True,  qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                             norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., use_abs_pos_emb=True, 
                             use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)

@register_model
def ddrvq_encoder_base_decoder_3x200x12(pretrained=False, pretrained_weight=None, as_tokenzer=False, EEG_size=3200, 
                                            n_code=8192, code_dim=32, **kwargs):
    encoder_config, decoder_config = get_model_default_params(), get_model_default_params()

    # encoder settings
    encoder_config['EEG_size'] = EEG_size
    encoder_config['num_classes'] = 0
    # decoder settings
    decoder_config['EEG_size'] = EEG_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    decoder_out_dim = 200

    model = DDRVQ(encoder_config, decoder_config, n_code, code_dim, 
                 decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith('https'):
            weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
        else:
            weights = torch.load(pretrained_weight, map_location='cpu')
            
        if 'model' in weights:
            weights = weights['model']
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())
        
        for k in keys:
            if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
                del weights[k]
        model.load_state_dict(weights)
    return model

if __name__ == '__main__':
    pass






