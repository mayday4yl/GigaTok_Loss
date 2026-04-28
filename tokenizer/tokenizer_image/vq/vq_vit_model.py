# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers
#   maskgit: https://github.com/google-research/maskgit
#   REPA: https://github.com/sihyun-yu/REPA
#   DETR: https://github.com/facebookresearch/detr
from dataclasses import dataclass, field
import math
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, reduce, pack, unpack

import numpy as np


from tokenizer.tokenizer_image.vq.blocks import (
    ViTEncoder, ViTDecoder, Encoder, Decoder, ViTDecoder_V2,
    ViTEncoder2D, ViTDecoder2D,
    ChannelDownsampleResidual, ChannelUpsampleResidual,
)
from tokenizer.tokenizer_image.vq.gptc import (
    GPTC_models
)

from tokenizer.tokenizer_image.vq.lfq import (
    LFQ
)



def set_requires_grad(requires_grad, *models):
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, torch.nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (torch.nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            assert False, "unknown type %r" % type(model)


@dataclass
class VQVitModelPlusArgs:

    # for quantization
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0

    # tricks for lfq. deprecated
    use_lfq: bool = False
    bernoulli_sample: bool = False
    eval_deterministic: bool = False

    # SimVQ trick, deprecated
    simvq: bool = False
    codebook_transform: str = None
    freeze_codebook: bool = False
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    model_size: str = 'small'
    encoder_size: str = None 
    decoder_size: str = None
    num_latent_tokens: int = 256
    z_channels: int = 256   # the dimension of the intermediate downsample towards codebook dimension
    dropout_p: float = 0.0

    # use rope for the decoder Q-former attention. 
    use_rope: bool = False
    use_qk_norm: bool = False

    # TODO: remove option. flash attention is automatically used when calling scaled_dot_product_attention
    use_flash_attn: bool = False

    # the setting for initializing the 1d queries for the 2dto1d encoder
    multi_level_query_init: bool = False
    learnable_1d_query_init: bool = False
    rope_1d: bool = False

    # the initialization for the 2d queries. the "level" corresponds to the 
    # "level" division for "multi_level_query_init". It assumes 1d tokens have levels
    # all false means simply using the global average of the 1d tokens to initialize
    # the 2d queries.
    last_level_2d_query_init: bool = False
    multi_level_2d_query_init: bool = False
    learnable_2d_query_init: bool = False

    # tricks for the CNN 2d decoder
    adaptive_gn: bool = False
    d2s_up: bool = False
    res_up_down_sample: bool = False
    downsample_match_channel: bool = False
    upsample_match_channel: bool = False
    res_codebook_updown_sample: bool = False
    downsample_improve: bool = False
    # whether to use attention in the 2d encoder or decoder
    # suggested not to. May be unstable and slower
    use_attn: bool = True

    # rope 2d only supports the 1dto2d decoder queries (since Q-former)
    rope_2d: bool = False

    # the rotation trick for quantizer. The influence is limited
    rot: bool = False

    # for stochastic quantization. Closed by default
    stochastic: bool = False
    stochastic_temperature: float = 0.03

    # distillation setting
    distill_depth: int = None
    # whether to distill from encoder. Not tested yet.
    # (to be deleted)
    encoder_2d_distill: bool = False

    # for semantic distillation regularization
    # the default 768 is for dino-v2 base
    out_inner_dim: int = 768

    fea_rec_loss_type: str = "cosine"
    fea_rec_loss_weight: float = 1.0

    # for gptc model, which tries to utilize AR prior for 
    # training tokenizers. The effect is limited and this feature
    # is deprecated.
    # for ar prior model
    with_prior_model: bool = False
    prior_model_config: dict = None




class VQVitModelPlus(nn.Module):
    def __init__(self, config: VQVitModelPlusArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
                        ch_mult=config.encoder_ch_mult, 
                        z_channels=config.z_channels, 
                        dropout=config.dropout_p, 
                        use_attn=config.use_attn,
                        res_down_sample=config.res_up_down_sample,
                        downsample_match_channel=config.downsample_match_channel,
                        )

        if config.encoder_2d_distill:
            # setting is from REPA
            self.distill_mlp = nn.Sequential(
                    nn.Linear(config.z_channels, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.out_inner_dim),
                    )


        # set the size of the transformer encoder/decoder size
        encoder_size = config.model_size if config.encoder_size is None else config.encoder_size
        decoder_size = config.model_size if config.decoder_size is None else config.decoder_size
       
        # when encoder size or decoder size is given, model size should be none
        if config.encoder_size is not None or config.decoder_size is not None:
            assert config.model_size is None
        
        if config.encoder_2d_distill:
            assert config.distill_depth is None



        self.s2to1encoder = ViTEncoder(model_size=encoder_size, num_latent_tokens=config.num_latent_tokens, 
                               token_size=config.z_channels, dropout=config.dropout_p, 
                               patch_size=2**(len(config.encoder_ch_mult) - 1),
                               multi_level_query_init=config.multi_level_query_init,
                               learnable_1d_query_init=config.learnable_1d_query_init,
                               rope_1d=config.rope_1d,
                               downsample_improve=config.downsample_improve,
                               use_qk_norm=config.use_qk_norm,
                               use_flash_attn=config.use_flash_attn,
                               )

        if config.use_rope:
            # V2 model is specifically designed for rope2d
            self.s1to2decoder = ViTDecoder_V2(model_size=decoder_size, num_latent_tokens=config.num_latent_tokens, 
                                token_size=config.z_channels, dropout=config.dropout_p,
                                patch_size=2**(len(config.decoder_ch_mult) - 1),
                                last_level_2d_query_init=config.last_level_2d_query_init,
                                multi_level_2d_query_init=config.multi_level_2d_query_init,
                                learnable_2d_query_init=config.learnable_2d_query_init,
                                rope_2d=True,
                                use_qk_norm=config.use_qk_norm,
                                use_flash_attn=config.use_flash_attn,
                                )
        
        else:
            self.s1to2decoder = ViTDecoder(model_size=decoder_size, num_latent_tokens=config.num_latent_tokens, 
                                token_size=config.z_channels, dropout=config.dropout_p,
                                patch_size=2**(len(config.decoder_ch_mult) - 1),
                                last_level_2d_query_init=config.last_level_2d_query_init,
                                multi_level_2d_query_init=config.multi_level_2d_query_init,
                                learnable_2d_query_init=config.learnable_2d_query_init,
                                out_inner_feat=config.distill_depth is not None,
                                out_inner_depth=config.distill_depth,
                                out_inner_dim=config.out_inner_dim,
                                use_qk_norm=config.use_qk_norm,
                                use_flash_attn=config.use_flash_attn,
                                )

        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, 
                               z_channels=config.z_channels, 
                               dropout=config.dropout_p,
                               adaptive_gn=config.adaptive_gn,
                               d2s_up=config.d2s_up,
                               use_attn=config.use_attn,
                               res_up_sample=config.res_up_down_sample,
                               upsample_match_channel=config.upsample_match_channel,
                               )

        self.num_latent_tokens = config.num_latent_tokens
        # scale = self.s2to1encoder.width ** -0.5
        # self.latent_tokens = nn.Parameter(
        #     scale * torch.randn(self.num_latent_tokens, self.s2to1encoder.width))

        # the weight initialization seems to have ignored post_quant_conv and pre_quant_conv
        # and it potentially affects the prior model(deprecated) training
        # but weight initialization is also ignored(?) in llamagen implementation
        # currently this setting can just work.
        self.apply(self._init_weights)

        if self.config.with_prior_model:
            if self.config.use_lfq:
                raise NotImplementedError("LFQ is not implemented yet")
            else:
                self.quantize = VectorQuantizerWithPM(
                                                config.codebook_size, config.codebook_embed_dim, 
                                                config.commit_loss_beta, config.entropy_loss_ratio,
                                                config.codebook_l2_norm, config.codebook_show_usage,
                                                rot=config.rot, stochastic=config.stochastic,
                                                stochastic_temperature=config.stochastic_temperature,
                                                prior_model_config=config.prior_model_config,
                                                simvq=config.simvq,
                                                codebook_transform=config.codebook_transform,
                                                freeze_codebook=config.freeze_codebook,
                                        )

            
        else:
            if self.config.use_lfq:
                self.quantize = LFQ(
                    dim=config.codebook_embed_dim,
                    beta=config.commit_loss_beta,
                    entropy_loss_ratio=config.entropy_loss_ratio,
                    n_e=config.codebook_size,
                )
            else:
                self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                                config.commit_loss_beta, config.entropy_loss_ratio,
                                                config.codebook_l2_norm, config.codebook_show_usage,
                                                rot=config.rot, stochastic=config.stochastic,
                                                stochastic_temperature=config.stochastic_temperature,
                                                eval_deterministic=config.eval_deterministic,
                                                simvq=config.simvq,
                                                codebook_transform=config.codebook_transform,
                                                freeze_codebook=config.freeze_codebook,
                                                )

        if self.config.res_codebook_updown_sample:

            if self.config.downsample_improve:
                self.quant_conv = ChannelDownsampleResidual(self.s2to1encoder.width, config.codebook_embed_dim)
            else:
                self.quant_conv = ChannelDownsampleResidual(config.z_channels, config.codebook_embed_dim)

            self.post_quant_conv = ChannelUpsampleResidual(config.codebook_embed_dim, self.config.z_channels)
        else:
            self.quant_conv = nn.Conv2d(self.config.z_channels, config.codebook_embed_dim, 1)
            self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, self.config.z_channels, 1)

        self.text_projection = None
        self.text_type_embedding = None
        self.visual_type_embedding = None
        self.text_gate_logit = None
        self.visual_mask_token = None
        self.residual_head_mlp = None
        self.residual_head_gate = None
        self.residual_text_mlp = None
        self.residual_gate = None
        
        self.freeze_but_2d_decoder_flag = False

        def nan_hook(self, inp, output):
            if not isinstance(output, torch.Tensor):
                return
            if torch.isnan(output).any():
                print(f"NaN detected in {self}")
                raise RuntimeError("NaN detected")

        # for name, module in self.named_modules():
        #     module.register_forward_hook(nan_hook)

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        wrap_modules = []
        # Add encoder layers
        for layer in self.s2to1encoder.transformer:
            wrap_modules.append(layer)
        # Add decoder layers
        for layer in self.s1to2decoder.transformer:
            wrap_modules.append(layer)

        return wrap_modules
        

    # def eval(self):
        # delete unused modules for inferencing
        # - semantic distillation mlp
        # - ar prior model

        # if self.config.encoder_2d_distill:
        #     del self.distill_mlp
        
        # if self.config.distill_depth is not None:
        #     del self.s1to2decoder.distill_mlp
        
        # if self.config.with_prior_model:
        #     del self.quantize.prior_model
        # super().eval()
    

    def freeze_but_2d_decoder(self):
        """deprecated"""
        for param in self.parameters():
            param.requires_grad = False

        set_requires_grad(True, self.decoder)
        self.freeze_but_2d_decoder_flag = True

    def apply_stage1_finetune_freeze(
            self,
            freeze_encoder=False,
            freeze_quantizer=False,
            freeze_codebook=False,
            freeze_post_quant_conv=False):
        if freeze_encoder:
            set_requires_grad(False, self.encoder, self.s2to1encoder, self.quant_conv)

        if freeze_quantizer:
            set_requires_grad(False, self.quantize)

        if freeze_codebook and hasattr(self.quantize, "embedding"):
            self.quantize.embedding.weight.requires_grad = False

        if freeze_post_quant_conv:
            set_requires_grad(False, self.post_quant_conv)

    def configure_text_conditioning(
            self,
            text_feature_dim,
            text_projection="linear_layernorm",
            text_type_embedding=True,
            visual_type_embedding=False,
            text_gate_enabled=False,
            text_gate_init=1.0,
            visual_memory_mask_enabled=False,
            text_recon_mode=None,
            residual_head_gate_init=1e-3,
            residual_head_mlp_hidden_mult=4.0,
            residual_gate_init=1e-3,
            residual_mlp_hidden_mult=4.0):
        # Text-HR v2: build the small trainable bridge from frozen T5 hidden states
        # to the GigaTok transformer decoder width.
        decoder_width = self.s1to2decoder.width
        if text_projection == "linear":
            self.text_projection = nn.Linear(text_feature_dim, decoder_width)
        elif text_projection == "linear_layernorm":
            self.text_projection = nn.Sequential(
                nn.Linear(text_feature_dim, decoder_width),
                nn.LayerNorm(decoder_width),
            )
        elif text_projection == "identity":
            if text_feature_dim != decoder_width:
                raise ValueError(
                    f"identity text_projection requires text_feature_dim={text_feature_dim} "
                    f"to match decoder width={decoder_width}"
                )
            self.text_projection = nn.Identity()
        else:
            raise ValueError(f"Unknown text_projection: {text_projection}")

        if isinstance(self.text_projection, nn.Module):
            self.text_projection.apply(self._init_weights)

        if text_type_embedding:
            self.text_type_embedding = nn.Parameter(torch.zeros(1, 1, decoder_width))
            nn.init.trunc_normal_(self.text_type_embedding, mean=0.0, std=0.02)
        else:
            self.text_type_embedding = None

        if visual_type_embedding:
            self.visual_type_embedding = nn.Parameter(torch.zeros(1, 1, decoder_width))
        else:
            self.visual_type_embedding = None

        if text_gate_enabled:
            init = float(text_gate_init)
            if init <= 0.0 or init >= 1.0:
                raise ValueError(f"text_gate_init must be in (0, 1), got {text_gate_init}")
            self.text_gate_logit = nn.Parameter(torch.tensor(math.log(init / (1.0 - init))))
        else:
            self.text_gate_logit = None

        if visual_memory_mask_enabled:
            self.visual_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_width))
        else:
            self.visual_mask_token = None

        if text_recon_mode == "residual_head":
            hidden_dim = int(decoder_width * float(residual_head_mlp_hidden_mult))
            rec_spatial_channels = int(self.s1to2decoder.token_size)
            self.residual_head_mlp = nn.Sequential(
                nn.Linear(decoder_width, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, rec_spatial_channels),
            )
            self.residual_head_mlp.apply(self._init_weights)
            self.residual_head_gate = nn.Parameter(torch.tensor(float(residual_head_gate_init)))
        else:
            self.residual_head_mlp = None
            self.residual_head_gate = None

        if text_recon_mode == "residual_pooled_layer":
            hidden_dim = int(decoder_width * float(residual_mlp_hidden_mult))
            self.residual_text_mlp = nn.Sequential(
                nn.Linear(decoder_width, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, decoder_width),
            )
            self.residual_text_mlp.apply(self._init_weights)
            self.residual_gate = nn.Parameter(torch.tensor(float(residual_gate_init)))
        else:
            self.residual_text_mlp = None
            self.residual_gate = None

    def project_text_memory(self, decoder_text_features):
        # Text-HR v2: decoder_text_features are selected T5 layer features [B, T, d_t5].
        # This projects them to decoder memory tokens [B, T, d_dec].
        if decoder_text_features is None:
            return None
        if self.text_projection is None:
            raise RuntimeError("Text conditioning is enabled, but text_projection is not configured.")
        text_memory = self.text_projection(decoder_text_features)
        if self.text_type_embedding is not None:
            text_memory = text_memory + self.text_type_embedding.to(
                device=text_memory.device,
                dtype=text_memory.dtype,
            )
        return text_memory

    def project_text_memory_for_pooling(self, decoder_text_features):
        # Text reconstruction residual modes use projected T5 features directly:
        # no modality/type embedding and no concat text gate.
        if decoder_text_features is None:
            return None
        if self.text_projection is None:
            raise RuntimeError("Text conditioning is enabled, but text_projection is not configured.")
        return self.text_projection(decoder_text_features)

    @staticmethod
    def masked_mean_text(text_memory, text_key_padding_mask=None, eps=1e-6):
        if text_memory.dim() != 3:
            raise ValueError(f"text_memory must be [B, T, C], got {text_memory.shape}")
        if text_key_padding_mask is None:
            return text_memory.mean(dim=1)
        text_key_padding_mask = text_key_padding_mask.to(
            device=text_memory.device,
            dtype=torch.bool,
        )
        if text_key_padding_mask.shape != text_memory.shape[:2]:
            raise ValueError(
                f"text_key_padding_mask shape={text_key_padding_mask.shape}, "
                f"expected={text_memory.shape[:2]}"
            )
        valid = (~text_key_padding_mask).to(dtype=text_memory.dtype)
        denom = valid.sum(dim=1).clamp_min(float(eps))
        return (text_memory * valid.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)

    def _text_gate(self, device, dtype):
        if self.text_gate_logit is None:
            return torch.ones((), device=device, dtype=dtype)
        return torch.sigmoid(self.text_gate_logit.to(device=device, dtype=dtype))

    @staticmethod
    def _mean_token_norm(tensor):
        return tensor.float().norm(dim=-1).mean()

    def project_text_memory_by_layer(
            self,
            decoder_text_features_by_layer: Optional[Mapping[int, torch.Tensor]],
            selected_decoder_layer=None):
        if not decoder_text_features_by_layer:
            return None, {}
        if self.text_projection is None:
            raise RuntimeError("Text conditioning is enabled, but text_projection is not configured.")

        text_memory_by_layer: Dict[int, torch.Tensor] = {}
        before_norms = []
        after_norms = []
        selected_norm = None
        gate = None

        for decoder_layer, decoder_text_features in decoder_text_features_by_layer.items():
            text_memory = self.text_projection(decoder_text_features)
            if self.text_type_embedding is not None:
                text_memory = text_memory + self.text_type_embedding.to(
                    device=text_memory.device,
                    dtype=text_memory.dtype,
                )
            if gate is None:
                gate = self._text_gate(text_memory.device, text_memory.dtype)
            before_norm = self._mean_token_norm(text_memory)
            gated_text_memory = gate * text_memory
            after_norm = self._mean_token_norm(gated_text_memory)
            text_memory_by_layer[int(decoder_layer)] = gated_text_memory
            before_norms.append(before_norm)
            after_norms.append(after_norm)
            if selected_decoder_layer is not None and int(decoder_layer) == int(selected_decoder_layer):
                selected_norm = after_norm

        if gate is None:
            gate = torch.tensor(1.0)
        if selected_norm is None and after_norms:
            selected_norm = torch.stack(after_norms).mean()
        stats = {
            "text_gate": gate.float(),
            "text_memory_norm_before_gate_mean": torch.stack(before_norms).mean(),
            "text_memory_norm_after_gate_mean": torch.stack(after_norms).mean(),
            "selected_text_memory_norm": selected_norm,
        }
        return text_memory_by_layer, stats

    def project_residual_text_by_layer(
            self,
            decoder_text_features_by_layer: Optional[Mapping[int, torch.Tensor]],
            decoder_text_key_padding_mask=None):
        if not decoder_text_features_by_layer:
            return None, {}
        if self.residual_text_mlp is None or self.residual_gate is None:
            return None, {}
        if self.text_projection is None:
            raise RuntimeError("Text conditioning is enabled, but text_projection is not configured.")

        residual_text_by_layer: Dict[int, torch.Tensor] = {}
        residual_norms = []
        gate = None
        for decoder_layer, decoder_text_features in decoder_text_features_by_layer.items():
            text_memory = self.project_text_memory_for_pooling(decoder_text_features)
            pooled_text = self.masked_mean_text(text_memory, decoder_text_key_padding_mask)
            residual = self.residual_text_mlp(pooled_text)
            residual_text_by_layer[int(decoder_layer)] = residual
            residual_norms.append(residual.float().norm(dim=-1).mean())
            if gate is None:
                gate = self.residual_gate.to(device=residual.device, dtype=residual.dtype)

        if gate is None:
            gate = torch.tensor(0.0)
        stats = {
            "residual_gate": gate.float(),
            "residual_norm_mean": torch.stack(residual_norms).mean(),
        }
        return residual_text_by_layer, stats

    @staticmethod
    def _merge_text_recon_stats(text_stats, decoder_stats):
        if not text_stats and not decoder_stats:
            return {}
        stats = {}
        if text_stats:
            stats.update(text_stats)
        if decoder_stats:
            stats.update(decoder_stats)
        text_norm = stats.get("text_memory_norm_after_gate_mean", None)
        visual_norm = stats.get("visual_memory_norm_mean", None)
        if text_norm is not None and visual_norm is not None:
            stats["text_visual_norm_ratio"] = text_norm / visual_norm.clamp_min(1e-8)
        return stats

    def apply_residual_head(
            self,
            rec_spatial,
            decoder_head_text_features=None,
            decoder_text_key_padding_mask=None,
            return_text_recon_stats=False):
        if decoder_head_text_features is None:
            return rec_spatial, {}
        if self.residual_head_mlp is None or self.residual_head_gate is None:
            raise RuntimeError("decoder_head_text_features requires text_recon_conditioning.mode=residual_head.")

        expected_channels = int(self.s1to2decoder.token_size)
        assert rec_spatial.shape[1] == expected_channels, (
            f"residual_head expected rec_spatial channels={expected_channels}, "
            f"got rec_spatial.shape={tuple(rec_spatial.shape)}"
        )

        text_memory = self.project_text_memory_for_pooling(decoder_head_text_features)
        pooled_text = self.masked_mean_text(text_memory, decoder_text_key_padding_mask)
        text_vec = self.residual_head_mlp(pooled_text)
        text_spatial = text_vec[:, :, None, None]
        assert text_spatial.shape[1] == rec_spatial.shape[1], (
            f"residual_head text_spatial channels={text_spatial.shape[1]} do not match "
            f"rec_spatial channels={rec_spatial.shape[1]}"
        )

        text_spatial = text_spatial.to(device=rec_spatial.device, dtype=rec_spatial.dtype)
        gate = self.residual_head_gate.to(device=rec_spatial.device, dtype=rec_spatial.dtype)
        rec_spatial = rec_spatial + gate * text_spatial.expand_as(rec_spatial)

        if not return_text_recon_stats:
            return rec_spatial, {}

        stats = {
            "residual_head_gate": gate.float(),
            "residual_head_norm_mean": text_spatial.float().flatten(1).norm(dim=1).mean(),
            "residual_head_rec_spatial_channels": rec_spatial.new_tensor(float(rec_spatial.shape[1])),
            "residual_head_rec_spatial_height": rec_spatial.new_tensor(float(rec_spatial.shape[2])),
            "residual_head_rec_spatial_width": rec_spatial.new_tensor(float(rec_spatial.shape[3])),
            "residual_head_text_spatial_channels": rec_spatial.new_tensor(float(text_spatial.shape[1])),
            "residual_head_token_size": rec_spatial.new_tensor(float(expected_channels)),
        }
        return rec_spatial, stats
    
    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, 
               return_code=True, 
               return_feat=False, 
               return_cont_feat=False,      # return the feature before quantization
               return_fix_dim_feat=False,   # the fix dimension is the conv out before to codebook dim
               num_en_q_level=None, 
               causal_type=None,        # deprecated
               random_mix_reg=False,    # deprecated
               replace_ratio=None,      # deprecated
               global_step=None,        # prior model related, deprecated
               max_steps=None,          # prior model related, deprecated
               ):
        # causal_type = causal_type if causal_type is not None else self.config.causal_type
        if return_feat:
            assert (not return_code) and (not return_fix_dim_feat)
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level, 
                causal_type=causal_type, 
                return_feat=True)
            # return the feature of exactly the same width as the vit encoder
            return h, None, None
        
        if return_cont_feat:
            assert (not return_code) and (not return_fix_dim_feat)
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level,
                causal_type=causal_type,
                return_feat=False)
            # return the feature before quantization
            h = self.quant_conv(h)
            return h, None, None
        
        if return_fix_dim_feat:
            s = self.encoder(x)
            h = self.s2to1encoder(
                s, num_q_level=num_en_q_level,
                causal_type=causal_type,
                return_feat=False)
            # return the feature of the fix width (e.g. 256) before further downsampled to codebook dim
            return h, None, None


        s = self.encoder(x)
        h = self.s2to1encoder(s, num_q_level=num_en_q_level, causal_type=causal_type)
        # print("s shape:", s.shape)

        h = self.quant_conv(h)
        if self.training and self.config.with_prior_model:
            quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio,
                                                   global_step=global_step, max_steps=max_steps)
        else:
            quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio)

        if return_code:
            return quant, emb_loss, info
       
        return quant, emb_loss, s

    def decode(
            self, quant, 
            ret_inner_feat=False, # the feature passed through a MLP for alignment loss
            return_feat=False,    # the feature for linear probe
            selected_decoder_layer=None,
            decoder_text_features=None,
            decoder_text_features_by_layer=None,
            decoder_head_text_features=None,
            decoder_text_key_padding_mask=None,
            text_injection_layers=None,
            visual_memory_mask_enabled=False,
            visual_memory_mask_ratio=0.0,
            return_text_recon_stats=False,
            ):
        quant = self.post_quant_conv(quant)
        text_memory = self.project_text_memory(decoder_text_features)
        if text_memory is not None and selected_decoder_layer is None:
            raise ValueError("decoder_text_features requires selected_decoder_layer for text injection.")
        residual_text_by_layer, residual_text_stats = self.project_residual_text_by_layer(
            decoder_text_features_by_layer,
            decoder_text_key_padding_mask=decoder_text_key_padding_mask,
        )
        if residual_text_by_layer is None:
            text_memory_by_layer, text_stats = self.project_text_memory_by_layer(
                decoder_text_features_by_layer,
                selected_decoder_layer=selected_decoder_layer,
            )
            residual_gate = None
        else:
            text_memory_by_layer, text_stats = None, residual_text_stats
            residual_gate = self.residual_gate
        # Text-HR v2: only the selected decoder layer receives text_memory and
        # returns its post-softmax cross-attention weights for the HR loss.
        if ret_inner_feat:
            if selected_decoder_layer is not None:
                decoder_outputs = self.s1to2decoder(
                    quant,
                    ret_inner_feat=True,
                    selected_decoder_layer=selected_decoder_layer,
                    text_memory=text_memory,
                    text_memory_by_layer=text_memory_by_layer,
                    text_key_padding_mask=decoder_text_key_padding_mask,
                    text_injection_layers=text_injection_layers,
                    visual_type_embedding=self.visual_type_embedding,
                    visual_mask_token=self.visual_mask_token,
                    visual_memory_mask_enabled=visual_memory_mask_enabled,
                    visual_memory_mask_ratio=visual_memory_mask_ratio,
                    residual_text_by_layer=residual_text_by_layer,
                    residual_gate=residual_gate,
                    return_text_recon_stats=return_text_recon_stats,
                )
                if return_text_recon_stats:
                    rec_spatial, inner_feat, decoder_cross_attn, decoder_stats = decoder_outputs
                else:
                    rec_spatial, inner_feat, decoder_cross_attn = decoder_outputs
            else:
                decoder_outputs = self.s1to2decoder(
                    quant,
                    ret_inner_feat=True,
                    text_memory_by_layer=text_memory_by_layer,
                    text_key_padding_mask=decoder_text_key_padding_mask,
                    text_injection_layers=text_injection_layers,
                    visual_type_embedding=self.visual_type_embedding,
                    visual_mask_token=self.visual_mask_token,
                    visual_memory_mask_enabled=visual_memory_mask_enabled,
                    visual_memory_mask_ratio=visual_memory_mask_ratio,
                    residual_text_by_layer=residual_text_by_layer,
                    residual_gate=residual_gate,
                    return_text_recon_stats=return_text_recon_stats,
                )
                if return_text_recon_stats:
                    rec_spatial, inner_feat, decoder_stats = decoder_outputs
                else:
                    rec_spatial, inner_feat = decoder_outputs
                decoder_cross_attn = None
            rec_spatial, residual_head_stats = self.apply_residual_head(
                rec_spatial,
                decoder_head_text_features=decoder_head_text_features,
                decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                return_text_recon_stats=return_text_recon_stats,
            )
            pixel_dec = self.decoder(rec_spatial)
            text_recon_stats = self._merge_text_recon_stats(
                text_stats,
                decoder_stats if return_text_recon_stats else None,
            )
            if residual_head_stats:
                text_recon_stats.update(residual_head_stats)
            if selected_decoder_layer is not None:
                if return_text_recon_stats:
                    return pixel_dec, rec_spatial, inner_feat, decoder_cross_attn, text_recon_stats
                return pixel_dec, rec_spatial, inner_feat, decoder_cross_attn
            if return_text_recon_stats:
                return pixel_dec, rec_spatial, inner_feat, text_recon_stats
            return pixel_dec, rec_spatial, inner_feat
        elif return_feat:
            # specifically for linear probe
            _, inner_feat = self.s1to2decoder(quant, return_feat=True)
            # pixel_dec = self.decoder(rec_spatial)
            return None, None, inner_feat
        else:
            if selected_decoder_layer is not None:
                decoder_outputs = self.s1to2decoder(
                    quant,
                    selected_decoder_layer=selected_decoder_layer,
                    text_memory=text_memory,
                    text_memory_by_layer=text_memory_by_layer,
                    text_key_padding_mask=decoder_text_key_padding_mask,
                    text_injection_layers=text_injection_layers,
                    visual_type_embedding=self.visual_type_embedding,
                    visual_mask_token=self.visual_mask_token,
                    visual_memory_mask_enabled=visual_memory_mask_enabled,
                    visual_memory_mask_ratio=visual_memory_mask_ratio,
                    residual_text_by_layer=residual_text_by_layer,
                    residual_gate=residual_gate,
                    return_text_recon_stats=return_text_recon_stats,
                )
                if return_text_recon_stats:
                    rec_spatial, decoder_cross_attn, decoder_stats = decoder_outputs
                else:
                    rec_spatial, decoder_cross_attn = decoder_outputs
            else:
                decoder_outputs = self.s1to2decoder(
                    quant,
                    text_memory_by_layer=text_memory_by_layer,
                    text_key_padding_mask=decoder_text_key_padding_mask,
                    text_injection_layers=text_injection_layers,
                    visual_type_embedding=self.visual_type_embedding,
                    visual_mask_token=self.visual_mask_token,
                    visual_memory_mask_enabled=visual_memory_mask_enabled,
                    visual_memory_mask_ratio=visual_memory_mask_ratio,
                    residual_text_by_layer=residual_text_by_layer,
                    residual_gate=residual_gate,
                    return_text_recon_stats=return_text_recon_stats,
                )
                if return_text_recon_stats:
                    rec_spatial, decoder_stats = decoder_outputs
                else:
                    rec_spatial = decoder_outputs
                decoder_cross_attn = None
            rec_spatial, residual_head_stats = self.apply_residual_head(
                rec_spatial,
                decoder_head_text_features=decoder_head_text_features,
                decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                return_text_recon_stats=return_text_recon_stats,
            )
            pixel_dec = self.decoder(rec_spatial)
            text_recon_stats = self._merge_text_recon_stats(
                text_stats,
                decoder_stats if return_text_recon_stats else None,
            )
            if residual_head_stats:
                text_recon_stats.update(residual_head_stats)
            if selected_decoder_layer is not None:
                if return_text_recon_stats:
                    return pixel_dec, rec_spatial, decoder_cross_attn, text_recon_stats
                return pixel_dec, rec_spatial, decoder_cross_attn
            if return_text_recon_stats:
                return pixel_dec, rec_spatial, text_recon_stats
            return pixel_dec, rec_spatial

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec, rec_spatial = self.decode(quant_b)
        return dec

    def forward(
            self, 
            input, 
            num_en_q_level=None, 
            causal_type=None, 
            rec_loss=True, 
            ret_inner_feat=False,
            random_mix_reg=False,
            replace_ratio=None,
            global_step=None,
            max_steps=None,
            selected_decoder_layer=None,
            decoder_text_features=None,
            decoder_text_features_by_layer=None,
            decoder_head_text_features=None,
            decoder_text_key_padding_mask=None,
            text_injection_layers=None,
            visual_memory_mask_enabled=False,
            visual_memory_mask_ratio=0.0,
            return_text_recon_stats=False,
            ):
        # Text-HR v2: selected_decoder_layer / decoder_text_features keep the
        # original image-only path unchanged when they are None.
        quant, diff, spatial = self.encode(
                                    input, 
                                    return_code=False, 
                                    num_en_q_level=num_en_q_level, 
                                    causal_type=causal_type,
                                    random_mix_reg=random_mix_reg,
                                    replace_ratio=replace_ratio,
                                    global_step=global_step,
                                    max_steps=max_steps
                                    )
        if ret_inner_feat:
            if self.config.encoder_2d_distill:
                inner_feat = rearrange(spatial, 'b c h w -> b (h w) c')
                inner_feat = self.distill_mlp(inner_feat)
                if selected_decoder_layer is not None:
                    decode_outputs = self.decode(
                        quant,
                        selected_decoder_layer=selected_decoder_layer,
                        decoder_text_features=decoder_text_features,
                        decoder_text_features_by_layer=decoder_text_features_by_layer,
                        decoder_head_text_features=decoder_head_text_features,
                        decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                        text_injection_layers=text_injection_layers,
                        visual_memory_mask_enabled=visual_memory_mask_enabled,
                        visual_memory_mask_ratio=visual_memory_mask_ratio,
                        return_text_recon_stats=return_text_recon_stats,
                    )
                    if return_text_recon_stats:
                        dec, rec_spatial, decoder_cross_attn, text_recon_stats = decode_outputs
                    else:
                        dec, rec_spatial, decoder_cross_attn = decode_outputs
                else:
                    decode_outputs = self.decode(
                        quant,
                        decoder_text_features_by_layer=decoder_text_features_by_layer,
                        decoder_head_text_features=decoder_head_text_features,
                        decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                        text_injection_layers=text_injection_layers,
                        visual_memory_mask_enabled=visual_memory_mask_enabled,
                        visual_memory_mask_ratio=visual_memory_mask_ratio,
                        return_text_recon_stats=return_text_recon_stats,
                    )
                    if return_text_recon_stats:
                        dec, rec_spatial, text_recon_stats = decode_outputs
                    else:
                        dec, rec_spatial = decode_outputs
                    decoder_cross_attn = None
            else:
                if selected_decoder_layer is not None:
                    decode_outputs = self.decode(
                        quant,
                        ret_inner_feat=True,
                        selected_decoder_layer=selected_decoder_layer,
                        decoder_text_features=decoder_text_features,
                        decoder_text_features_by_layer=decoder_text_features_by_layer,
                        decoder_head_text_features=decoder_head_text_features,
                        decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                        text_injection_layers=text_injection_layers,
                        visual_memory_mask_enabled=visual_memory_mask_enabled,
                        visual_memory_mask_ratio=visual_memory_mask_ratio,
                        return_text_recon_stats=return_text_recon_stats,
                    )
                    if return_text_recon_stats:
                        dec, rec_spatial, inner_feat, decoder_cross_attn, text_recon_stats = decode_outputs
                    else:
                        dec, rec_spatial, inner_feat, decoder_cross_attn = decode_outputs
                else:
                    decode_outputs = self.decode(
                        quant,
                        ret_inner_feat=True,
                        decoder_text_features_by_layer=decoder_text_features_by_layer,
                        decoder_head_text_features=decoder_head_text_features,
                        decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                        text_injection_layers=text_injection_layers,
                        visual_memory_mask_enabled=visual_memory_mask_enabled,
                        visual_memory_mask_ratio=visual_memory_mask_ratio,
                        return_text_recon_stats=return_text_recon_stats,
                    )
                    if return_text_recon_stats:
                        dec, rec_spatial, inner_feat, text_recon_stats = decode_outputs
                    else:
                        dec, rec_spatial, inner_feat = decode_outputs
                    decoder_cross_attn = None
        else:
            if selected_decoder_layer is not None:
                decode_outputs = self.decode(
                    quant,
                    selected_decoder_layer=selected_decoder_layer,
                    decoder_text_features=decoder_text_features,
                    decoder_text_features_by_layer=decoder_text_features_by_layer,
                    decoder_head_text_features=decoder_head_text_features,
                    decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                    text_injection_layers=text_injection_layers,
                    visual_memory_mask_enabled=visual_memory_mask_enabled,
                    visual_memory_mask_ratio=visual_memory_mask_ratio,
                    return_text_recon_stats=return_text_recon_stats,
                )
                if return_text_recon_stats:
                    dec, rec_spatial, decoder_cross_attn, text_recon_stats = decode_outputs
                else:
                    dec, rec_spatial, decoder_cross_attn = decode_outputs
            else:
                decode_outputs = self.decode(
                    quant,
                    decoder_text_features_by_layer=decoder_text_features_by_layer,
                    decoder_head_text_features=decoder_head_text_features,
                    decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                    text_injection_layers=text_injection_layers,
                    visual_memory_mask_enabled=visual_memory_mask_enabled,
                    visual_memory_mask_ratio=visual_memory_mask_ratio,
                    return_text_recon_stats=return_text_recon_stats,
                )
                if return_text_recon_stats:
                    dec, rec_spatial, text_recon_stats = decode_outputs
                else:
                    dec, rec_spatial = decode_outputs
                decoder_cross_attn = None

        if self.training:
            if rec_loss:
                if self.config.fea_rec_loss_type == "cosine":
                    fea_rec_loss = self.config.fea_rec_loss_weight * compute_cosinesim_loss(spatial.detach(), rec_spatial, 1)
                elif self.config.fea_rec_loss_type == "mse":
                    fea_rec_loss = self.config.fea_rec_loss_weight * F.mse_loss(spatial.detach(), rec_spatial)
            else:
                fea_rec_loss = 0

        if self.training:
            if rec_loss:
                dir_dec = self.decoder(spatial)
            else:
                dir_dec = None
            
            if ret_inner_feat:
                if selected_decoder_layer is not None:
                    if return_text_recon_stats:
                        return [dec, dir_dec], [diff, fea_rec_loss], inner_feat, decoder_cross_attn, text_recon_stats
                    return [dec, dir_dec], [diff, fea_rec_loss], inner_feat, decoder_cross_attn
                if return_text_recon_stats:
                    return [dec, dir_dec], [diff, fea_rec_loss], inner_feat, text_recon_stats
                return [dec, dir_dec], [diff, fea_rec_loss], inner_feat
            if selected_decoder_layer is not None:
                if return_text_recon_stats:
                    return [dec, dir_dec], [diff, fea_rec_loss], decoder_cross_attn, text_recon_stats
                return [dec, dir_dec], [diff, fea_rec_loss], decoder_cross_attn
            if return_text_recon_stats:
                return [dec, dir_dec], [diff, fea_rec_loss], text_recon_stats
            return [dec, dir_dec], [diff, fea_rec_loss]

        if selected_decoder_layer is not None:
            if return_text_recon_stats:
                return dec, diff, decoder_cross_attn, text_recon_stats
            return dec, diff, decoder_cross_attn
        if return_text_recon_stats:
            return dec, diff, text_recon_stats
        return dec, diff



@dataclass
class VQVitModel2DPlusArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True
    codebook_show_usage: bool = True
    commit_loss_beta: float = 0.25
    entropy_loss_ratio: float = 0.0
    
    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    model_size: str = 'small'
    num_latent_tokens: int = 256
    encoder_size: str = None 
    decoder_size: str = None
    transformer_layer_type: str = "TransformerDecoderLayer"
    z_channels: int = 256
    dropout_p: float = 0.0

    adaptive_gn: bool = False
    d2s_up: bool = False

    rot: bool = False
    distill_depth: int = None

    encoder_2d_distill: bool = False

    # for semantic distillation regularization
    # the default 768 is for dino-v2 base
    out_inner_dim: int = 768

    fea_rec_loss_type: str = "cosine"
    fea_rec_loss_weight: float = 1.0
    use_attn: bool = True


class VQVitModel2DPlus(nn.Module):
    def __init__(self, config: VQVitModelPlusArgs):
        super().__init__()
        self.config = config
        self.encoder = Encoder(
                        ch_mult=config.encoder_ch_mult, 
                        z_channels=config.z_channels, 
                        dropout=config.dropout_p, 
                        use_attn=config.use_attn,
                        )

        if config.encoder_2d_distill:
            self.distill_mlp = nn.Sequential(
                    nn.Linear(config.z_channels, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.z_channels * 4),
                    nn.SiLU(),
                    nn.Linear(config.z_channels * 4, config.out_inner_dim),
                    )


        if config.encoder_size is not None:
            encoder_size = config.encoder_size
        else:
            encoder_size = config.model_size
        
        if config.decoder_size is not None:
            decoder_size = config.decoder_size
        else:
            decoder_size = config.model_size
        
        # when encoder size or decoder size is given, model size should be none
        if config.encoder_size is not None or config.decoder_size is not None:
            assert config.model_size is None
        
        if config.encoder_2d_distill:
            assert config.distill_depth is None



        self.s2dencoder = ViTEncoder2D(
            model_size=encoder_size, 
            token_size=config.z_channels, dropout=config.dropout_p, 
            patch_size=2**(len(config.encoder_ch_mult) - 1),
            transformer_layer_type=config.transformer_layer_type,
            )

        self.s2ddecoder = ViTDecoder2D(
            model_size=decoder_size,
            token_size=config.z_channels, dropout=config.dropout_p,
            patch_size=2**(len(config.decoder_ch_mult) - 1),
            out_inner_feat=config.distill_depth is not None,
            out_inner_depth=config.distill_depth,
            out_inner_dim=config.out_inner_dim,
            transformer_layer_type=config.transformer_layer_type,
            )

        self.decoder = Decoder(ch_mult=config.decoder_ch_mult, 
            z_channels=config.z_channels, 
            dropout=config.dropout_p,
            adaptive_gn=config.adaptive_gn,
            d2s_up=config.d2s_up,
            use_attn=config.use_attn
            )

        self.apply(self._init_weights)

        self.quantize = VectorQuantizer(config.codebook_size, config.codebook_embed_dim, 
                                        config.commit_loss_beta, config.entropy_loss_ratio,
                                        config.codebook_l2_norm, config.codebook_show_usage,
                                        rot=config.rot
                                        )
        self.quant_conv = nn.Conv2d(self.config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(config.codebook_embed_dim, self.config.z_channels, 1)

        def nan_hook(self, inp, output):
            if not isinstance(output, torch.Tensor):
                return
            if torch.isnan(output).any():
                print(f"NaN detected in {self}")
                raise RuntimeError("NaN detected")

        for name, module in self.named_modules():
            module.register_forward_hook(nan_hook)
    
    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x, 
               return_code=True, 
               return_feat=False, 
               random_mix_reg=False,
               replace_ratio=0.1,
               **kwargs
               ):
        # causal_type = causal_type if causal_type is not None else self.config.causal_type
        if return_feat:
            s = self.encoder(x)
            h = self.s2dencoder(s, return_feat=True)
            # return the feature before quantization
            return h, None, None

        s = self.encoder(x)
        h = self.s2dencoder(s)
        # print("s shape:", s.shape)

        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, random_replace=random_mix_reg, replace_ratio=replace_ratio)

        if return_code:
            return quant, emb_loss, info
       
        return quant, emb_loss, s

    def decode(self, quant, ret_inner_feat=False, return_feat=False):
        quant = self.post_quant_conv(quant)
        if ret_inner_feat:
            rec_spatial, inner_feat = self.s2ddecoder(quant, ret_inner_feat=True)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial, inner_feat
        elif return_feat:
            # specifically for linear probe or visualization (don not go through mlp)
            _, feat = self.s2ddecoder(quant, return_feat=True)
            # pixel_dec = self.decoder(rec_spatial)
            return _, feat
        else:
            rec_spatial = self.s2ddecoder(quant)
            pixel_dec = self.decoder(rec_spatial)
            return pixel_dec, rec_spatial

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec, rec_spatial = self.decode(quant_b)
        return dec

    def forward(
            self, 
            input, 
            num_en_q_level=None, 
            causal_type=None, 
            rec_loss=True, 
            ret_inner_feat=False,
            random_mix_reg=False,
            replace_ratio=None,
            global_step=None,
            max_steps=None,
            ):
        quant, diff, spatial = self.encode(
                                    input, 
                                    return_code=False, 
                                    random_mix_reg=random_mix_reg,
                                    replace_ratio=replace_ratio
                                    )
        if ret_inner_feat:
            if self.config.encoder_2d_distill:
                inner_feat = rearrange(spatial, 'b c h w -> b (h w) c')
                inner_feat = self.distill_mlp(inner_feat)
                dec, rec_spatial = self.decode(quant)
            else:
                dec, rec_spatial, inner_feat = self.decode(quant, ret_inner_feat=True)
        else:
            dec, rec_spatial = self.decode(quant)
        # if torch.isnan(dec).any():
        #     print("nan in dec")
        # if torch.isnan(rec_spatial).any():
        #     print("nan in rec_spatial")

        if self.training:
            if rec_loss:
                if self.config.fea_rec_loss_type == "cosine":
                    fea_rec_loss = self.config.fea_rec_loss_weight * compute_cosinesim_loss(spatial.detach(), rec_spatial, 1)
                elif self.config.fea_rec_loss_type == "mse":
                    fea_rec_loss = self.config.fea_rec_loss_weight * F.mse_loss(spatial.detach(), rec_spatial)
            else:
                fea_rec_loss = 0

        if self.training:
            if rec_loss:
                dir_dec = self.decoder(spatial)
            else:
                dir_dec = None
            
            if ret_inner_feat:
                return [dec, dir_dec], [diff, fea_rec_loss], inner_feat
            return [dec, dir_dec], [diff, fea_rec_loss]

        return dec, diff




class VectorQuantizer(nn.Module):
    def __init__(
            self, 
            n_e, 
            e_dim, 
            beta, 
            entropy_loss_ratio, 
            l2_norm, 
            show_usage, 
            rot=False,
            stochastic=False,
            stochastic_temperature=1.0,
            eval_deterministic=False,
            simvq=False,
            codebook_transform=None,
            freeze_codebook=False,
            ):
        """
        Args:
            n_e: the size of the codebook
            e_dim: the dimension of the codebook vectors
            beta: the commitment loss weight
            entropy_loss_ratio: the ratio of the entropy loss to the commitment loss
            l2_norm: whether to normalize the codebook vectors
            show_usage: whether to show the usage of the codebook vectors
            rot: whether to use rotation trick
            stochastic: whether to use stochastic quantization
            stochastic_temperature: the temperature of the stochastic quantization
            eval_deterministic: whether to use deterministic quantization in evaluation mode
            simvq: whether to use simvq https://arxiv.org/abs/2411.02038
            codebook_transform: the transform to apply to the codebook vectors,
                choices from [ None, "linear", "mlp"]
            freeze_codebook: whether to freeze the codebook vectors
        """

        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.rot = rot
        self.stochastic = stochastic
        self.eval_deterministic = eval_deterministic

        self.simvq = simvq
        self.codebook_transform = codebook_transform
        self.freeze_codebook = freeze_codebook


        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

        if self.stochastic:
            if stochastic_temperature > 0: # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else: # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))
        
        if self.simvq:
            if codebook_transform == "linear":
                codebook_transform = nn.Linear(self.e_dim, self.e_dim, bias=False)
            elif codebook_transform == "mlp":
                codebook_transform = nn.Sequential(
                    nn.Linear(self.e_dim, self.e_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.e_dim * 4, self.e_dim),
                )
            else:
                raise ValueError("codebook_transform: {} Not Acceptable".format(codebook_transform))
            self.codebook_transform = codebook_transform

            if self.freeze_codebook:
                self.embedding.weight.requires_grad = False

    def get_emb(self):
        if self.simvq:
            return self.codebook_transform(self.embedding.weight)
        else:
            return self.embedding.weight

    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        # from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
        torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    
    def forward(self, z, random_replace=False, replace_ratio=0.1):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        if self.stochastic:
            # sample the softmaxed cosine similarity
            # reference: LARP
            assert self.l2_norm, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, embedding)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                min_encoding_indices = torch.argmax(probs, dim=-1)

            else:
                min_encoding_indices = torch.multinomial(probs, 1)
                min_encoding_indices = min_encoding_indices.squeeze(-1)
        else:
            # look up by l2 distance, argmin
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

            min_encoding_indices = torch.argmin(d, dim=1)   # (b*h*w)

        z_q = embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            if self.entropy_loss_ratio > 0:
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            else:
                entropy_loss = 0

        b, h, w, c = z.shape
        if self.rot:
            # adapted from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
            b, h, w, c = z.shape
            z = z / torch.norm(z, dim=-1, keepdim=True)
            # assert self.l2_norm, "Rot requires l2 normalization"
            z = rearrange(z, 'b h w c-> (b h w) c')
            z_q= rearrange(z_q, 'b h w c -> (b h w) c')
            pre_norm_q = self.get_very_efficient_rotation(z / (torch.norm(z, dim=1, keepdim=True) + 1e-6),
                                                            z_q / (torch.norm(z_q, dim=1, keepdim=True) + 1e-6),
                                                            z.unsqueeze(1)).squeeze()
            z_q = pre_norm_q * (
                    torch.norm(z_q, dim=1, keepdim=True) / (torch.norm(z, dim=1, keepdim=True) + 1e-6)).detach()
            z_q = rearrange(z_q, '(b h w) c -> b h w c', b=b, h=h, w=w)
        else:
            # preserve gradients
            z_q = z + (z_q - z).detach()

        if random_replace and self.training:
            # randomly replace the quantized vectors with the continuous input
            z = rearrange(z, '(b h w) c -> b h w c', b=b, h=h, w=w)
            mask = torch.bernoulli(torch.full(z.shape[:-1], replace_ratio)).unsqueeze(-1).to(z.device)  # replace_ratio chance of replacement
            z_q = torch.where(mask.bool(), z, z_q)

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, [vq_loss, commit_loss, entropy_loss, codebook_usage], (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q



class VectorQuantizerWithPM(nn.Module):
    def __init__(
            self, 
            n_e, 
            e_dim, 
            beta, 
            entropy_loss_ratio, 
            l2_norm, 
            show_usage, 
            rot=False,
            stochastic=False,
            stochastic_temperature=1.0,
            eval_deterministic=False,
            simvq=False,
            codebook_transform=None,
            freeze_codebook=False,
            prior_model_config=None
            ):
        """
        Args:
            n_e: the size of the codebook
            e_dim: the dimension of the codebook vectors
            beta: the commitment loss weight
            entropy_loss_ratio: the ratio of the entropy loss to the commitment loss
            l2_norm: whether to normalize the codebook vectors
            show_usage: whether to show the usage of the codebook vectors
            rot: whether to use rotation trick
            stochastic: whether to use stochastic quantization
            stochastic_temperature: the temperature of the stochastic quantization
            eval_deterministic: whether to use deterministic quantization in evaluation mode
            simvq: whether to use simvq https://arxiv.org/abs/2411.02038
            codebook_transform: the transform to apply to the codebook vectors,
                choices from [ None, "linear", "mlp"]
            freeze_codebook: whether to freeze the codebook vectors
            prior_model_config: the config for the prior model

        - prior ar model for ntp regularization
            - returns the prior model loss along with other codebook loss
        """
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage
        self.rot = rot
        self.stochastic = stochastic
        self.prior_model_config = prior_model_config
        self.eval_deterministic = eval_deterministic

        self.simvq = simvq
        self.codebook_transform = codebook_transform
        self.freeze_codebook = freeze_codebook

        # prior model training config 
        if prior_model_config is not None:
            self.prior_n_rounds = prior_model_config["train_args"]["n_rounds"]
            self.prior_no_grad_before_last_round = prior_model_config["train_args"]["no_grad_before_last_round"]
            self.prior_avg_loss_over_rounds = prior_model_config["train_args"]["avg_loss_over_rounds"]
            self.use_mix_ss = prior_model_config["train_args"]["use_mix_ss"]
            self.mix_ss_max_ratio = prior_model_config["train_args"]["mix_ss_max_ratio"]
            self.mix_ss_peak_steps_ratio = prior_model_config["train_args"]["mix_ss_peak_steps_ratio"]
            self.prior_latent_ce_temperature = prior_model_config["train_args"].get("latent_ce_temperature", 1.0)
 

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(self.embedding.weight.data, p=2, dim=-1)
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

        if self.stochastic:
            if stochastic_temperature > 0: # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else: # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        if prior_model_config is None:
            self.prior_model = None
        else:
            prior_model_additional_args = {
                'n_ind': self.e_dim, 
                'n_classes': self.n_e
            }

            self.ar_prior_loss_weight = prior_model_config["train_args"].get('prior_loss_weight', 0.06)
            if prior_model_config["train_args"].get('no_dropout', False):
                prior_model_additional_args['embd_pdrop'] = 0.0
                prior_model_additional_args['resid_pdrop'] = 0.0
                prior_model_additional_args['attn_pdrop'] = 0.0
                print(f"Warning: prior_loss is using no dropout")
            
            # initialize
            self.prior_model = GPTC_models[self.prior_model_config['name']](
                    **prior_model_config['init_args'], 
                    **prior_model_additional_args
                )

        if self.simvq:
            if codebook_transform == "linear":
                codebook_transform = nn.Linear(self.e_dim, self.e_dim, bias=False)
            elif codebook_transform == "mlp":
                codebook_transform = nn.Sequential(
                    nn.Linear(self.e_dim, self.e_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.e_dim * 4, self.e_dim),
                )
            else:
                raise ValueError("codebook_transform: {} Not Acceptable".format(codebook_transform))
            self.codebook_transform = codebook_transform

            if self.freeze_codebook:
                self.embedding.weight.requires_grad = False

        
    def get_emb(self):
        if self.simvq:
            return self.codebook_transform(self.embedding.weight)
        else:
            return self.embedding.weight


    @staticmethod
    def get_very_efficient_rotation(u, q, e):
        # from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
        w = ((u + q) / torch.norm(u + q, dim=1, keepdim=True)).detach()
        e = e - 2 * torch.bmm(torch.bmm(e, w.unsqueeze(-1)), w.unsqueeze(1)) + 2 * torch.bmm(
        torch.bmm(e, u.unsqueeze(-1).detach()), q.unsqueeze(1).detach())
        return e

    def logits_to_token_embedding_with_ss(
            self, 
            logits, 
            ar_input_staring_from_idx_1, 
            global_step,
            max_steps,
            mask=None):
        """
        adapted from https://github.com/hywang66/LARP/
        """
        # logits: (b, n - 1, codebook_size), sequence index from 1 to n-1 (inclusive)
        # ar_input_staring_from_idx_1: (b, n - 1, d=16), requires_grad=True
        if mask is None:
            b, n_minus_1, _ = logits.size()
            if self.use_mix_ss:
                ss_ratio = (global_step / (max_steps * self.mix_ss_peak_steps_ratio )) * self.mix_ss_max_ratio
                ss_ratio = min(ss_ratio, self.mix_ss_max_ratio)
            else:
                ss_ratio = 1.0

            mask = torch.rand(b, n_minus_1, 1, device=logits.device) < ss_ratio
            mask = mask.expand(-1, -1, self.e_dim) # (b, n - 1, d=16)

        with torch.autocast(device_type='cuda', enabled=False):
            logits = logits.float()
            probs = F.softmax(logits, dim=-1) # (b, n - 1, codebook_size)
            indices = torch.multinomial(probs.view(-1, self.n_e), 1).view(*probs.size()[:-1]) # (b, n - 1)
        token_embedding = F.embedding(indices, self.get_emb()) # (b, n - 1, d=16)
        token_embedding = torch.where(mask, token_embedding, ar_input_staring_from_idx_1)

        return token_embedding

    def calculate_logits_and_ar_pred_cont(self, prior_model_output):
        ar_pred_cont = prior_model_output # (b, n, d=16)
        # the prior_model_output and the embedding should have been normalized (-1 dim)
        logits = F.linear(prior_model_output, self.get_emb())[:, 1:]
        logits = logits.mul_(1 / self.prior_latent_ce_temperature)
        logits = logits.contiguous() # (b, n - 1, codebook_size)
        return logits, ar_pred_cont


    def prior_ar_predict_n_rounds_ss(
            self, 
            ar_input, 
            global_step,
            max_steps,
        ):
        """
        adapted from https://github.com/hywang66/LARP/
        """
        prior_model = self.prior_model
        n_rounds = self.prior_n_rounds
        no_grad_before_last_round = self.prior_no_grad_before_last_round

        b, n, _ = ar_input.size()
        n_minus_1 = n - 1
        if self.use_mix_ss:
            peak_steps_ratio = torch.tensor(self.mix_ss_peak_steps_ratio, dtype=torch.float32)
            max_ratio = torch.tensor(self.mix_ss_max_ratio, dtype=torch.float32)

            ss_ratio = (global_step / (max_steps * peak_steps_ratio)) * max_ratio
            ss_ratio = torch.min(ss_ratio, max_ratio)
        else:
            ss_ratio = torch.tensor(1.0, dtype=torch.float32)

        mask_ss = torch.rand(b, n_minus_1, 1, device=ar_input.device) < ss_ratio
        mask_ss = mask_ss.expand(-1, -1, self.e_dim) # (b, n - 1, d=16)

        logits_all_rounds = []
        next_ar_input = ar_input # (b, n, d=16)
        for i in range(n_rounds):
            if no_grad_before_last_round and i < n_rounds - 1:
                # we can not use "with torch.no_grad()" here due to a pytorch's bug!
                # https://github.com/pytorch/pytorch/issues/112583
                prior_model.requires_grad_(False)
                prior_model_output = prior_model.ar_predict(next_ar_input.detach()) # (b, n - 1, codebook_size)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)
                prior_model.requires_grad_(True)
            else:
                prior_model_output = prior_model.ar_predict(next_ar_input) # (b, n, d=16)(1 orig + n - 1 pred)
                logits, ar_pred_cont = self.calculate_logits_and_ar_pred_cont(prior_model_output)   # (b, n - 1, codebook_size)
                logits_all_rounds.append(logits)


            if i < n_rounds - 1:
                token_embedding = self.logits_to_token_embedding_with_ss(
                                            logits, 
                                            ar_input[:, 1:], 
                                            global_step=global_step,
                                            max_steps=max_steps,
                                            mask=mask_ss) # (b, n - 1, d=16)
                next_ar_input = torch.cat([ar_input[:, :1], token_embedding], dim=1) # (b, n, d=16)

        if self.prior_avg_loss_over_rounds:
            logits_all_rounds = torch.stack(logits_all_rounds, dim=0) # (n_rounds, b, n - 1, codebook_size)

        else:
            logits_all_rounds = torch.stack([logits_all_rounds[-1]], dim=0) # (1, b, n - 1, codebook_size)

        return logits_all_rounds, ar_pred_cont, next_ar_input # here the next_ar_input is actually the last round's ar_input

    def calculate_prior_loss_with_pred(
            self, 
            encode_output, 
            indices,
            global_step, 
            max_steps,
            return_sampled_indices=False,
            sample_temperature=1.0,
        ):
        """
        adapted from https://github.com/hywang66/LARP/
        """
        B = encode_output.size(0)
        ar_input = encode_output # (b, n, d) normalized
        labels = indices[:, 1:].contiguous() # (b, n - 1)
        logits_all_rounds, ar_pred_cont, regularized_z_ss = self.prior_ar_predict_n_rounds_ss(
                                                                    ar_input, 
                                                                    global_step=global_step, 
                                                                    max_steps=max_steps,
                                                                ) # regularized_z_ss: (b, n, d=16)
        labels_all_rounds = labels.unsqueeze(0).expand(logits_all_rounds.size(0), -1, -1).contiguous() # (n_rounds or 1, b, n - 1)
        
        loss_latent_ce = F.cross_entropy(logits_all_rounds.view(-1, self.n_e), labels_all_rounds.view(-1))
        # return_dict['loss_latent_ce'] = loss_latent_ce
        # topk_accuracies = utils.calculate_topk_accuracy(logits_all_rounds[0], labels, topk=(1, 5), prepend='prior_')
        # return_dict.update(topk_accuracies)

        if return_sampled_indices:
            # sample the indices from the last round prediction because it is closer
            # to the downstream gpt prediction error pattern
            sampled_indices = torch.multinomial(F.softmax(logits_all_rounds[-1] / sample_temperature, dim=-1), 1).squeeze(-1)


        return loss_latent_ce 


    
    def forward(
            self, 
            z, 
            max_steps=None, # for pm training
            global_step=None,
            random_replace=False, 
            replace_ratio=0.1,
            ):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum('b c h w -> b h w c', z).contiguous()
        b, h, w, c = z.shape
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()


        if self.stochastic:
            # sample the softmaxed cosine similarity
            # reference: LARP
            assert self.l2_norm, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, embedding)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                min_encoding_indices = torch.argmax(probs, dim=-1)
            else:
                min_encoding_indices = torch.multinomial(probs, 1)
                min_encoding_indices = min_encoding_indices.squeeze(-1)
        else:
            # look up by l2 distance, argmin
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, torch.einsum('n d -> d n', embedding))

            min_encoding_indices = torch.argmin(d, dim=1)

        z_q = embedding[min_encoding_indices].view(z.shape)

        perplexity = None
        min_encodings = None
        vq_loss = None
        ar_prior_loss = None
        commit_loss = None
        entropy_loss = None
        codebook_usage = 0

        if self.show_usage and self.training:
            cur_len = min_encoding_indices.shape[0]
            self.codebook_used[:-cur_len] = self.codebook_used[cur_len:].clone()
            self.codebook_used[-cur_len:] = min_encoding_indices
            codebook_usage = len(torch.unique(self.codebook_used)) / self.n_e

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2) 
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) 
            if self.entropy_loss_ratio > 0:
                entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)
            else:
                entropy_loss = 0

        if self.rot:
            # adapted from https://github.com/cfifty/rotation_trick/blob/main/src/models/vq_vae.py
            b, h, w, c = z.shape
            z = z / torch.norm(z, dim=-1, keepdim=True)
            # assert self.l2_norm, "Rot requires l2 normalization"
            z = rearrange(z, 'b h w c-> (b h w) c')
            z_q= rearrange(z_q, 'b h w c -> (b h w) c')
            pre_norm_q = self.get_very_efficient_rotation(z / (torch.norm(z, dim=1, keepdim=True) + 1e-6),
                                                            z_q / (torch.norm(z_q, dim=1, keepdim=True) + 1e-6),
                                                            z.unsqueeze(1)).squeeze()
            z_q = pre_norm_q * (
                    torch.norm(z_q, dim=1, keepdim=True) / (torch.norm(z, dim=1, keepdim=True) + 1e-6)).detach()
            z_q = rearrange(z_q, '(b h w) c -> b h w c', b=b, h=h, w=w)
        else:
            # preserve gradients
            z_q = z + (z_q - z).detach()
        
        if self.prior_model is not None and self.training:
            # ar prior training must be put after straight-through estimator, so that the gradients can be backpropagated
            assert global_step is not None and max_steps is not None, \
                "global_step and max_steps must be provided when using prior model"
            # when quantizing, there are only 2 dimensions, now change back to 
            # B N C for AR trianing
            min_indices = rearrange(min_encoding_indices, '(b n) -> b n', b=b)
            ar_prior_loss = self.ar_prior_loss_weight * self.calculate_prior_loss_with_pred(
                                rearrange(z_q, 'b h w c -> b (h w) c'),
                                indices=min_indices,
                                global_step=global_step, 
                                max_steps=max_steps
                                )
        else:
            ar_prior_loss = None


        if random_replace and self.training:
            # randomly replace the quantized vectors with the continuous input
            z = rearrange(z, '(b h w) c -> b h w c', b=b, h=h, w=w)
            mask = torch.bernoulli(torch.full(z.shape[:-1], replace_ratio)).unsqueeze(-1).to(z.device)  # replace_ratio chance of replacement
            z_q = torch.where(mask.bool(), z, z_q)
        

        # reshape back to match original input shape
        z_q = torch.einsum('b h w c -> b c h w', z_q)

        return z_q, (vq_loss, commit_loss, entropy_loss, ar_prior_loss, codebook_usage), (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.get_emb(), p=2, dim=-1)
        else:
            embedding = self.get_emb()

        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q



def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """
    modified from llamagen and magvit
    Args:
        affinity: (b, n, n), the affinity matrix, where affinity[i, j] is the affinity 
                between encoed vector i and codebook vector j
        loss_type: how to turn the affinity into probability distribution
    """
    # shape: (b n) n
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    # target_probs.shape: (b, n, n), and sum(target_probs, dim=-1) = 1
    avg_probs = torch.mean(target_probs, dim=0) # (,n)
    # average entropy corresponeds (negatively) to the diversity of indices for a single position
    avg_entropy = - torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    # sample entropy is the confidence for the quantization process
    # (bn, n) -> (bn) -> avg 
    sample_entropy = - torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    loss = sample_entropy - avg_entropy
    return loss

def compute_cosinesim_loss(feat1, feat2, dim):
    cos_sim = F.cosine_similarity(feat1, feat2, dim=dim)
    loss = 1 - cos_sim
    return torch.mean(loss)  
