# Modified from:
#   taming-transformers:  https://github.com/CompVis/taming-transformers
#   muse-maskgit-pytorch: https://github.com/lucidrains/muse-maskgit-pytorch/blob/main/muse_maskgit_pytorch/vqgan_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import os
import math
import numpy as np

from tokenizer.tokenizer_image.lpips import ResNet50ImgSim, LPIPS, DinoV2ImgSim
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminator as PatchGANDiscriminator
from tokenizer.tokenizer_image.discriminator_patchgan import NLayerDiscriminatorV2 as PatchGANDiscriminatorV2
from tokenizer.tokenizer_image.discriminator_stylegan import Discriminator as StyleGANDiscriminator

# deprecated
from tokenizer.tokenizer_image.discriminator_patchgan_SeD import PatchGANSeDiscriminatorV3
from tokenizer.tokenizer_image.discriminator_patchr3gan import PatchViTDiscriminator


from tokenizer.tokenizer_image.discriminator_dino import DinoDisc as DINODiscriminator
from tokenizer.tokenizer_image.diffaug import DiffAug

from utils.resume_log import wandb_cache_file_append

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

########################################
# R3GAN
# code modified from R3GAN: https://github.com/brownvc/R3GAN
########################################
def zero_centered_gradient_penalty(samples, critics):
    gradient, = torch.autograd.grad(
                    outputs=critics.sum(), 
                    inputs=samples, 
                    create_graph=True,
                    )
    return gradient.square().sum([1, 2, 3])


def r3gan_d_loss(
        logits_real, 
        logits_fake, 
        samples_real, 
        samples_fake,
        gamma=15,   # this may need to be tuned
        ema=None,
        iter=None,
        ):
    assert ema is None
    # assert iter is None
    # Relativistic discriminator loss
    relat_logits = logits_real - logits_fake
    adv_loss = nn.functional.softplus(-relat_logits)

    # R1 gradient penalty
    r1_penalty = zero_centered_gradient_penalty(samples=samples_real, critics=logits_real)
    # R2 gradient penalty
    r2_penalty = zero_centered_gradient_penalty(samples=samples_fake, critics=logits_fake)

    try:
        disc_loss = torch.mean(adv_loss) + torch.mean((gamma / 2) * (r1_penalty + r2_penalty))
    except:
        print("adv_loss shape", adv_loss.shape)
        print("r1_penalty shape", r1_penalty.shape)
        print("r2_penalty shape", r2_penalty.shape)
        exit()
    return disc_loss


def r3gan_gen_loss(logits_real, logits_fake):
    relat_logits = logits_fake - logits_real 
    adv_loss = nn.functional.softplus(-relat_logits)

    return torch.mean(adv_loss)


########################################
# hinge
########################################
def loss_hinge_dis(dis_fake, dis_real, ema=None, it=None):
  if ema is not None:
    # track the prediction
    ema.update(torch.mean(dis_fake).item(), 'D_fake', it)
    ema.update(torch.mean(dis_real).item(), 'D_real', it)

  loss_real = F.relu(1. - dis_real)
  loss_fake = F.relu(1. + dis_fake)
  return torch.mean(loss_real), torch.mean(loss_fake)


def hinge_d_loss(logits_real, logits_fake, ema=None, iter=None):
    if ema is not None:
        # track the prediction
        ema.update(torch.mean(logits_fake).item(), 'D_fake', iter)
        ema.update(torch.mean(logits_real).item(), 'D_real', iter)

    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def non_saturating_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logits_real),  logits_real))
    loss_fake = torch.mean(F.binary_cross_entropy_with_logits(torch.zeros_like(logits_fake), logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def hinge_gen_loss(logit_fake):
    return -torch.mean(logit_fake)


def non_saturating_gen_loss(logit_fake):
    return torch.mean(F.binary_cross_entropy_with_logits(torch.ones_like(logit_fake),  logit_fake))


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

def high_rank_attention_loss(attn_weights, eps=1e-6):
    sigma = torch.linalg.svdvals(attn_weights.float())
    p = sigma / (sigma.sum(dim=-1, keepdim=True) + eps)
    num_sigma = sigma.shape[-1]
    per_matrix_loss = (p - 1.0 / num_sigma).pow(2).mean(dim=-1)
    hr_loss = per_matrix_loss.mean()

    if num_sigma > 1:
        spectrum_uniformity = 1.0 - (per_matrix_loss.detach() * num_sigma * num_sigma / (num_sigma - 1)).mean()
    else:
        spectrum_uniformity = torch.ones((), device=attn_weights.device, dtype=hr_loss.dtype)

    return hr_loss, spectrum_uniformity


def _frobenius_normalize_attention_matrix(matrix, eps=1e-8):
    """Normalize one 2D attention matrix by its raw Frobenius norm."""
    matrix = matrix.float()
    fro_norm = torch.norm(matrix, p="fro")
    return matrix / (fro_norm + eps), fro_norm


def _sigma_mean_mse_loss(sigma):
    """Penalize variance across Frobenius-normalized singular values."""
    return (sigma - sigma.mean()).pow(2).mean()


def _gram_scaled_identity_loss(matrix_normed):
    """Scaled Gram loss after Frobenius normalization.

    Since ||A||_F=1, trace(A.T @ A)=1, so the identity target is scaled by
    the valid text-token count instead of using an unscaled identity matrix.
    """
    valid_token_count = matrix_normed.shape[1]
    gram = matrix_normed.transpose(0, 1).matmul(matrix_normed)
    target = torch.eye(valid_token_count, device=matrix_normed.device, dtype=matrix_normed.dtype)
    target = target / max(1, valid_token_count)
    return (gram - target).pow(2).mean()


def _log_participation_ratio_loss(sigma, eps=1e-8):
    """Negative log participation ratio, matching the advisor's formula."""
    rank_score = sigma.sum().pow(2) / (sigma.pow(2).sum() + eps)
    return -torch.log(rank_score + eps)


def _singular_energy_stats(sigma, eps=1e-8):
    energy = sigma.pow(2)
    p = energy / (energy.sum() + eps)
    p_for_log = p.clamp_min(eps)
    effective_rank = (-(p_for_log * p_for_log.log()).sum()).exp()
    participation_rank = sigma.sum().pow(2) / (energy.sum() + eps)
    participation_rank_ratio = participation_rank / max(1, sigma.numel())
    topk = min(5, p.numel())
    energy_top5 = torch.topk(p, topk).values.sum() if topk > 0 else sigma.new_zeros(())
    return {
        "energy": p,
        "effective_rank": effective_rank,
        "participation_rank": participation_rank,
        "participation_rank_ratio": participation_rank_ratio,
        "energy_top1": p.max(),
        "energy_top5": energy_top5,
    }


def high_rank_image_text_attention_loss(
        attn_weights,
        text_attention_mask,
        image_token_len=256,
        tau=1.0,
        skip_if_valid_tokens_lt=2,
        svd_mode="frobenius_uniform",
        eps=1e-8):
    """Text-HR v2: compute HR loss on the image-query to valid-text-key slice.

    attn_weights is the selected decoder layer post-softmax cross-attention
    [B, H, Q, K]. The first image_token_len keys are image tokens, and the
    following text_attention_mask.shape[1] keys are padded T5 text tokens.
    """
    if attn_weights.dim() != 4:
        raise ValueError(f"Expected attention shape [B, H, Q, K], got {attn_weights.shape}")
    if text_attention_mask is None:
        raise ValueError("text_attention_mask is required for image-to-text HR loss.")

    batch_size, num_heads, num_queries, num_keys = attn_weights.shape
    text_attention_mask = text_attention_mask.to(device=attn_weights.device, dtype=torch.bool)
    if text_attention_mask.dim() != 2 or text_attention_mask.shape[0] != batch_size:
        raise ValueError(
            f"Expected text_attention_mask shape [B, T], got {text_attention_mask.shape} "
            f"for attention batch={batch_size}"
        )

    text_token_len = text_attention_mask.shape[1]
    if num_keys < image_token_len + text_token_len:
        raise ValueError(
            f"Attention key length {num_keys} is smaller than image_token_len + text_token_len "
            f"({image_token_len} + {text_token_len})."
        )

    # Text-HR v2: keep only the text-key slice; padding tokens are removed
    # per sample before SVD.
    text_attn = attn_weights[:, :, :, image_token_len:image_token_len + text_token_len]
    svd_mode = str(svd_mode)
    frobenius_uniform_modes = {"frobenius_uniform", "frobenius_energy_uniform", "energy_uniform_mse"}
    advisor_modes = {"sigma_mean_mse", "gram_scaled_identity", "log_participation_ratio"}
    frobenius_modes = frobenius_uniform_modes | advisor_modes
    legacy_modes = {"legacy_tau", "raw_tau", "per_sample_all_heads_sqrt_norm"}
    if svd_mode not in frobenius_modes and svd_mode not in legacy_modes:
        raise ValueError(
            f"Unknown text_hr svd_mode={svd_mode}. "
            f"Expected one of {sorted(frobenius_modes | legacy_modes)}."
        )

    losses = []
    current_sigma_means = []
    current_sigma_mins = []
    current_sigma_maxs = []
    raw_sigma_means = []
    raw_sigma_mins = []
    raw_sigma_maxs = []
    normed_sigma_means = []
    normed_sigma_mins = []
    normed_sigma_maxs = []
    fro_norms = []
    participation_ranks = []
    participation_rank_ratios = []
    effective_ranks = []
    energy_top1s = []
    energy_top5s = []
    gram_losses = []
    text_attention_masses = []
    visual_attention_masses = []
    valid_token_counts = []
    skipped_samples = 0

    for batch_idx in range(batch_size):
        valid_mask = text_attention_mask[batch_idx]
        valid_token_count = int(valid_mask.sum().item())
        if valid_token_count < skip_if_valid_tokens_lt:
            skipped_samples += 1
            continue

        matrix = text_attn[batch_idx, :, :, valid_mask]
        raw_matrix = matrix.reshape(num_heads * num_queries, valid_token_count).float()
        # Attention mass checks whether the selected decoder layer actually
        # attends to text keys after visual/text memory concat.
        text_attention_masses.append(matrix.sum(dim=-1).float().mean().detach())
        visual_attention_masses.append(
            attn_weights[batch_idx, :, :, :image_token_len].sum(dim=-1).float().mean().detach()
        )
        # Text-HR v2: SVD is computed in float32 for stability even under bf16 training.
        raw_sigma = torch.linalg.svdvals(raw_matrix)
        matrix_normed, fro_norm = _frobenius_normalize_attention_matrix(raw_matrix, eps=eps)
        normed_sigma = torch.linalg.svdvals(matrix_normed)
        gram_loss = _gram_scaled_identity_loss(matrix_normed)

        if svd_mode in frobenius_modes:
            sigma = normed_sigma
            if svd_mode in frobenius_uniform_modes:
                sigma_energy = sigma.pow(2)
                p = sigma_energy / (sigma_energy.sum() + eps)
                target = sigma.new_tensor(1.0 / max(1, sigma.numel()))
                losses.append((p - target).pow(2).mean())
            elif svd_mode == "sigma_mean_mse":
                losses.append(_sigma_mean_mse_loss(sigma))
            elif svd_mode == "gram_scaled_identity":
                losses.append(gram_loss)
            elif svd_mode == "log_participation_ratio":
                losses.append(_log_participation_ratio_loss(sigma, eps=eps))
        else:
            legacy_matrix = raw_matrix / math.sqrt(num_heads)
            sigma = torch.linalg.svdvals(legacy_matrix)
            losses.append(torch.abs(sigma - sigma.new_tensor(tau)).mean())

        energy_stats = _singular_energy_stats(sigma, eps=eps)

        current_sigma_means.append(sigma.detach().mean())
        current_sigma_mins.append(sigma.detach().min())
        current_sigma_maxs.append(sigma.detach().max())
        raw_sigma_means.append(raw_sigma.detach().mean())
        raw_sigma_mins.append(raw_sigma.detach().min())
        raw_sigma_maxs.append(raw_sigma.detach().max())
        normed_sigma_means.append(normed_sigma.detach().mean())
        normed_sigma_mins.append(normed_sigma.detach().min())
        normed_sigma_maxs.append(normed_sigma.detach().max())
        fro_norms.append(fro_norm.detach())
        participation_ranks.append(energy_stats["participation_rank"].detach())
        participation_rank_ratios.append(energy_stats["participation_rank_ratio"].detach())
        effective_ranks.append(energy_stats["effective_rank"].detach())
        energy_top1s.append(energy_stats["energy_top1"].detach())
        energy_top5s.append(energy_stats["energy_top5"].detach())
        gram_losses.append(gram_loss.detach())
        valid_token_counts.append(valid_token_count)

    if not losses:
        zero = attn_weights.sum() * 0.0
        stats = {
            "sigma_mean": zero.detach(),
            "sigma_min": zero.detach(),
            "sigma_max": zero.detach(),
            "raw_sigma_mean": zero.detach(),
            "raw_sigma_min": zero.detach(),
            "raw_sigma_max": zero.detach(),
            "normed_sigma_mean": zero.detach(),
            "normed_sigma_min": zero.detach(),
            "normed_sigma_max": zero.detach(),
            "fro_norm_mean": zero.detach(),
            "participation_rank_mean": zero.detach(),
            "participation_rank_ratio_mean": zero.detach(),
            "effective_rank_mean": zero.detach(),
            "energy_top1_mean": zero.detach(),
            "energy_top5_mean": zero.detach(),
            "gram_loss_mean": zero.detach(),
            "text_attention_mass_mean": zero.detach(),
            "visual_attention_mass_mean": zero.detach(),
            "valid_text_tokens_mean": zero.detach(),
            "valid_samples": 0,
            "skipped_samples": skipped_samples,
        }
        return zero, stats

    hr_loss = torch.stack(losses).mean()
    valid_tokens = torch.tensor(valid_token_counts, device=attn_weights.device, dtype=torch.float32)
    stats = {
        # sigma_* follows the current mode's actual loss sigma.
        "sigma_mean": torch.stack(current_sigma_means).mean(),
        "sigma_min": torch.stack(current_sigma_mins).mean(),
        "sigma_max": torch.stack(current_sigma_maxs).mean(),
        # raw sigma stats expose whether image-to-text attention mass is collapsing.
        "raw_sigma_mean": torch.stack(raw_sigma_means).mean(),
        "raw_sigma_min": torch.stack(raw_sigma_mins).mean(),
        "raw_sigma_max": torch.stack(raw_sigma_maxs).mean(),
        # normed sigma stats expose the Frobenius-normalized spectrum.
        "normed_sigma_mean": torch.stack(normed_sigma_means).mean(),
        "normed_sigma_min": torch.stack(normed_sigma_mins).mean(),
        "normed_sigma_max": torch.stack(normed_sigma_maxs).mean(),
        "fro_norm_mean": torch.stack(fro_norms).mean(),
        "participation_rank_mean": torch.stack(participation_ranks).mean(),
        "participation_rank_ratio_mean": torch.stack(participation_rank_ratios).mean(),
        "effective_rank_mean": torch.stack(effective_ranks).mean(),
        "energy_top1_mean": torch.stack(energy_top1s).mean(),
        "energy_top5_mean": torch.stack(energy_top5s).mean(),
        "gram_loss_mean": torch.stack(gram_losses).mean(),
        "text_attention_mass_mean": torch.stack(text_attention_masses).mean(),
        "visual_attention_mass_mean": torch.stack(visual_attention_masses).mean(),
        "valid_text_tokens_mean": valid_tokens.mean(),
        "valid_samples": len(losses),
        "skipped_samples": skipped_samples,
    }
    return hr_loss, stats

# LeCam Regularziation loss
# from https://github.com/google/lecam-gan/
def lecam_reg(dis_real, dis_fake, ema):
  reg = torch.mean(F.relu(dis_real - ema.D_fake).pow(2)) + \
        torch.mean(F.relu(ema.D_real - dis_fake).pow(2))
  return reg


# Simple wrapper that applies EMA to losses.
# from https://github.com/google/lecam-gan/blob/f9af9485eda4637b25e694c142ce8e6992eb7243/third_party/utils.py#L636C1-L661C60
class ema_losses(object):
    def __init__(self, init=1000., decay=0.99, start_itr=0):
        self.G_loss = init
        self.D_loss_real = init
        self.D_loss_fake = init
        self.D_real = init
        self.D_fake = init
        self.decay = decay
        self.start_itr = start_itr

    def update(self, cur, mode, itr):
        if itr < self.start_itr:
            decay = 0.0
        else:
            decay = self.decay
        if mode == 'G_loss':
          self.G_loss = self.G_loss*decay + cur*(1 - decay)
        elif mode == 'D_loss_real':
          self.D_loss_real = self.D_loss_real*decay + cur*(1 - decay)
        elif mode == 'D_loss_fake':
          self.D_loss_fake = self.D_loss_fake*decay + cur*(1 - decay)
        elif mode == 'D_real':
          self.D_real = self.D_real*decay + cur*(1 - decay)
        elif mode == 'D_fake':
          self.D_fake = self.D_fake*decay + cur*(1 - decay)

class VQLoss(nn.Module):
    def __init__(self, 
                 disc_start, disc_loss="hinge", disc_dim=64, disc_type='patchgan', image_size=256,
                 disc_num_layers=3, disc_in_channels=3, disc_weight=1.0, disc_adaptive_weight = False,
                 gen_adv_loss='hinge', reconstruction_loss='l2', reconstruction_weight=1.0, 
                 codebook_weight=1.0, perceptual_weight=1.0, aux_loss_end=0, entropy_loss_end=None,
                 use_direct_rec_loss=True, norm="batch",kw=4, blur_ds=False, lecam=False, lecam_weight=0.001,
                 resnet_perceptual=False,     # (to be deleted) conflict with perceptual_model, fix it
                 proj_weight=0.5,             #  loss for the semantic regularization loss
                 gen_start=0,                 # (to be deleted) allow to train generator later 
                 dhead=32,                    # (deprecated)for Semantic discriminator cross-attn, 
                 disc_semantic_type="local",  # (deprecated)choosing from "local" or "global", for Semantic discriminator
                 use_semantic_input=False,    # (deprecated)choosing from "local" or "global", for Semantic discriminator
                 perceptual_model="vgg",      # for perceptual loss setting
                 gamma=15,  # for r3gan R1+R2 panelty, deprecated
                 discriminator_device="cuda",
    ):
        super().__init__()
        # discriminator loss

        assert disc_type in ["patchgan", "stylegan", "patchgan_SeD", "patchganv2", "dinodisc", "patchvit"]
        assert disc_loss in ["hinge", "vanilla", "non-saturating", "r3gan"]

        self.disc_type = disc_type

        self.disc_semantic_type = disc_semantic_type
        self.use_semantic_input = use_semantic_input

        self.use_direct_rec_loss = use_direct_rec_loss

        # specially for r3gan
        self.gamma = gamma

        if disc_type == "patchgan":
            self.discriminator = PatchGANDiscriminator(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
                norm=norm,
                kw=kw,
                blur_ds=blur_ds
            )
        elif disc_type == "patchganv2":
            self.discriminator = PatchGANDiscriminatorV2(
                input_nc=disc_in_channels, 
                n_layers=disc_num_layers,
                ndf=disc_dim,
                norm=norm,
                kw=kw,
                blur_ds=blur_ds,
                use_semantic_input=use_semantic_input
            )
        elif disc_type == "stylegan":
            self.discriminator = StyleGANDiscriminator(
                input_nc=disc_in_channels, 
                image_size=image_size,
            )
        elif disc_type == "patchgan_SeD":
            self.discriminator = PatchGANSeDiscriminatorV3(
                input_nc=disc_in_channels, 
                ndf=disc_dim,
                kw=kw,
                blur_ds=blur_ds,
                dhead=dhead,
            )
        elif disc_type == "dinodisc":
            aug_prob = 1.0
            self.discriminator = DINODiscriminator(norm_type="bn", device=discriminator_device)  # default 224 otherwise crop
            self.daug = DiffAug(prob=aug_prob, cutout=0.2)
        elif disc_type == "patchvit":
            self.discriminator = PatchViTDiscriminator(
                model_size="base",
            )
        else:
            raise ValueError(f"Unknown GAN discriminator type '{disc_type}'.")

        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        elif disc_loss == "non-saturating":
            self.disc_loss = non_saturating_d_loss
        elif disc_loss == "r3gan":
            self.disc_loss = r3gan_d_loss
        else:
            raise ValueError(f"Unknown GAN discriminator loss '{disc_loss}'.")
        
        self.disc_loss_type = disc_loss
        self.disc_type = disc_type
        self.gen_adv_loss_type = gen_adv_loss

        self.discriminator_iter_start = disc_start
        self.disc_weight = disc_weight
        self.disc_adaptive_weight = disc_adaptive_weight

        self.proj_weight = proj_weight

        assert gen_adv_loss in ["hinge", "non-saturating", "r3gan"]
        # gen_adv_loss
        if gen_adv_loss == "hinge":
            self.gen_adv_loss = hinge_gen_loss
        elif gen_adv_loss == "non-saturating":
            self.gen_adv_loss = non_saturating_gen_loss
        elif gen_adv_loss == "r3gan":
            self.gen_adv_loss = r3gan_gen_loss
        else:
            raise ValueError(f"Unknown GAN generator loss '{gen_adv_loss}'.")

        # perceptual loss
        if perceptual_model == "resent50":
            self.perceptual_loss = ResNet50ImgSim().eval()
        elif perceptual_model == "dinov2-s":
            self.perceptual_loss = DinoV2ImgSim().eval()
        elif perceptual_model == "vgg":
            self.perceptual_loss = LPIPS().eval()
        else:
            raise ValueError(f"Unknown perceptual model '{perceptual_model}'.")
        self.perceptual_weight = perceptual_weight

        # reconstruction loss
        if reconstruction_loss == "l1":
            self.rec_loss = F.l1_loss
        elif reconstruction_loss == "l2":
            self.rec_loss = F.mse_loss
        else:
            raise ValueError(f"Unknown rec loss '{reconstruction_loss}'.")
        self.rec_weight = reconstruction_weight

        # codebook loss
        self.codebook_weight = codebook_weight

        # iteration to stop using auxiliary loss
        self.aux_loss_end = aux_loss_end
        self.entropy_loss_end = entropy_loss_end


        # Special config for logging
        self.log_update_cache_generator = []
        self.log_update_cache_discriminator = []

        self.log_update_cache_multi_level = {}

        self.lecam = lecam
        if self.lecam:
            self.ema_logits = ema_losses(start_itr=self.discriminator_iter_start + 1000)
        else:
            self.ema_logits = None
        self.lecam_weight = lecam_weight


    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        return d_weight.detach()

    def forward(self, inter_loss_set, inputs, all_reconstructions, optimizer_idx, global_step, exp_dir, last_layer=None, 
                logger=None, log_every=100, ckpt_every=500, num_en_q_level=None, causal_type=None,
                check_nan_loss=True, inner_feat=None, sem_enc_feat=None,
                hr_attn_weights=None, hr_loss_weight=0.0, selected_layer=None, hr_eps=1e-6,
                text_hr_attn_weights=None, text_attention_mask=None, text_hr_loss_weight=0.0,
                selected_text_layer=None, text_hr_tau=1.0,
                text_hr_skip_if_valid_tokens_lt=2, text_hr_image_token_len=256,
                text_hr_svd_mode="frobenius_uniform", text_hr_eps=1e-8,
                text_recon_stats=None
                ):
        assert len(inter_loss_set) == 2
        assert isinstance(all_reconstructions, list)

        # if sem_enc_feat is not None:
        #     # B L D
        #     print(sem_enc_feat.shape)

        if self.use_direct_rec_loss:
            reconstructions, direct_reconstructions = all_reconstructions
        else:
            reconstructions = all_reconstructions[0]

        codebook_loss, feature_rec_loss = inter_loss_set

        rank = dist.get_rank() 
        node_rank = int(os.environ.get('NODE_RANK', 0))
        # generator update
        if optimizer_idx == 0:
            # reconstruction loss
            rec_loss = self.rec_loss(inputs.contiguous(), reconstructions.contiguous())

            direct_rec_loss = self.rec_loss(inputs.contiguous(), direct_reconstructions.contiguous()) \
                                if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0

            # perceptual loss
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            direct_p_loss = self.perceptual_loss(inputs.contiguous(), direct_reconstructions.contiguous()) \
                                if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0
            p_loss = torch.mean(p_loss)
            direct_p_loss = torch.mean(direct_p_loss) if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0

            # if num_en_q_level is not None and causal_type == "per-level" and :
            #     assert causal_type is not None, "causal_type must be specified when cur_level is not None"
            #     if f"p_loss_{num_en_q_level}_{causal_type}" not in self.log_update_cache_multi_level:
            #         self.log_update_cache_multi_level[f"p_loss_{num_en_q_level}_{causal_type}"] = \
            #             [p_loss]
            #     else:
            #         self.log_update_cache_multi_level[f"p_loss_{num_en_q_level}_{causal_type}"].append(p_loss)


            # discriminator loss
            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)
            if global_step < self.discriminator_iter_start:
                logits_fake = 0
                generator_adv_loss = 0
                direct_logits_fake = 0
                direct_generator_adv_loss = 0
            else:
                if "SeD" in self.disc_type or self.use_semantic_input:

                    assert sem_enc_feat is not None, "Semantic discriminator needs sem_enc_feat as input"
                    if self.disc_semantic_type == "global":
                        global_sem_enc_feat = sem_enc_feat.detach()
                        if sem_enc_feat.dim() == 3:
                            # B (H W) C -> B 1 C -> B (H W) C
                            global_sem_enc_feat = global_sem_enc_feat.mean(dim=1, keepdim=True)
                            global_sem_enc_feat = global_sem_enc_feat.expand(-1, sem_enc_feat.shape[1], -1)
                        else:
                            # B C H W -> B C 1 1 -> B C H W
                            global_sem_enc_feat = global_sem_enc_feat.mean(dim=(2, 3), keepdim=True)
                            global_sem_enc_feat = global_sem_enc_feat.expand(-1, -1, sem_enc_feat.shape[2], sem_enc_feat.shape[3])

                        disc_sem_feat = global_sem_enc_feat
                    elif self.disc_semantic_type == "local":
                        disc_sem_feat = sem_enc_feat.detach()
                    else:
                        raise ValueError("disc_semantic_type must be global or local")

                    logits_fake = self.discriminator(reconstructions.contiguous().detach(), disc_sem_feat)
                    direct_logits_fake = self.discriminator(direct_reconstructions.contiguous().detach(), disc_sem_feat) \
                                            if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0
                elif self.disc_type == "dinodisc":
                    fade_blur_schedule = 0
                    logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous(), fade_blur_schedule))
                    direct_logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous(), fade_blur_schedule)) \
                                            if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0
                else:
                    logits_fake = self.discriminator(reconstructions.contiguous())
                    direct_logits_fake = self.discriminator(direct_reconstructions.contiguous()) \
                                            if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0

                if self.gen_adv_loss_type == "r3gan":
                    logits_real = self.discriminator(inputs.contiguous())
                    generator_adv_loss = self.gen_adv_loss(logits_real, logits_fake)
                    direct_generator_adv_loss = self.gen_adv_loss(logits_real, direct_logits_fake) \
                                                    if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0
                
                else:
                    generator_adv_loss = self.gen_adv_loss(logits_fake)
                    direct_generator_adv_loss = self.gen_adv_loss(direct_logits_fake) \
                                                if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0
                
            if self.disc_adaptive_weight:
                null_loss = self.rec_weight * (rec_loss + direct_rec_loss) + self.perceptual_weight * p_loss
                disc_adaptive_weight = self.calculate_adaptive_weight(null_loss, generator_adv_loss, last_layer=last_layer)
            else:
                disc_adaptive_weight = 1
            
            if global_step >= self.aux_loss_end:
                direct_rec_loss = direct_rec_loss * 0.0
                feature_rec_loss = feature_rec_loss * 0.0
                direct_p_loss = direct_p_loss * 0.0
                direct_generator_adv_loss = direct_generator_adv_loss * 0.0
            
            if self.entropy_loss_end is not None and global_step >= self.entropy_loss_end:
                codebook_loss[2] = 0.0
            
            # compute semantic distill loss
            # projection loss
            proj_loss = 0.
            if inner_feat is not None and sem_enc_feat is not None:
                assert inner_feat.shape == sem_enc_feat.shape, f"inner_feat.shape: {inner_feat.shape}, sem_enc_feat.shape: {sem_enc_feat.shape}"
                bsz = inner_feat.shape[0]
                # TODO: the for loop is ugly, change to array calculation
                for j, (z_j, z_tilde_j) in enumerate(zip(inner_feat, sem_enc_feat)):
                    z_tilde_j = torch.nn.functional.normalize(z_tilde_j, dim=-1) 
                    z_j = torch.nn.functional.normalize(z_j, dim=-1) 
                    proj_loss += mean_flat(-(z_j * z_tilde_j).sum(dim=-1))
                proj_loss /= bsz
            
            codebook_loss_sum = self.codebook_weight * sum(codebook_loss[:-1]) # the last one is codebook usage

            hr_loss = None
            hr_spectrum_uniformity = None
            if hr_attn_weights is not None:
                hr_loss, hr_spectrum_uniformity = high_rank_attention_loss(hr_attn_weights, eps=hr_eps)
            elif hr_loss_weight != 0:
                raise ValueError("hr_loss_weight is non-zero but hr_attn_weights is None.")

            hr_loss_term = hr_loss_weight * hr_loss if hr_loss is not None else 0.0
            text_hr_loss = None
            text_hr_stats = {}
            if text_hr_attn_weights is not None:
                # Text-HR v2: this is the only new loss term for the text-guided
                # experiment; it is logged separately from reconstruction losses.
                text_hr_loss, text_hr_stats = high_rank_image_text_attention_loss(
                    text_hr_attn_weights,
                    text_attention_mask=text_attention_mask,
                    image_token_len=text_hr_image_token_len,
                    tau=text_hr_tau,
                    skip_if_valid_tokens_lt=text_hr_skip_if_valid_tokens_lt,
                    svd_mode=text_hr_svd_mode,
                    eps=text_hr_eps,
                )
            elif text_hr_loss_weight != 0:
                raise ValueError("text_hr_loss_weight is non-zero but text_hr_attn_weights is None.")

            text_hr_loss_term = text_hr_loss_weight * text_hr_loss if text_hr_loss is not None else 0.0
            text_recon_stats = text_recon_stats or {}
            def text_recon_stat_float(name, default=0.0):
                value = text_recon_stats.get(name, default)
                if torch.is_tensor(value):
                    return float(value.detach().float().mean().item())
                return float(value)
            loss = self.rec_weight * (rec_loss + direct_rec_loss) + \
                self.perceptual_weight * (p_loss + direct_p_loss) + \
                disc_adaptive_weight * disc_weight * (generator_adv_loss + direct_generator_adv_loss) + \
                codebook_loss_sum + feature_rec_loss + self.proj_weight * proj_loss + \
                hr_loss_term + text_hr_loss_term

            if check_nan_loss:
                if torch.isnan(loss).any():
                    error_info = ""
                    rec_loss = self.rec_weight * rec_loss
                    direct_rec_loss = self.rec_weight * direct_rec_loss
                    p_loss = self.perceptual_weight * p_loss
                    direct_p_loss = self.perceptual_weight * direct_p_loss
                    generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                    direct_generator_adv_loss = disc_adaptive_weight * disc_weight * direct_generator_adv_loss
                    # check any nan in reconstruction input
                    if torch.isnan(inputs).any():
                        error_info += "input contains nan\n"
                    if torch.isnan(reconstructions).any():
                        error_info += "reconstructions contains nan\n"
                    error_info += (f"(Generator) rec_loss: {rec_loss:.4f}, direct_rec_loss: {direct_rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, direct_p_loss: {direct_p_loss:.4f}, "
                                f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
                                f"feature_rec_loss: {feature_rec_loss:.4f}, codebook_usage: {codebook_loss[-1]:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, direct_generator_adv_loss: {direct_generator_adv_loss:.4f}, "
                                f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}\n")
                    if hr_loss is not None:
                        error_info += (
                            f"hr_loss: {hr_loss:.4e}, hr_loss_weight: {hr_loss_weight:.4e}, "
                            f"selected_layer: {selected_layer}, hr_spectrum_uniformity: {hr_spectrum_uniformity:.4f}\n"
                        )
                    if text_hr_loss is not None:
                        error_info += (
                            f"text_hr_loss: {text_hr_loss:.4e}, text_hr_loss_weight: {text_hr_loss_weight:.4e}, "
                            f"selected_decoder_layer: {selected_layer}, selected_text_layer: {selected_text_layer}, "
                            f"text_hr_svd_mode: {text_hr_svd_mode}, "
                            f"text_hr_sigma_mean: {text_hr_stats['sigma_mean']:.4e}, "
                            f"text_hr_raw_sigma_mean: {text_hr_stats['raw_sigma_mean']:.4e}, "
                            f"text_hr_normed_sigma_mean: {text_hr_stats['normed_sigma_mean']:.4e}, "
                            f"text_hr_fro_norm_mean: {text_hr_stats['fro_norm_mean']:.4e}, "
                            f"text_hr_participation_rank_mean: {text_hr_stats['participation_rank_mean']:.2f}, "
                            f"text_hr_participation_rank_ratio_mean: {text_hr_stats['participation_rank_ratio_mean']:.4f}, "
                            f"text_hr_effective_rank_mean: {text_hr_stats['effective_rank_mean']:.2f}, "
                            f"text_hr_gram_loss_mean: {text_hr_stats['gram_loss_mean']:.4e}, "
                            f"text_hr_text_attention_mass_mean: {text_hr_stats['text_attention_mass_mean']:.4f}, "
                            f"text_hr_visual_attention_mass_mean: {text_hr_stats['visual_attention_mass_mean']:.4f}, "
                            f"text_hr_valid_text_tokens_mean: {text_hr_stats['valid_text_tokens_mean']:.2f}, "
                            f"text_hr_skipped_samples: {text_hr_stats['skipped_samples']}\n"
                        )
                    get_ip_cmd = """hostname -I | awk '{split($0, a, " "); print a[1]}'"""
                    ip_addr = os.popen(get_ip_cmd).read().strip()
                    error_info += f"ip: {ip_addr}\n"
                    error_info += f"current iteration:{global_step}"
                    raise RuntimeError(error_info)
                    logger.info(error_info)
                    # print(error_info)
            
            if rank == 0 and node_rank == 0 and (global_step % log_every == 0):
                rec_loss = self.rec_weight * rec_loss
                direct_rec_loss = self.rec_weight * direct_rec_loss
                p_loss = self.perceptual_weight * p_loss
                direct_p_loss = self.perceptual_weight * direct_p_loss
                generator_adv_loss = disc_adaptive_weight * disc_weight * generator_adv_loss
                direct_generator_adv_loss = disc_adaptive_weight * disc_weight * direct_generator_adv_loss
                log_msg = (f"(Generator) rec_loss: {rec_loss:.4f}, direct_rec_loss: {direct_rec_loss:.4f}, perceptual_loss: {p_loss:.4f}, direct_p_loss: {direct_p_loss:.4f}, "
                           f"vq_loss: {codebook_loss[0]:.4f}, commit_loss: {codebook_loss[1]:.4f}, entropy_loss: {codebook_loss[2]:.4f}, "
                           f"feature_rec_loss: {feature_rec_loss:.4f}, codebook_usage: {codebook_loss[-1]:.4f}, generator_adv_loss: {generator_adv_loss:.4f}, direct_generator_adv_loss: {direct_generator_adv_loss:.4f}, "
                           f"disc_adaptive_weight: {disc_adaptive_weight:.4f}, disc_weight: {disc_weight:.4f}, "
                           f"proj_loss: {proj_loss:.4f}, "
                           f"ar_prior_loss: {0 if len(codebook_loss) <= 4 else codebook_loss[3]:.2e}")
                if hr_loss is not None:
                    log_msg += (f", hr_loss: {hr_loss:.4e}, weighted_hr_loss: {hr_loss_term:.4e}, "
                                f"selected_layer: {selected_layer}, hr_spectrum_uniformity: {hr_spectrum_uniformity:.4f}")
                if text_hr_loss is not None:
                    log_msg += (
                        f", text_hr_loss: {text_hr_loss:.4e}, weighted_text_hr_loss: {text_hr_loss_term:.4e}, "
                        f"selected_decoder_layer: {selected_layer}, selected_text_layer: {selected_text_layer}, "
                        f"text_hr_svd_mode: {text_hr_svd_mode}, "
                        f"text_hr_sigma_mean: {text_hr_stats['sigma_mean']:.4e}, "
                        f"text_hr_raw_sigma_mean: {text_hr_stats['raw_sigma_mean']:.4e}, "
                        f"text_hr_normed_sigma_mean: {text_hr_stats['normed_sigma_mean']:.4e}, "
                        f"text_hr_fro_norm_mean: {text_hr_stats['fro_norm_mean']:.4e}, "
                        f"text_hr_participation_rank_mean: {text_hr_stats['participation_rank_mean']:.2f}, "
                        f"text_hr_participation_rank_ratio_mean: {text_hr_stats['participation_rank_ratio_mean']:.4f}, "
                        f"text_hr_effective_rank_mean: {text_hr_stats['effective_rank_mean']:.2f}, "
                        f"text_hr_energy_top1_mean: {text_hr_stats['energy_top1_mean']:.4f}, "
                        f"text_hr_gram_loss_mean: {text_hr_stats['gram_loss_mean']:.4e}, "
                        f"text_hr_text_attention_mass_mean: {text_hr_stats['text_attention_mass_mean']:.4f}, "
                        f"text_hr_visual_attention_mass_mean: {text_hr_stats['visual_attention_mass_mean']:.4f}, "
                        f"text_hr_valid_text_tokens_mean: {text_hr_stats['valid_text_tokens_mean']:.2f}, "
                        f"text_hr_skipped_samples: {text_hr_stats['skipped_samples']}"
                    )
                if text_recon_stats:
                    printed_text_recon_keys = {
                        "text_recon_enabled",
                        "text_injection_layer_count",
                        "text_gate",
                        "text_memory_norm_before_gate_mean",
                        "text_memory_norm_after_gate_mean",
                        "selected_text_memory_norm",
                        "visual_memory_norm_mean",
                        "text_visual_norm_ratio",
                        "visual_memory_mask_ratio",
                        "visual_memory_mask_actual_ratio",
                        "empty_text_count",
                        "text_valid_tokens_mean",
                    }
                    log_msg += (
                        f", text_recon_enabled: {text_recon_stat_float('text_recon_enabled'):.0f}, "
                        f"text_injection_layer_count: {text_recon_stat_float('text_injection_layer_count'):.0f}, "
                        f"text_gate: {text_recon_stat_float('text_gate'):.4f}, "
                        f"text_memory_norm_before_gate_mean: "
                        f"{text_recon_stat_float('text_memory_norm_before_gate_mean'):.4e}, "
                        f"text_memory_norm_after_gate_mean: "
                        f"{text_recon_stat_float('text_memory_norm_after_gate_mean'):.4e}, "
                        f"selected_text_memory_norm: {text_recon_stat_float('selected_text_memory_norm'):.4e}, "
                        f"visual_memory_norm_mean: {text_recon_stat_float('visual_memory_norm_mean'):.4e}, "
                        f"text_visual_norm_ratio: {text_recon_stat_float('text_visual_norm_ratio'):.4e}, "
                        f"visual_memory_mask_ratio: {text_recon_stat_float('visual_memory_mask_ratio'):.4f}, "
                        f"visual_memory_mask_actual_ratio: "
                        f"{text_recon_stat_float('visual_memory_mask_actual_ratio'):.4f}, "
                        f"empty_text_count: {text_recon_stat_float('empty_text_count'):.0f}, "
                        f"text_valid_tokens_mean: {text_recon_stat_float('text_valid_tokens_mean'):.2f}"
                    )
                    for stat_key in sorted(text_recon_stats):
                        if stat_key in printed_text_recon_keys:
                            continue
                        log_msg += f", {stat_key}: {text_recon_stat_float(stat_key):.4e}"
                logger.info(log_msg)

                # update to wandb
                update_info = {
                    "(Generator)rec_loss": rec_loss,
                    "(Generator)perceptual_loss": p_loss,
                    "(Generator)vq_loss": codebook_loss[0],
                    "(Generator)commit_loss": codebook_loss[1],
                    "(Generator)entropy_loss": codebook_loss[2],
                    "(Generator)codebook_usage": codebook_loss[-1],
                    "(Generator)generator_adv_loss": generator_adv_loss,
                    "(Generator)disc_adaptive_weight": disc_adaptive_weight,
                    "(Generator)disc_weight": disc_weight,
                    "(Generator)direct_rec_loss": direct_rec_loss,
                    "(Generator)direct_p_loss": direct_p_loss,
                    "iteration": global_step,
                    "(Generator)proj_loss": proj_loss,
                    "(Generator)ar_prior_loss": 0 if len(codebook_loss) <= 4 else codebook_loss[3]
                }
                if hr_loss is not None:
                    update_info.update({
                        "(Generator)hr_loss": hr_loss.detach(),
                        "(Generator)weighted_hr_loss": hr_loss_term.detach(),
                        "(Generator)hr_selected_layer": selected_layer,
                        "(Generator)hr_spectrum_uniformity": hr_spectrum_uniformity.detach(),
                    })
                if text_hr_loss is not None:
                    update_info.update({
                        "(Generator)text_hr_loss": text_hr_loss.detach(),
                        "(Generator)weighted_text_hr_loss": text_hr_loss_term.detach(),
                        "(Generator)text_hr_selected_decoder_layer": selected_layer,
                        "(Generator)text_hr_selected_text_layer": selected_text_layer,
                        "(Generator)text_hr_sigma_mean": text_hr_stats["sigma_mean"].detach(),
                        "(Generator)text_hr_sigma_min": text_hr_stats["sigma_min"].detach(),
                        "(Generator)text_hr_sigma_max": text_hr_stats["sigma_max"].detach(),
                        "(Generator)text_hr_raw_sigma_mean": text_hr_stats["raw_sigma_mean"].detach(),
                        "(Generator)text_hr_raw_sigma_min": text_hr_stats["raw_sigma_min"].detach(),
                        "(Generator)text_hr_raw_sigma_max": text_hr_stats["raw_sigma_max"].detach(),
                        "(Generator)text_hr_normed_sigma_mean": text_hr_stats["normed_sigma_mean"].detach(),
                        "(Generator)text_hr_normed_sigma_min": text_hr_stats["normed_sigma_min"].detach(),
                        "(Generator)text_hr_normed_sigma_max": text_hr_stats["normed_sigma_max"].detach(),
                        "(Generator)text_hr_fro_norm_mean": text_hr_stats["fro_norm_mean"].detach(),
                        "(Generator)text_hr_participation_rank_mean": text_hr_stats["participation_rank_mean"].detach(),
                        "(Generator)text_hr_participation_rank_ratio_mean": text_hr_stats["participation_rank_ratio_mean"].detach(),
                        "(Generator)text_hr_effective_rank_mean": text_hr_stats["effective_rank_mean"].detach(),
                        "(Generator)text_hr_energy_top1_mean": text_hr_stats["energy_top1_mean"].detach(),
                        "(Generator)text_hr_energy_top5_mean": text_hr_stats["energy_top5_mean"].detach(),
                        "(Generator)text_hr_gram_loss_mean": text_hr_stats["gram_loss_mean"].detach(),
                        "(Generator)text_hr_text_attention_mass_mean": text_hr_stats["text_attention_mass_mean"].detach(),
                        "(Generator)text_hr_visual_attention_mass_mean": text_hr_stats["visual_attention_mass_mean"].detach(),
                        "(Generator)text_hr_valid_text_tokens_mean": text_hr_stats["valid_text_tokens_mean"].detach(),
                        "(Generator)text_hr_valid_samples": text_hr_stats["valid_samples"],
                        "(Generator)text_hr_skipped_samples": text_hr_stats["skipped_samples"],
                    })
                if text_recon_stats:
                    for stat_key, stat_value in text_recon_stats.items():
                        if torch.is_tensor(stat_value):
                            update_info[f"(Generator){stat_key}"] = stat_value.detach()
                        else:
                            update_info[f"(Generator){stat_key}"] = stat_value

                # if proj_loss > 0:
                #     update_info["(Generator)proj_loss"] = proj_loss

                if num_en_q_level is not None and causal_type == "per-level":
                    #     print(key)
                    #     print(value)
                    multi_level_update_info = {
                        key: np.mean(value)
                        for key, value in self.log_update_cache_multi_level.items()
                    }
                    update_info.update(multi_level_update_info)

                    multi_level_info = "\n".join(
                        [f"{key}: {np.mean(value):.4f}, " for key, value in multi_level_update_info.items()]
                    )
                    logger.info(multi_level_info)

                self.log_update_cache_multi_level = {}
                self.log_update_cache_generator.append(update_info)


            if rank == 0 and node_rank == 0 and (global_step % ckpt_every == 0 and global_step > 0):
                # update to wandb
                wandb_cache_file_append(self.log_update_cache_generator, exp_dir)
                self.log_update_cache_generator = []

            return loss

        # discriminator update
        if optimizer_idx == 1:
            if global_step < self.discriminator_iter_start:
                return 0.0

            if "SeD" in self.disc_type or self.use_semantic_input:
                # Semantic discriminator needs sem_enc_feat as input
                assert sem_enc_feat is not None, "Semantic discriminator needs sem_enc_feat as input"

                # use only global feature 
                # B C H W -> B C 1 1 -> B C H W
                if self.disc_semantic_type == "global":
                    global_sem_enc_feat = sem_enc_feat.detach()
                    if sem_enc_feat.dim() == 3:
                        # B (H W) C -> B 1 C -> B (H W) C
                        global_sem_enc_feat = global_sem_enc_feat.mean(dim=1, keepdim=True)
                        global_sem_enc_feat = global_sem_enc_feat.expand(-1, sem_enc_feat.shape[1], -1)
                    else:
                        assert sem_enc_feat.dim() == 4, "sem_enc_feat.dim() must be 3 or 4"
                        # B C H W -> B C 1 1 -> B C H W
                        global_sem_enc_feat = global_sem_enc_feat.mean(dim=(2, 3), keepdim=True)
                        global_sem_enc_feat = global_sem_enc_feat.expand(-1, -1, sem_enc_feat.shape[2], sem_enc_feat.shape[3])
                    disc_sem_feat = global_sem_enc_feat
                elif self.disc_semantic_type == "local":
                    disc_sem_feat = sem_enc_feat.detach()
                else:
                    raise ValueError("disc_semantic_type must be global or local")


                # TODO: check if this detach() is leading to problems
                logits_real = self.discriminator(inputs.contiguous().detach(), disc_sem_feat)
                logits_fake = self.discriminator(reconstructions.contiguous().detach(), disc_sem_feat)
                direct_logits_fake = self.discriminator(direct_reconstructions.contiguous().detach(), disc_sem_feat) \
                                        if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0
            elif self.disc_type == "dinodisc":
                fade_blur_schedule = 0
                # add blur since disc is too strong
                logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous().detach(), fade_blur_schedule))
                logits_real = self.discriminator(self.daug.aug(inputs.contiguous().detach(), fade_blur_schedule))
                # depracated
                direct_logits_fake = self.discriminator(self.daug.aug(reconstructions.contiguous().detach(), fade_blur_schedule)) \
                                        if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0
            elif self.disc_loss_type == "r3gan":
                samples_real = inputs.contiguous().detach().clone().requires_grad_(True)
                samples_fake = reconstructions.contiguous().detach().clone().requires_grad_(True)
                logits_real = self.discriminator(samples_real)
                logits_fake = self.discriminator(samples_fake)
                if self.use_direct_rec_loss and global_step < self.aux_loss_end:
                    dir_samples_fake = direct_reconstructions.contiguous().detach().clone().requires_grad_(True)
                    direct_logits_fake = self.discriminator(dir_samples_fake)
                else:
                    direct_logits_fake = 0.0

            else:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                direct_logits_fake = self.discriminator(direct_reconstructions.contiguous().detach()) \
                                        if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0

            disc_weight = adopt_weight(self.disc_weight, global_step, threshold=self.discriminator_iter_start)

            if self.disc_loss_type == "r3gan":
                d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake, 
                                                                  samples_fake=samples_fake,
                                                                  samples_real=samples_real,
                                                                  gamma=self.gamma,
                                                                  ema=self.ema_logits, iter=global_step,
                                                                  )
                if self.use_direct_rec_loss and global_step < self.aux_loss_end:
                    d_d_adversarial_loss = disc_weight * self.disc_loss(
                                                                logits_real, direct_logits_fake, 
                                                                samples_real=samples_real, samples_fake=dir_samples_fake,
                                                                gamma=self.gamma,
                                                                ema=self.ema_logits, iter=global_step)
                else:
                    d_d_adversarial_loss = 0.0
                lecam_regularization = 0.0
            else:
                d_adversarial_loss = disc_weight * self.disc_loss(logits_real, logits_fake, ema=self.ema_logits, iter=global_step)
                d_d_adversarial_loss = disc_weight * self.disc_loss(logits_real, direct_logits_fake, ema=self.ema_logits, iter=global_step) \
                                        if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0
                lecam_regularization = self.lecam_weight * lecam_reg(logits_real, logits_fake, ema=self.ema_logits) \
                                        if self.lecam and global_step > self.discriminator_iter_start else 0.0

            if global_step >= self.aux_loss_end:
                d_d_adversarial_loss = d_d_adversarial_loss * 0.0
            
            if global_step % log_every == 0:
                logits_real = logits_real.detach().mean()
                logits_fake = logits_fake.detach().mean()
                direct_logits_fake = direct_logits_fake.detach().mean() \
                                    if self.use_direct_rec_loss and global_step < self.aux_loss_end else 0.0
                logger.info(f"(Discriminator) " 
                            f"discriminator_adv_loss: {d_adversarial_loss:.4f}, d_d_adversarial_loss: {d_d_adversarial_loss:.4f}, disc_weight: {disc_weight:.4f}, "
                            f"logits_real: {logits_real:.4f}, logits_fake: {logits_fake:.4f}, direct_logits_fake: {direct_logits_fake:.4f}")

                update_info = {
                    "(Discriminator)discriminator_adv_loss": d_adversarial_loss,
                    "(Discriminator)disc_weight": disc_weight,
                    "(Discriminator)logits_real": logits_real,
                    "(Discriminator)logits_fake": logits_fake,
                    "iteration": global_step,
                }
                self.log_update_cache_discriminator.append(update_info)

            rank = dist.get_rank()             
            if rank == 0 and (global_step % ckpt_every == 0 and global_step > 0):
                # update to wandb
                wandb_cache_file_append(self.log_update_cache_discriminator, exp_dir)
                self.log_update_cache_discriminator = []

            return d_adversarial_loss + d_d_adversarial_loss + lecam_regularization
