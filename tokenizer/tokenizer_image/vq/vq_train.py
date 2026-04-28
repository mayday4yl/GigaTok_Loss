# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image

import os
import time
import argparse
import csv
import json
from contextlib import nullcontext
from math import ceil
from glob import glob
from copy import deepcopy
from itertools import islice

import random

import numpy as np

from utils.logger import create_logger
import logging
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.model_init import load_model_from_config, custom_load

from utils.resume_log import (
    init_wandb, upload_wandb_cache, wandb_cache_file_append,
    manage_ckpt_num, get_int_prefix_value, int_prefix_paths, wsd_find_newest_ckpt
)
from utils.model_init import load_encoders
import yaml

try:
    import wandb
except Exception:  # noqa: BLE001 - --no-wandb should work even if wandb env is broken.
    wandb = None

try:
    from transformers import AutoTokenizer, T5EncoderModel
except ImportError:
    AutoTokenizer = None
    T5EncoderModel = None

try:
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:
    ssim_metric = None


from dataset.augmentation import random_crop_arr
from dataset.build import build_dataset
from tokenizer.tokenizer_image.vq.vq_loss import VQLoss
from tokenizer.tokenizer_image.scheduler import cosine_lr, const_lr, cosine_schedule_with_warmup_v2, wsd_lr

from torchvision.transforms import Normalize
import warnings
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
warnings.filterwarnings('ignore')


wandb_project_name = "vqvit_imagenet_train"


class JsonImageDataset(Dataset):
    def __init__(self, json_path, transform=None, max_images=0):
        with open(json_path, "r") as handle:
            image_paths = json.load(handle)
        if max_images and max_images > 0:
            image_paths = image_paths[:max_images]
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(0)


def resize_pad_arr(pil_image, image_size, fill=255):
    pil_image = pil_image.convert("RGB")
    width, height = pil_image.size
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    pil_image = pil_image.resize((resized_width, resized_height), Image.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), color=(fill, fill, fill))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas.paste(pil_image, (left, top))
    return canvas


def prepare_device_backend(device_backend):
    if device_backend == "cuda":
        assert torch.cuda.is_available(), "Training currently requires at least one CUDA GPU."
        return torch.cuda
    if device_backend == "npu":
        import torch_npu  # noqa: F401
        assert torch.npu.is_available(), "Training currently requires at least one Ascend NPU."
        return torch.npu
    raise ValueError(f"Unsupported device backend: {device_backend}")


def autocast_context(device_backend, mixed_precision, dtype=None):
    if mixed_precision == "none":
        return nullcontext()
    if device_backend == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    if device_backend == "npu":
        npu_amp = getattr(getattr(torch, "npu", None), "amp", None)
        if npu_amp is not None and hasattr(npu_amp, "autocast"):
            return npu_amp.autocast(dtype=dtype)
        return torch.autocast(device_type="npu", dtype=dtype)
    raise ValueError(f"Unsupported device backend: {device_backend}")


def synchronize_device(device_backend):
    if device_backend == "cuda":
        torch.cuda.synchronize()
    elif device_backend == "npu":
        torch.npu.synchronize()
    else:
        raise ValueError(f"Unsupported device backend: {device_backend}")


def build_text_layer_pairs(text_hr_cfg, decoder_num_layers, text_num_layers=None, config_name="text_hr"):
    raw_pairs = text_hr_cfg.get("layer_pairs", None)
    if not raw_pairs:
        shared_layers = min(decoder_num_layers, text_num_layers) if text_num_layers is not None else decoder_num_layers
        start = shared_layers // 3
        end = min(shared_layers, start + max(1, shared_layers // 3))
        raw_pairs = [[layer_idx, layer_idx] for layer_idx in range(start, end)]

    layer_pairs = []
    for pair in raw_pairs:
        if len(pair) != 2:
            raise ValueError(f"{config_name}.layer_pairs entries must be [t5_layer, decoder_layer], got {pair}")
        text_layer, decoder_layer = int(pair[0]), int(pair[1])
        if decoder_layer < 0 or decoder_layer >= decoder_num_layers:
            raise ValueError(
                f"decoder_layer={decoder_layer} is out of range for decoder_num_layers={decoder_num_layers}"
            )
        if text_num_layers is not None and (text_layer < 0 or text_layer >= text_num_layers):
            raise ValueError(f"text_layer={text_layer} is out of range for T5 num_layers={text_num_layers}")
        layer_pairs.append((text_layer, decoder_layer))

    if not layer_pairs:
        raise ValueError("No valid text/decode layer pairs configured.")
    return layer_pairs


def parse_text_recon_layers(text_recon_cfg, decoder_num_layers=None):
    raw_layers = text_recon_cfg.get("layers", None)
    if not raw_layers:
        raw_pairs = text_recon_cfg.get("layer_pairs", None)
        if raw_pairs:
            raw_layers = [pair[1] for pair in raw_pairs]
    if not raw_layers:
        return []
    layers = [int(layer) for layer in raw_layers]
    if len(set(layers)) != len(layers):
        raise ValueError(f"text_recon_conditioning.layers must not contain duplicates, got {layers}")
    if decoder_num_layers is not None:
        out_of_range = [layer for layer in layers if layer < 0 or layer >= decoder_num_layers]
        if out_of_range:
            raise ValueError(
                f"text_recon_conditioning.layers out of range for decoder_num_layers={decoder_num_layers}: {out_of_range}"
            )
    return layers


def append_csv_row(path, fieldnames, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def compute_reconstruction_metrics(
        vq_model,
        val_loader,
        device,
        args,
        causal_type,
        text_tokenizer=None,
        text_encoder=None,
        text_layer_pairs=None,
        text_recon_layer_pairs=None,
        text_injection_layers=None,
        text_max_length=None,
        mixed_precision_dtype=None,
):
    vq_model.eval()
    mse_sum = torch.tensor(0.0, device=device)
    mae_sum = torch.tensor(0.0, device=device)
    elem_count = torch.tensor(0.0, device=device)
    image_count = torch.tensor(0.0, device=device)
    ssim_sum = torch.tensor(0.0, device=device)
    ssim_count = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for imgs, text_batch in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            selected_decoder_layer = None
            decoder_text_features = None
            decoder_text_features_by_layer = None
            decoder_text_key_padding_mask = None

            if text_tokenizer is not None:
                if text_encoder is None:
                    raise ValueError("Text-conditioned validation requires text_encoder.")
                if not isinstance(text_batch, (list, tuple)):
                    raise TypeError("Text-conditioned validation requires the validation dataset to return text strings.")

                text_inputs = text_tokenizer(
                    [str(text) for text in text_batch],
                    padding="max_length",
                    truncation=True,
                    max_length=text_max_length,
                    return_tensors="pt",
                )
                input_ids = text_inputs["input_ids"].to(device, non_blocking=True)
                text_attention_mask = text_inputs["attention_mask"].to(device, non_blocking=True)
                decoder_text_key_padding_mask = ~text_attention_mask.bool()
                with autocast_context(args.device_backend, args.mixed_precision, dtype=mixed_precision_dtype):
                    text_outputs = text_encoder(
                        input_ids=input_ids,
                        attention_mask=text_attention_mask,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                hidden_states = text_outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("T5 encoder did not return hidden_states during validation.")
                t5_layer_states = hidden_states[1:]
                if text_recon_layer_pairs:
                    decoder_text_features_by_layer = {}
                    for text_layer, decoder_layer in text_recon_layer_pairs:
                        if text_layer >= len(t5_layer_states):
                            raise ValueError(
                                f"text_layer={text_layer} is out of range for T5 layer outputs={len(t5_layer_states)}"
                            )
                        decoder_text_features_by_layer[int(decoder_layer)] = t5_layer_states[text_layer].detach()
                else:
                    if not text_layer_pairs:
                        raise ValueError("Text-conditioned validation requires text_layer_pairs.")
                    selected_text_layer, selected_decoder_layer = text_layer_pairs[0]
                    if selected_text_layer >= len(t5_layer_states):
                        raise ValueError(
                            f"selected_text_layer={selected_text_layer} is out of range for "
                            f"T5 layer outputs={len(t5_layer_states)}"
                        )
                    decoder_text_features = t5_layer_states[selected_text_layer].detach()

            with autocast_context(args.device_backend, args.mixed_precision, dtype=mixed_precision_dtype):
                outputs = vq_model(
                    imgs,
                    causal_type=causal_type,
                    selected_decoder_layer=selected_decoder_layer,
                    decoder_text_features=decoder_text_features,
                    decoder_text_features_by_layer=decoder_text_features_by_layer,
                    decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                    text_injection_layers=text_injection_layers,
                )
            recons = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            diff = (recons.float() - imgs.float()) * 0.5
            mse_sum += diff.pow(2).sum()
            mae_sum += diff.abs().sum()
            elem_count += diff.numel()
            image_count += imgs.shape[0]
            if args.val_compute_ssim and ssim_metric is not None:
                rec_np = torch.clamp((recons.float() + 1.0) * 0.5, 0.0, 1.0).detach().cpu().permute(0, 2, 3, 1).numpy()
                gt_np = torch.clamp((imgs.float() + 1.0) * 0.5, 0.0, 1.0).detach().cpu().permute(0, 2, 3, 1).numpy()
                for gt, rec in zip(gt_np, rec_np):
                    ssim_sum += float(ssim_metric(gt, rec, channel_axis=-1, data_range=1.0))
                    ssim_count += 1

    for value in (mse_sum, mae_sum, elem_count, image_count, ssim_sum, ssim_count):
        dist.all_reduce(value, op=dist.ReduceOp.SUM)

    mse = (mse_sum / elem_count.clamp_min(1.0)).item()
    mae = (mae_sum / elem_count.clamp_min(1.0)).item()
    psnr = -10.0 * np.log10(max(mse, 1e-12))
    val_ssim = float("nan")
    if ssim_count.item() > 0:
        val_ssim = (ssim_sum / ssim_count).item()
    return {
        "val_mse": mse,
        "val_mae": mae,
        "val_psnr": float(psnr),
        "val_ssim": val_ssim,
        "val_images": int(image_count.item()),
    }


def metric_is_better(value, best_value, mode):
    if not np.isfinite(value):
        return False
    if best_value is None:
        return True
    if mode == "min":
        return value < best_value
    if mode == "max":
        return value > best_value
    raise ValueError(f"Unsupported best mode: {mode}")


def save_training_checkpoint(path, vq_model, vq_loss, optimizer, optimizer_disc, train_steps, args, ema=None, compile_model=False):
    if compile_model:
        model_weight = vq_model.module._orig_mod.state_dict()
    else:
        model_weight = vq_model.module.state_dict()
    checkpoint = {
        "model": model_weight,
        "optimizer": optimizer.state_dict(),
        "discriminator": vq_loss.module.discriminator.state_dict(),
        "optimizer_disc": optimizer_disc.state_dict(),
        "steps": train_steps,
        "args": args
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, path)


def plot_train_val_curves(metrics_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    train_path = os.path.join(metrics_dir, "train_metrics.csv")
    val_path = os.path.join(metrics_dir, "val_metrics.csv")
    if not os.path.exists(train_path) and not os.path.exists(val_path):
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    if os.path.exists(train_path):
        train_steps = []
        train_losses = []
        with open(train_path, "r") as handle:
            for row in csv.DictReader(handle):
                train_steps.append(int(row["step"]))
                train_losses.append(float(row["train_loss"]))
        if train_steps:
            axes[0].plot(train_steps, train_losses, label="train_loss", linewidth=1.6)
    axes[0].set_ylabel("train_loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    val_steps = []
    val_ssim = []
    if os.path.exists(val_path):
        val_mse = []
        val_psnr = []
        with open(val_path, "r") as handle:
            for row in csv.DictReader(handle):
                val_steps.append(int(row["step"]))
                val_mse.append(float(row["val_mse"]))
                val_psnr.append(float(row["val_psnr"]))
                if "val_ssim" in row and row["val_ssim"] not in ("", "nan"):
                    val_ssim.append(float(row["val_ssim"]))
        if val_steps:
            axes[1].plot(val_steps, val_mse, label="val_mse", linewidth=1.6)
            ax2 = axes[1].twinx()
            ax2.plot(val_steps, val_psnr, label="val_psnr", color="tab:orange", linewidth=1.6)
            ax2.set_ylabel("val_psnr")
            ax2.legend(loc="upper right")
    axes[1].set_ylabel("val_mse")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left")

    if os.path.exists(val_path) and len(val_ssim) == len(val_steps) and val_steps:
        axes[2].plot(val_steps, val_ssim, label="val_ssim", color="tab:green", linewidth=1.6)
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("val_ssim")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    fig.suptitle("Stage-1 train / validation curves")
    fig.tight_layout()
    fig.savefig(os.path.join(metrics_dir, "train_val_curves.png"), dpi=160)
    plt.close(fig)


CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

SIGLIP_DEFAULT_MEAN = (0.5, 0.5, 0.5)
SIGLIP_DEFAULT_STD = (0.5, 0.5, 0.5)

def preprocess_raw_image(x, enc_type):
    """
    x range is [-1 , 1]
    from https://github.com/sihyun-yu/REPA/blob/main/train.py
    """
    if 'clip' in enc_type:
        x = (x + 1) / 2.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'siglip' in enc_type:
        x = (x + 1) / 2.
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
        x = Normalize(SIGLIP_DEFAULT_MEAN, SIGLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')
    elif 'dinov1' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224, mode='bicubic')

    return x





#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new model.
    """
    device_module = prepare_device_backend(args.device_backend)

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)


    # setting the constants used during training
    

    # the causal settings are the attributes of a model, and also necessary for training process
    causal_type = config["model"]["causal_settings"]["causal_type"]
    dynamic_length_train = config["model"]["causal_settings"]["dynamic_length_train"]
    min_level = config["model"]["causal_settings"]["min_level"]
    dynamic_level_range = config["model"]["causal_settings"].get("dynamic_level_range", None)
    power_sample_T = config["model"]["causal_settings"].get("power_sample_T", None)
    hr_on = config["trainer"].get("hr_on", False)
    hr_loss_weight = float(config["trainer"].get("hr_loss_weight", 0.0))
    hr_random_one_layer = config["trainer"].get("hr_random_one_layer", True)
    # Text-HR v2: separate config blocks make the text-conditioned path optional.
    # If both switches are off, the original GigaTok stage-1 path is used.
    text_conditioning_cfg = config.get("text_conditioning", {})
    text_recon_cfg = config.get("text_recon_conditioning", {})
    text_hr_cfg = config.get("text_hr", {})
    text_conditioning_on = bool(text_conditioning_cfg.get("enabled", False))
    text_recon_on = bool(text_recon_cfg.get("enabled", False))
    text_hr_on = bool(text_hr_cfg.get("enabled", False))
    text_recon_mode = str(text_recon_cfg.get("mode", "concat_memory"))
    visual_memory_mask_cfg = text_recon_cfg.get("visual_memory_mask", {})
    visual_memory_mask_enabled = bool(visual_memory_mask_cfg.get("enabled", False))
    visual_memory_mask_ratio = float(visual_memory_mask_cfg.get("ratio", 0.0))
    if text_recon_on and not text_conditioning_on:
        raise ValueError("text_recon_conditioning.enabled requires text_conditioning.enabled=True.")
    if text_hr_on and not text_conditioning_on:
        raise ValueError("text_hr.enabled requires text_conditioning.enabled=True.")
    if hr_on and text_hr_on:
        raise ValueError("v1 hr_on and v2 text_hr.enabled cannot be enabled at the same time.")
    if text_recon_on:
        if text_recon_mode not in {"concat_memory", "concat_memory_visual_mask"}:
            raise NotImplementedError(
                "Only text_recon_conditioning.mode=concat_memory or concat_memory_visual_mask is implemented."
            )
        if text_recon_mode == "concat_memory" and visual_memory_mask_enabled:
            raise ValueError("text_recon_conditioning.visual_memory_mask.enabled must be false for mode=concat_memory.")
        if text_recon_mode == "concat_memory_visual_mask" and not visual_memory_mask_enabled:
            raise ValueError(
                "text_recon_conditioning.visual_memory_mask.enabled must be true for "
                "mode=concat_memory_visual_mask."
            )
        if visual_memory_mask_enabled:
            if str(visual_memory_mask_cfg.get("mode", "learned_mask_token")) != "learned_mask_token":
                raise NotImplementedError("Only visual_memory_mask.mode=learned_mask_token is implemented.")
            if visual_memory_mask_ratio < 0.0 or visual_memory_mask_ratio >= 1.0:
                raise ValueError(
                    f"text_recon_conditioning.visual_memory_mask.ratio must be in [0, 1), "
                    f"got {visual_memory_mask_ratio}"
                )
        for block_name in ("residual", "adaln"):
            block_cfg = text_recon_cfg.get(block_name, {})
            if isinstance(block_cfg, dict) and bool(block_cfg.get("enabled", False)):
                raise NotImplementedError(f"text_recon_conditioning.{block_name} is planned but not implemented.")
    text_hr_loss_weight = float(text_hr_cfg.get("hr_loss_weight", 0.0)) if text_hr_on else 0.0
    text_hr_tau = float(text_hr_cfg.get("tau", 1.0))
    text_hr_svd_mode = str(text_hr_cfg.get("svd_mode", "frobenius_uniform"))
    text_hr_eps = float(text_hr_cfg.get("eps", 1e-8))
    text_hr_skip_if_valid_tokens_lt = int(text_hr_cfg.get("skip_if_valid_tokens_lt", 2))
    text_max_length = int(text_conditioning_cfg.get("max_length", 128))
    freeze_encoder = config["trainer"].get("freeze_encoder", False)
    freeze_quantizer = config["trainer"].get("freeze_quantizer", False)
    freeze_codebook = config["trainer"].get("freeze_codebook", False)
    freeze_post_quant_conv = config["trainer"].get("freeze_post_quant_conv", False)


    if args.sub_exp_dir is not None:
        experiment_dir = f"{args.save_path}/{args.sub_exp_dir}"  # Create an experiment folder
    else:
        raise ValueError("Please specify a sub_exp_dir")

    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints

    # configs for constant + cooldown training
    if config["trainer"]["lr_scheduler"] == "wsd":
        use_wsd = True
        fract_decay = args.fract_decay if args.fract_decay is not None \
                      else config["trainer"].get("fract_decay", 0.2)

        cd_sub_dir = experiment_dir + "/cd_records" + f"/cd_fract_{fract_decay}_to_{args.iterations:07d}"
        cd_checkpoint_dir = f"{cd_sub_dir}/checkpoints"

        # add extra wsd setting for trial_name
        # trial_name = args.model_config.split("/")[-1].replace(".yaml", "") + f"_cd_fract_{fract_decay}_to_{args.iterations:07d}"
        trial_name = args.sub_exp_dir
    else:
        # trial_name = args.model_config.split("/")[-1].replace(".yaml", "")
        trial_name = args.sub_exp_dir
        use_wsd = False
        fract_decay = 0


    args.global_batch_size = args.global_batch_size if args.global_batch_size is not None else config["trainer"]["global_batch_size"]


    
    # Setup DDP:
    init_distributed_mode(args)
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    node_rank = int(os.environ.get('NODE_RANK', 0))
    print("rank", rank)
    print("node_rank", node_rank)
    local_device = rank % device_module.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    device_module.set_device(local_device)
    device = torch.device(f"{args.device_backend}:{local_device}")



    # Setup data:
    textatlas_jsonl_on = args.dataset == "textatlas_image_text"
    if text_conditioning_on or textatlas_jsonl_on:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: resize_pad_arr(pil_image, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: random_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
    dataset = build_dataset(args, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = None
    val_sampler = None
    if args.val_json_path is not None:
        if text_conditioning_on or textatlas_jsonl_on:
            val_transform = transforms.Compose([
                transforms.Lambda(lambda pil_image: resize_pad_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        else:
            val_transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
            ])
        if args.dataset == "textatlas_image_text":
            val_args = deepcopy(args)
            val_args.data_path = args.val_json_path
            val_args.json_path = args.val_json_path
            val_args.max_images = args.val_max_images
            val_dataset = build_dataset(val_args, transform=val_transform)
        else:
            val_dataset = JsonImageDataset(
                args.val_json_path,
                transform=val_transform,
                max_images=args.val_max_images,
            )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.val_num_workers,
            pin_memory=True,
            drop_last=False
        )


    total_steps = args.iterations
    if use_wsd:
        _ , const_end_flag = wsd_find_newest_ckpt(
                                                config, 
                                                const_ckpt_dir=checkpoint_dir, 
                                                cd_sub_dir=cd_sub_dir, 
                                                total_steps=total_steps,
                                                fract_decay=fract_decay,
                                                )
    else:
        const_end_flag = False

    exp_dir = cd_sub_dir if (use_wsd and const_end_flag) else experiment_dir
    # Setup an experiment folder:
    if rank == 0 and node_rank == 0:

        os.makedirs(args.save_path, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        metrics_dir = os.path.join(exp_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        if use_wsd:
            os.makedirs(cd_checkpoint_dir, exist_ok=True)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment directory created at {exp_dir}")

        # wandb
        if not args.no_wandb:
            try:
                init_wandb(
                    project_name=wandb_project_name,
                    config={"dataset":"imagenet"},
                    exp_dir=exp_dir,
                    name=trial_name,
                    )
            except Exception as e:
                print(e)
                logger.info("wandb init failed, please check your wandb config")
                logger.info("Running without wandb")
                args.no_wandb = True
    else:
        logger = create_logger(None)
        metrics_dir = None


    # training args
    logger.info(f"{args}")
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    if val_loader is not None:
        logger.info(f"Validation dataset contains {len(val_loader.dataset):,} images ({args.val_json_path})")


    # training env
    logger.info(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}, "
        f"device_backend={args.device_backend}."
    )

    ######################################################################
    ## Load Teacher model for distillation
    ######################################################################
    if config["trainer"].get("distill_loss", False) is True:
        model_out_dim_dict = {
            "siglip-sovit-400m": 1152,
            "clip_dfn-vit-l/14": 1024
        }
        distill_model = config["trainer"].get("distill_model", "dinov2-vit-b")
        distill_encoder, encoder_type, architecture = load_encoders(distill_model, device)
        if "dinov2" in distill_model:
            config["model"]["init_args"]["out_inner_dim"] = distill_encoder.embed_dim
        else:
            # for open clip model, the output dim is different from the input dim
            try:
                config["model"]["init_args"]["out_inner_dim"] = distill_encoder.visual.output_dim
            except:
                # if cannot get from attributes, then refer to a hardcoded dict
                config["model"]["init_args"]["out_inner_dim"] = model_out_dim_dict[distill_model]

    text_tokenizer = None
    text_encoder = None
    text_encoder_num_layers = None
    text_feature_dim = None
    if text_conditioning_on:
        # Text-HR v2: T5 is frozen and only provides hidden states for decoder
        # cross-attention; gradients do not update T5.
        if AutoTokenizer is None or T5EncoderModel is None:
            raise ImportError("text_conditioning.enabled=True requires transformers with T5EncoderModel.")
        text_encoder_name = text_conditioning_cfg.get("encoder_name", "google/t5-v1_1-xl")
        text_cache_dir = text_conditioning_cfg.get("cache_dir", None)
        text_local_files_only = bool(text_conditioning_cfg.get("local_files_only", False))
        pretrained_kwargs = {
            "cache_dir": text_cache_dir,
            "local_files_only": text_local_files_only,
        }
        text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, **pretrained_kwargs)
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, **pretrained_kwargs)
        text_feature_dim = getattr(text_encoder.config, "d_model", None)
        text_encoder_num_layers = getattr(text_encoder.config, "num_layers", None)
        if text_feature_dim is None:
            raise ValueError(f"Cannot read d_model from T5 encoder config: {text_encoder_name}")
        if text_conditioning_cfg.get("freeze", True):
            text_encoder.requires_grad_(False)
        text_encoder.eval()
        text_encoder = text_encoder.to(device)
        logger.info(
            f"Text conditioning enabled: encoder={text_encoder_name}, "
            f"d_model={text_feature_dim}, num_layers={text_encoder_num_layers}, "
            f"max_length={text_max_length}, cache_dir={text_cache_dir}, "
            f"local_files_only={text_local_files_only}"
        )
 
    vq_model = load_model_from_config(config)
    if text_conditioning_on:
        # Text-HR v2: add the lightweight projection parameters after loading
        # the base model definition, before checkpoint compatibility loading.
        text_gate_cfg = text_recon_cfg.get("text_gate", {}) if text_recon_on else {}
        vq_model.configure_text_conditioning(
            text_feature_dim=text_feature_dim,
            text_projection=text_conditioning_cfg.get("text_projection", "linear_layernorm"),
            text_type_embedding=text_conditioning_cfg.get("text_type_embedding", True),
            visual_type_embedding=text_recon_on and bool(text_recon_cfg.get("visual_type_embedding", False)),
            text_gate_enabled=text_recon_on and bool(text_gate_cfg.get("enabled", False)),
            text_gate_init=float(text_gate_cfg.get("init", 0.1)),
            visual_memory_mask_enabled=text_recon_on and text_recon_mode == "concat_memory_visual_mask",
        )

    # create and load model
    logger.info(f"VQ Model Parameters(training): {sum(p.numel() for p in vq_model.parameters()):,}")
    # logger.info(f"VQ Model Transformer Encoder Parameters: {sum(p.numel() for p in vq_model.s2to1encoder.parameters()):,}")
    # logger.info(f"VQ Model Transformer Decoder Parameters: {sum(p.numel() for p in vq_model.s1to2decoder.parameters()):,}")
    if args.ema:
        ema = deepcopy(vq_model).to(device)  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        logger.info(f"VQ Model EMA Parameters: {sum(p.numel() for p in ema.parameters()):,}")
    vq_model = vq_model.to(device)


    logger.info(str(config))
    # Ensure all default values are set in the config dictionary beforehand or assume they are already set
    vq_loss = VQLoss(
        disc_start=config["loss"]["params"]["disc_start"],
        gen_start=config["loss"]["params"].get("gen_start", 0),
        disc_weight=config["loss"]["params"]["disc_weight"],
        disc_dim=config["loss"]["params"]["disc_dim"],
        disc_num_layers=config["loss"]["params"].get("disc_num_layers", 3),
        disc_type=config["loss"]["params"]["disc_type"],
        disc_loss=config["loss"]["params"]["disc_loss"],
        gen_adv_loss=config["loss"]["params"]["gen_adv_loss"],
        image_size=config["loss"]["params"]["image_size"],  # Assuming image_size is moved to config
        perceptual_weight=config["loss"]["params"]["perceptual_weight"],
        reconstruction_weight=config["loss"]["params"]["reconstruction_weight"],
        reconstruction_loss=config["loss"]["params"]["reconstruction_loss"],
        codebook_weight=config["loss"]["params"]["codebook_weight"],
        aux_loss_end=config["loss"]["params"]["aux_loss_end"],
        entropy_loss_end=config["loss"]["params"].get("entropy_loss_end", None),
        norm=config["loss"]["params"]["norm"],
        use_direct_rec_loss=config["loss"]["params"].get("use_direct_rec_loss", False),
        kw=config["loss"]["params"]["kw"],
        blur_ds=config["loss"]["params"].get("blur_ds", False),
        lecam=config["loss"]["params"].get("lecam", False),
        proj_weight=config["loss"]["params"].get("proj_weight", 0.5),
        # resnet_perceptual=config["loss"]["params"].get("resnet_perceptual", False),
        # dhead=config["loss"]["params"].get("dhead", 32),    # Deprecated
        disc_semantic_type=config["loss"]["params"].get("disc_semantic_type", "local"),
        use_semantic_input=config["loss"]["params"].get("use_semantic_input", False),
        perceptual_model=config["loss"]["params"].get("perceptual_model", "vgg"),
        gamma=config["loss"]["params"].get("gamma", 15),
        discriminator_device=device,
    ).to(device)

    logger.info(f"Discriminator Parameters: {sum(p.numel() for p in vq_loss.discriminator.parameters()):,}")

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device_backend == "cuda" and args.mixed_precision =='fp16'))
    scaler_disc = torch.cuda.amp.GradScaler(enabled=(args.device_backend == "cuda" and args.mixed_precision =='fp16'))

    ###################################
    # Setup optimizer
    ###################################
    # different lr for differnt parts
    if hasattr(vq_model.quantize, "prior_model") and (getattr(vq_model.quantize, "prior_model") is not None):
        model_other_param_dict = {pn: p for pn, p in vq_model.named_parameters() if "prior_model" not in pn}
        prior_param_dict = {pn: p for pn, p in getattr(vq_model.quantize, "prior_model").named_parameters()}
        model_other_param = [model_other_param_dict[pn] for pn in sorted(list(model_other_param_dict.keys()))]
        prior_param = [prior_param_dict[pn] for pn in sorted(list(prior_param_dict.keys()))]
        assert len(model_other_param) + len(prior_param) == len(list(vq_model.named_parameters()))
        logger.info(f"Prior Model Parameters: {sum(p.numel() for p in prior_param):,}")
        logger.info(f"Prior Model learnig rate: {float(config['trainer']['lr']) * config['trainer']['prior_lr_mult']}")

        model_optimizer_groups = [
            {'params': prior_param, 'lr': float(config['trainer']['lr']) * config['trainer']['prior_lr_mult'], "lr_mult": config['trainer']['prior_lr_mult']},
            {'params': model_other_param}
        ]
        # assert config["trainer"].get("prior_lr_mult", 1.0) == 1.0, "prior_lr_mult is not supported"
    else:
        model_optimizer_groups = vq_model.parameters()


    if config["trainer"].get("optimizer", None) == "Adam":
        optimizer = torch.optim.Adam(
            model_optimizer_groups,
            lr=float(config["trainer"]["lr"]), 
            betas=(config["trainer"]["beta1"], config["trainer"]["beta2"]),
            weight_decay=float(config["trainer"]["weight_decay"])
        )

        optimizer_disc = torch.optim.Adam(
            vq_loss.discriminator.parameters(), 
            lr=float(config["trainer"]["lr"]), 
            betas=(config["trainer"]["beta1"], config["trainer"]["beta2"]),
            weight_decay=float(config["trainer"]["weight_decay"])
        )
    elif config["trainer"].get("optimizer", None) == "AdamW":
        optimizer = torch.optim.AdamW(
            model_optimizer_groups,
            lr=float(config["trainer"]["lr"]), 
            betas=(config["trainer"]["beta1"], config["trainer"]["beta2"]),
            weight_decay=float(config["trainer"]["weight_decay"])
        )

        optimizer_disc = torch.optim.AdamW(
                            vq_loss.discriminator.parameters(), 
                            lr=float(config["trainer"]["lr"]), 
                            betas=(config["trainer"]["beta1"], config["trainer"]["beta2"]),
                            weight_decay=float(config["trainer"]["weight_decay"])
                            )
    else:
        raise ValueError(f"Optimizer {config['trainer'].get('optimizer', 'Adam')} not supported.")


    

    ######################################
    # Prepare models for training:
    ######################################
    pretrain_loaded_flag = False
    skip_model_optimizer_load = args.finetune and text_conditioning_on
    text_conditioning_missing_keys = []
    if skip_model_optimizer_load:
        for name, _ in vq_model.named_parameters():
            if name in {"text_type_embedding", "visual_type_embedding", "text_gate_logit", "visual_mask_token"} \
                    or name.startswith("text_projection."):
                text_conditioning_missing_keys.append(name)
        for name, _ in vq_model.named_buffers():
            if name in {"text_type_embedding", "visual_type_embedding", "text_gate_logit", "visual_mask_token"} \
                    or name.startswith("text_projection."):
                text_conditioning_missing_keys.append(name)
    if args.vq_ckpt:
        checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
        custom_load(vq_model, checkpoint["model"], ignore_missing_keys=text_conditioning_missing_keys)
        # vq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            # ema.load_state_dict(checkpoint["ema"])
            custom_load(ema, checkpoint["ema"], ignore_missing_keys=text_conditioning_missing_keys)
        if skip_model_optimizer_load:
            logger.info("Skipping model optimizer state for text-conditioned finetune.")
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
        vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = train_steps // (len(dataset) // args.global_batch_size)
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.vq_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        pretrain_loaded_flag = True
    
    # auto resume accurately
    elif len(int_prefix_paths(f"{checkpoint_dir}/*.pt")) != 0:

        if config["trainer"]["lr_scheduler"] == "wsd":
            # Is the constant training already ended?
            resume_checkpoint, const_end_flag = wsd_find_newest_ckpt(
                                                    config, 
                                                    const_ckpt_dir=checkpoint_dir, 
                                                    cd_sub_dir=cd_sub_dir, 
                                                    total_steps=total_steps,
                                                    fract_decay=fract_decay
                                                 )

            latest_checkpoint = resume_checkpoint
            checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        else:
            latest_checkpoint = max(int_prefix_paths(f"{checkpoint_dir}/*.pt"), key=get_int_prefix_value)
            checkpoint = torch.load(latest_checkpoint, map_location="cpu")

        custom_load(vq_model, checkpoint["model"])
        # vq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            # ema.load_state_dict(checkpoint["ema"])
            custom_load(ema, checkpoint["ema"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:
            if rank == 0 and node_rank == 0:
                print("Optimizer state_dict not found in the checkpoint")
                # print(optimizer.state_dict())
                print(checkpoint["optimizer"].keys())
                print(len(optimizer.state_dict()["param_groups"][0]["params"]))
                print(len(optimizer.state_dict()["param_groups"][1]["params"]))
                print(len(checkpoint["optimizer"]["param_groups"][0]["params"]))
                print(checkpoint["optimizer"]["param_groups"])
                print(optimizer.state_dict()["param_groups"])
            exit()
        try:
            vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        except:
            print("You are using a unmatching discriminator")
            print("Assuming you are using a pretrained model with a different discriminator")
            print("reinitialize related model and optimizer")

        if not args.finetune:
            train_steps = checkpoint["steps"] if "steps" in checkpoint else int(args.vq_ckpt.split('/')[-1].split('.')[0])
            start_epoch = train_steps // (len(dataset) // args.global_batch_size)
        else:
            train_steps = 0
            start_epoch = 0           
        del checkpoint
        logger.info(f"Resume training from checkpoint: {latest_checkpoint}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        with open(os.path.join(experiment_dir, "message.txt"), "w") as f:
            f.write(f"Resume training from checkpoint: {latest_checkpoint}")
        pretrain_loaded_flag = True
    
    elif config["model"].get("init_ckpt", None) is not None:
        # the starting ckpt is given according to the config
        # init only when no existing ckpts are found
        init_ckpt = config["model"]["init_ckpt"]
        checkpoint = torch.load(init_ckpt, map_location="cpu")
        custom_load(vq_model, checkpoint["model"], ignore_missing_keys=text_conditioning_missing_keys)
        # vq_model.load_state_dict(checkpoint["model"])
        if args.ema:
            # ema.load_state_dict(checkpoint["ema"])
            custom_load(ema, checkpoint["ema"], ignore_missing_keys=text_conditioning_missing_keys)
        if skip_model_optimizer_load:
            logger.info("Skipping model optimizer state for text-conditioned init checkpoint.")
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
        try:
            vq_loss.discriminator.load_state_dict(checkpoint["discriminator"])
            optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
        except:
            # Using a pretrained model before starting discriminator training, 
            # reinitialize discriminator
            pass
        train_steps = 0
        start_epoch = 0           
        del checkpoint
        logger.info(f"Init training from checkpoint: {init_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        with open(os.path.join(experiment_dir, "message.txt"), "w") as f:
            f.write(f"Resume training from checkpoint: {init_ckpt}")
            f.write(f"Initial state: steps={train_steps}, epochs={start_epoch}")
        pretrain_loaded_flag = True
        
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(ema, vq_model, decay=0)  # Ensure EMA is initialized with synced weights

    if freeze_encoder or freeze_quantizer or freeze_codebook or freeze_post_quant_conv:
        if not pretrain_loaded_flag:
            raise ValueError("decoder-only finetune requires a pretrained checkpoint or an existing resume checkpoint.")
        vq_model.apply_stage1_finetune_freeze(
            freeze_encoder=freeze_encoder,
            freeze_quantizer=freeze_quantizer,
            freeze_codebook=freeze_codebook,
            freeze_post_quant_conv=freeze_post_quant_conv,
        )
        trainable_params = sum(p.numel() for p in vq_model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in vq_model.parameters() if not p.requires_grad)
        logger.info(
            f"Stage-1 finetune freeze: freeze_encoder={freeze_encoder}, "
            f"freeze_quantizer={freeze_quantizer}, freeze_codebook={freeze_codebook}, "
            f"freeze_post_quant_conv={freeze_post_quant_conv}, "
            f"trainable_params={trainable_params:,}, frozen_params={frozen_params:,}"
        )
    


    # decide the causal token keeping schedule
    if causal_type == "per-level":
        max_num_1d_tokens = vq_model.config.num_latent_tokens
        if dynamic_level_range is None:
            n_level = int(np.round(np.log2(max_num_1d_tokens)))
            choices = [i for i in range(min_level, n_level + 1)]
        else:
            choices = [i for i in range(dynamic_level_range[0], dynamic_level_range[1] + 1, dynamic_level_range[0])]
        probs = [1] * len(choices)
    elif causal_type == "per-token":
        max_num_1d_tokens = vq_model.config.num_latent_tokens
        if dynamic_level_range is None:
            n_level = max_num_1d_tokens
            choices = [i for i in range(min_level, n_level + 1)]
        else:
            if len(dynamic_level_range) > 2:
                choices = dynamic_level_range
            else:
                choices = [i for i in range(dynamic_level_range[0], dynamic_level_range[1] + 1, dynamic_level_range[0])]
        probs = [1] * len(choices)
    else:
        assert causal_type is None
    
    decoder_num_layers = vq_model.s1to2decoder.num_layers
    text_layer_pairs = None
    text_recon_layer_pairs = None
    text_injection_layers = []
    text_hr_image_token_len = int(text_hr_cfg.get("image_token_len", vq_model.config.num_latent_tokens))
    if text_conditioning_on:
        text_layer_pairs = build_text_layer_pairs(
            text_hr_cfg,
            decoder_num_layers=decoder_num_layers,
            text_num_layers=text_encoder_num_layers,
        )
        logger.info(
            f"Text-conditioned decoder layer pairs [t5_layer, decoder_layer]={text_layer_pairs}; "
            f"image_token_len={text_hr_image_token_len}"
        )
    if text_recon_on:
        configured_layers = parse_text_recon_layers(text_recon_cfg, decoder_num_layers=decoder_num_layers)
        if not configured_layers:
            raise ValueError("text_recon_conditioning.enabled=True requires non-empty layers or layer_pairs.")
        text_recon_pairs_cfg = dict(text_recon_cfg)
        if not text_recon_pairs_cfg.get("layer_pairs", None):
            text_recon_pairs_cfg["layer_pairs"] = [[layer_idx, layer_idx] for layer_idx in configured_layers]
        text_recon_layer_pairs = build_text_layer_pairs(
            text_recon_pairs_cfg,
            decoder_num_layers=decoder_num_layers,
            text_num_layers=text_encoder_num_layers,
            config_name="text_recon_conditioning",
        )
        text_injection_layers = [decoder_layer for _, decoder_layer in text_recon_layer_pairs]
        if configured_layers and sorted(configured_layers) != sorted(text_injection_layers):
            raise ValueError(
                f"text_recon_conditioning.layers={configured_layers} must match decoder layers from "
                f"text_recon_conditioning.layer_pairs={text_recon_layer_pairs}"
            )
        if len(set(text_injection_layers)) != len(text_injection_layers):
            raise ValueError(f"text_recon_conditioning decoder layers must be unique: {text_recon_layer_pairs}")
        if text_hr_on:
            hr_decoder_layers = {decoder_layer for _, decoder_layer in text_layer_pairs}
            missing_hr_layers = sorted(hr_decoder_layers - set(text_injection_layers))
            if missing_hr_layers:
                raise ValueError(
                    f"text_hr decoder layers must be included in text_recon_conditioning.layers, "
                    f"missing={missing_hr_layers}"
                )
        logger.info(
            f"Text reconstruction conditioning enabled: text_recon_mode={text_recon_mode}, "
            f"text_injection_layers={text_injection_layers}, text_recon_layer_pairs={text_recon_layer_pairs}, "
            f"visual_memory_mask_enabled={visual_memory_mask_enabled}, "
            f"visual_memory_mask_ratio={visual_memory_mask_ratio}"
        )
    if args.compile:
        logger.info("compiling the model... (may take several minutes)")
        vq_model = torch.compile(vq_model) # requires PyTorch 2.0        
    
    vq_model = DDP(vq_model.to(device), device_ids=[args.gpu], find_unused_parameters=True)
    vq_model.train()
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode
    vq_loss = DDP(vq_loss.to(device), device_ids=[args.gpu])
    vq_loss.train()

    ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

    total_steps = args.iterations

    if config["trainer"]["lr_scheduler"] == "cosine":
        scheduler = cosine_lr(
                        optimizer, 
                        float(config["trainer"]["lr"]), 
                        int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup, 
                        total_steps)
        scheduler_disc = cosine_lr(
                            optimizer_disc, 
                            float(config["trainer"]["lr"]), 
                            int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup, 
                            total_steps)
    elif config["trainer"]["lr_scheduler"] == "const":
        scheduler = const_lr(
                        optimizer, 
                        float(config["trainer"]["lr"]), 
                        int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup, 
                        total_steps)
        scheduler_disc = const_lr(
                            optimizer_disc, 
                            float(config["trainer"]["lr"]), 
                            int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup, 
                            total_steps)
    elif config["trainer"]["lr_scheduler"] == "cosine_v2":
        scheduler = cosine_schedule_with_warmup_v2(
                        optimizer,
                        float(config["trainer"]["lr"]),
                        int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup,
                        total_steps,
                        end_lr=float(config["trainer"].get("end_lr", 1e-5))
                    )
        scheduler_disc = cosine_schedule_with_warmup_v2(
                            optimizer_disc,
                            float(config["trainer"]["lr"]),
                            int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup,
                            total_steps,
                            end_lr=float(config["trainer"].get("end_lr", 1e-5))
                        )
    elif config["trainer"]["lr_scheduler"] == "wsd":
        scheduler = wsd_lr(
                        optimizer,
                        base_lr=float(config["trainer"]["lr"]),
                        warmup_length=int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup,
                        steps=total_steps,
                        fract_decay=float(fract_decay),
                    )
        

        scheduler_disc = wsd_lr(
                            optimizer_disc,
                            base_lr=float(config["trainer"]["lr"]),
                            warmup_length=int(config["trainer"]["warmup"]) if args.warmup is None else args.warmup,
                            steps=total_steps,
                            fract_decay=float(fract_decay),
                        )
    else:
        logging.error(
            f'Unknown scheduler, {config["trainer"]["lr_scheduler"]}. Available options are: cosine, const, const-cooldown.')
        exit(1)


    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_step_one_epoch = train_steps % (len(dataset) // args.global_batch_size)
    start_time = time.time()

    wandb_updates = []
    epochs = ceil(total_steps / (len(dataset) // args.global_batch_size))
    best_val_metric = None

    logger.info(f"Training for {epochs} epochs...")
    if use_wsd:
        hold_iterations = int(total_steps * (1 - fract_decay))
    breaking_flag = False
    for epoch in range(start_epoch, epochs):
        if args.iterations is not None and args.iterations <= train_steps:
            breaking_flag = True
            break

        if args.early_stop_iter is not None and args.early_stop_iter <= train_steps:
            breaking_flag = True
            break

        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        if epoch == start_epoch:
            loader_iter = iter(loader)
            loader_iter = islice(loader_iter, start_step_one_epoch, None)
        else:
            loader_iter = iter(loader)

        for x, y in loader_iter:
            if args.early_stop_iter is not None and args.early_stop_iter <= train_steps:
                breaking_flag = True
                break

            imgs = x.to(device, non_blocking=True)

            # dynamic token length training setting, depracated
            if causal_type is not None and dynamic_length_train:
                random.seed(train_steps)
                if power_sample_T is not None:
                    probs = [(l/max(choices))**(epoch / power_sample_T) for l in choices]
                num_en_q_level = random.choices(choices, weights=probs)[0]
                # np.random.choice(choices, p=probs, seed=train_steps)
            else:
                num_en_q_level = None

            selected_text_layer = None
            decoder_text_features = None
            decoder_text_features_by_layer = None
            decoder_text_key_padding_mask = None
            text_attention_mask = None
            text_recon_stats = None

            if text_conditioning_on:
                pair_rng = random.Random(train_steps + 1 + args.global_seed)
                # Text-HR v2: randomly choose one [T5 layer, decoder layer] pair
                # per training step, matching the current experimental design.
                if text_hr_cfg.get("random_one_pair_per_step", True):
                    selected_text_layer, selected_decoder_layer = text_layer_pairs[
                        pair_rng.randrange(len(text_layer_pairs))
                    ]
                else:
                    selected_text_layer, selected_decoder_layer = text_layer_pairs[0]

                if not isinstance(y, (list, tuple)):
                    raise TypeError(
                        "text_conditioning.enabled=True requires the dataset to return a batch of text strings."
                    )
                texts = [str(text) for text in y]
                text_inputs = text_tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=text_max_length,
                    return_tensors="pt",
                )
                input_ids = text_inputs["input_ids"].to(device, non_blocking=True)
                text_attention_mask = text_inputs["attention_mask"].to(device, non_blocking=True)
                decoder_text_key_padding_mask = ~text_attention_mask.bool()
                with torch.no_grad():
                    with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
                        text_outputs = text_encoder(
                            input_ids=input_ids,
                            attention_mask=text_attention_mask,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                hidden_states = text_outputs.hidden_states
                if hidden_states is None:
                    raise RuntimeError("T5 encoder did not return hidden_states.")
                t5_layer_states = hidden_states[1:]
                if text_recon_on:
                    decoder_text_features_by_layer = {}
                    for text_layer, decoder_layer in text_recon_layer_pairs:
                        if text_layer >= len(t5_layer_states):
                            raise ValueError(
                                f"text_layer={text_layer} is out of range for T5 layer outputs={len(t5_layer_states)}"
                            )
                        decoder_text_features_by_layer[int(decoder_layer)] = t5_layer_states[text_layer].detach()
                else:
                    if selected_text_layer >= len(t5_layer_states):
                        raise ValueError(
                            f"selected_text_layer={selected_text_layer} is out of range for "
                            f"T5 layer outputs={len(t5_layer_states)}"
                        )
                    # Text-HR v2 legacy: this hidden state is projected inside
                    # VQVitModelPlus and injected only into selected_decoder_layer.
                    decoder_text_features = t5_layer_states[selected_text_layer].detach()
                if text_recon_on:
                    text_recon_stats = {
                        "text_recon_enabled": torch.tensor(1.0, device=device),
                        "text_injection_layer_count": torch.tensor(float(len(text_injection_layers)), device=device),
                        "empty_text_count": torch.tensor(
                            float(sum(1 for text in texts if not str(text).strip())), device=device),
                        "text_valid_tokens_mean": text_attention_mask.float().sum(dim=1).mean(),
                    }
            elif hr_on:
                if hr_random_one_layer:
                    layer_rng = random.Random(train_steps + 1 + args.global_seed)
                    selected_decoder_layer = layer_rng.randrange(decoder_num_layers)
                else:
                    selected_decoder_layer = 0
            else:
                selected_decoder_layer = None

            # generator training
            optimizer.zero_grad()
            # ptdtype = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.mixed_precision]

            # if use_wsd:
            #     # update the wsd state for constant stage
            #     const_end_flag = train_steps >= hold_iterations
            #     # also update the exp_dir
            #     exp_dir = cd_sub_dir if (use_wsd and const_end_flag) else experiment_dir


            if config["trainer"].get("freeze_but_decoder_iter", None) is not None:
                if train_steps >= config["trainer"]["freeze_but_decoder_iter"] \
                    and not vq_model.module.freeze_but_2d_decoder_flag:
                    vq_model.module.freeze_but_2d_decoder()
                    
                    # disable distill_loss
                    config["trainer"]["distill_loss"] = False

            # prepare the distillive features
            if config["trainer"].get("distill_loss", False) is True:
                with torch.no_grad():
                    with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
                        # if preprocess is defined as a local variable then use it
                           # use REPA style code
                        raw_image_ = preprocess_raw_image(imgs, encoder_type)
                        
                        # encode image
                        if "dinov2" in encoder_type:
                            z = distill_encoder.forward_features(raw_image_)
                            z = z['x_norm_patchtokens']
                        elif "siglip" in encoder_type:
                            outputs = distill_encoder(pixel_values=raw_image_)
                            z = outputs.last_hidden_state
                        elif "clip" in encoder_type:
                            z = distill_encoder(raw_image_)[-1]
            else:
                z = None


            with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):  
                if config["trainer"].get("distill_loss", False) is True:
                    vq_outputs = vq_model(
                        imgs,
                        causal_type=causal_type,
                        num_en_q_level=num_en_q_level,
                        rec_loss=train_steps+1 < config["loss"]["params"]["aux_loss_end"],
                        ret_inner_feat=True,
                        random_mix_reg=config["trainer"].get("random_mix_reg", False),
                        replace_ratio=config["trainer"].get("replace_ratio", None),
                        global_step=train_steps+1,
                        max_steps=total_steps,
                        selected_decoder_layer=selected_decoder_layer,
                        decoder_text_features=decoder_text_features,
                        decoder_text_features_by_layer=decoder_text_features_by_layer,
                        decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                        text_injection_layers=text_injection_layers if text_recon_on else None,
                        visual_memory_mask_enabled=text_recon_on and visual_memory_mask_enabled,
                        visual_memory_mask_ratio=visual_memory_mask_ratio,
                        return_text_recon_stats=text_recon_on,
                    )
                    if selected_decoder_layer is not None:
                        if text_recon_on:
                            recons_imgs, inter_loss_set, inner_feat, hr_attn_weights, model_text_recon_stats = vq_outputs
                            text_recon_stats.update(model_text_recon_stats)
                        else:
                            recons_imgs, inter_loss_set, inner_feat, hr_attn_weights = vq_outputs
                    else:
                        if text_recon_on:
                            recons_imgs, inter_loss_set, inner_feat, model_text_recon_stats = vq_outputs
                            text_recon_stats.update(model_text_recon_stats)
                        else:
                            recons_imgs, inter_loss_set, inner_feat = vq_outputs
                        hr_attn_weights = None
                else:
                    vq_outputs = vq_model(
                        imgs,
                        causal_type=causal_type,
                        num_en_q_level=num_en_q_level,
                        rec_loss=train_steps+1 < config["loss"]["params"]["aux_loss_end"],
                        random_mix_reg=config["trainer"].get("random_mix_reg", False),
                        replace_ratio=config["trainer"].get("replace_ratio", None),
                        global_step=train_steps+1,
                        max_steps=total_steps,
                        selected_decoder_layer=selected_decoder_layer,
                        decoder_text_features=decoder_text_features,
                        decoder_text_features_by_layer=decoder_text_features_by_layer,
                        decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                        text_injection_layers=text_injection_layers if text_recon_on else None,
                        visual_memory_mask_enabled=text_recon_on and visual_memory_mask_enabled,
                        visual_memory_mask_ratio=visual_memory_mask_ratio,
                        return_text_recon_stats=text_recon_on,
                    )
                    if selected_decoder_layer is not None:
                        if text_recon_on:
                            recons_imgs, inter_loss_set, hr_attn_weights, model_text_recon_stats = vq_outputs
                            text_recon_stats.update(model_text_recon_stats)
                        else:
                            recons_imgs, inter_loss_set, hr_attn_weights = vq_outputs
                    else:
                        if text_recon_on:
                            recons_imgs, inter_loss_set, model_text_recon_stats = vq_outputs
                            text_recon_stats.update(model_text_recon_stats)
                        else:
                            recons_imgs, inter_loss_set = vq_outputs
                        hr_attn_weights = None
                    inner_feat = None
                loss_gen = vq_loss(inter_loss_set, imgs, recons_imgs, exp_dir=exp_dir, optimizer_idx=0, global_step=train_steps+1, 
                                   last_layer=None,
                                   logger=logger, log_every=args.log_every, ckpt_every=args.ckpt_every,
                                   causal_type=causal_type, num_en_q_level=num_en_q_level,
                                   inner_feat=inner_feat,
                                   sem_enc_feat=z,
                                   hr_attn_weights=hr_attn_weights if hr_on else None,
                                   hr_loss_weight=hr_loss_weight if hr_on else 0.0,
                                   selected_layer=selected_decoder_layer,
                                   # Text-HR v2: pass the selected layer attention
                                   # and text mask into VQLoss for image-to-text SVD.
                                   text_hr_attn_weights=hr_attn_weights if text_hr_on else None,
                                   text_attention_mask=text_attention_mask if text_hr_on else None,
                                   text_hr_loss_weight=text_hr_loss_weight if text_hr_on else 0.0,
                                   selected_text_layer=selected_text_layer,
                                   text_hr_tau=text_hr_tau,
                                   text_hr_svd_mode=text_hr_svd_mode,
                                   text_hr_eps=text_hr_eps,
                                   text_hr_skip_if_valid_tokens_lt=text_hr_skip_if_valid_tokens_lt,
                                   text_hr_image_token_len=text_hr_image_token_len,
                                   text_recon_stats=text_recon_stats,
                                   )
            
            if train_steps + 1 >= int(config["loss"]["params"].get("gen_start", 0)):
                scaler.scale(loss_gen).backward()
                if config["trainer"]["max_grad_norm"] != 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(vq_model.parameters(), config["trainer"]["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                current_lr = scheduler(train_steps)
                if args.ema:
                    update_ema(ema, vq_model.module._orig_mod if args.compile else vq_model.module)
            else:
                loss_gen = torch.tensor(0.0, device=device)
                current_lr = scheduler(train_steps)

            # discriminator training            
            if train_steps + 1 >= int(config["loss"]["params"]["disc_start"]):
                optimizer_disc.zero_grad()
                with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
                    loss_disc = vq_loss(inter_loss_set, imgs, recons_imgs, optimizer_idx=1, global_step=train_steps+1, exp_dir=exp_dir,
                                        logger=logger, log_every=args.log_every, ckpt_every=args.ckpt_every, sem_enc_feat=z)
                scaler_disc.scale(loss_disc).backward()
                if config["trainer"]["max_grad_norm"] != 0.0:
                    scaler_disc.unscale_(optimizer_disc)
                    torch.nn.utils.clip_grad_norm_(vq_loss.module.discriminator.parameters(), config["trainer"]["max_grad_norm"])
                scaler_disc.step(optimizer_disc)
                scaler_disc.update()
                current_lr_disc = scheduler_disc(train_steps)
            else:
                loss_disc = torch.tensor(0.0, device=device)
                current_lr_disc = 0
                
            # # Log loss values:
            running_loss += loss_gen.item() + loss_disc.item()


            if use_wsd:
                # update the wsd state for constant stage
                # for the last constant iteration, constant_end_flag is False
                # so that the ckpts will be saved in the constant dir
                const_end_flag = train_steps >= hold_iterations
                # also update the exp_dir
                exp_dir = cd_sub_dir if (use_wsd and const_end_flag) else experiment_dir
            
            log_steps += 1
            train_steps += 1



            if train_steps % args.log_every == 0:
                # Measure training speed:
                synchronize_device(args.device_backend)
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}, LR: {current_lr:.4e}, LR_disc: {current_lr_disc:.4e}")
                if rank == 0 and node_rank == 0:
                    append_csv_row(
                        os.path.join(metrics_dir, "train_metrics.csv"),
                        ["step", "epoch", "train_loss", "steps_per_sec", "lr", "lr_disc"],
                        {
                            "step": train_steps,
                            "epoch": epoch,
                            "train_loss": avg_loss,
                            "steps_per_sec": steps_per_sec,
                            "lr": current_lr,
                            "lr_disc": current_lr_disc,
                        }
                    )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

                wandb_updates.append(
                    {
                        "train_loss": avg_loss,
                        "train_steps_per_sec": steps_per_sec,
                        "iteration": train_steps,
                        "epoch": epoch,
                        "lr": current_lr,
                        "lr_disc": current_lr_disc,
                    }
                )

            if (
                val_loader is not None
                and args.val_every is not None
                and args.val_every > 0
                and train_steps % args.val_every == 0
                and train_steps > 0
            ):
                val_start_time = time.time()
                val_metrics = compute_reconstruction_metrics(
                    vq_model=vq_model,
                    val_loader=val_loader,
                    device=device,
                    args=args,
                    causal_type=causal_type,
                    text_tokenizer=text_tokenizer if text_conditioning_on else None,
                    text_encoder=text_encoder if text_conditioning_on else None,
                    text_layer_pairs=text_layer_pairs if text_conditioning_on else None,
                    text_recon_layer_pairs=text_recon_layer_pairs if text_recon_on else None,
                    text_injection_layers=text_injection_layers if text_recon_on else None,
                    text_max_length=text_max_length if text_conditioning_on else None,
                    mixed_precision_dtype=ptdtype,
                )
                metric_value = val_metrics[args.best_metric]
                is_best = metric_is_better(metric_value, best_val_metric, args.best_mode)
                if is_best:
                    best_val_metric = metric_value

                if rank == 0 and node_rank == 0:
                    logger.info(
                        f"(step={train_steps:07d}) Val MSE: {val_metrics['val_mse']:.6f}, "
                        f"Val MAE: {val_metrics['val_mae']:.6f}, "
                        f"Val PSNR: {val_metrics['val_psnr']:.4f}, "
                        f"Val SSIM: {val_metrics['val_ssim']:.4f}, "
                        f"best={is_best}"
                    )
                    append_csv_row(
                        os.path.join(metrics_dir, "val_metrics.csv"),
                        ["step", "epoch", "val_mse", "val_mae", "val_psnr", "val_ssim", "val_images", "is_best"],
                        {
                            "step": train_steps,
                            "epoch": epoch,
                            "val_mse": val_metrics["val_mse"],
                            "val_mae": val_metrics["val_mae"],
                            "val_psnr": val_metrics["val_psnr"],
                            "val_ssim": val_metrics["val_ssim"],
                            "val_images": val_metrics["val_images"],
                            "is_best": int(is_best),
                        }
                    )
                    if args.save_best and is_best:
                        save_ckpt_dir = cd_checkpoint_dir if (use_wsd and const_end_flag) else checkpoint_dir
                        best_path = f"{save_ckpt_dir}/best.pt"
                        save_training_checkpoint(
                            best_path,
                            vq_model,
                            vq_loss,
                            optimizer,
                            optimizer_disc,
                            train_steps,
                            args,
                            ema=ema if args.ema else None,
                            compile_model=args.compile,
                        )
                        logger.info(f"Saved best checkpoint to {best_path}")

                vq_model.train()
                dist.barrier()
                start_time += time.time() - val_start_time

            # Save checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0 and node_rank == 0:
                    if args.compile:
                        model_weight = vq_model.module._orig_mod.state_dict()
                    else:
                        model_weight = vq_model.module.state_dict()  
                    checkpoint = {
                        "model": model_weight,
                        "optimizer": optimizer.state_dict(),
                        "discriminator": vq_loss.module.discriminator.state_dict(),
                        "optimizer_disc": optimizer_disc.state_dict(),
                        "steps": train_steps,
                        "args": args
                    }
                    if args.ema:
                        checkpoint["ema"] = ema.state_dict()
                    
                    save_ckpt_dir = cd_checkpoint_dir if (use_wsd and const_end_flag) else checkpoint_dir
                    checkpoint_path = f"{save_ckpt_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

                    if not args.no_wandb:
                        wandb_cache_file_append(wandb_updates, exp_dir=exp_dir)
                        upload_wandb_cache(exp_dir=exp_dir)
                        wandb.finish()
                        init_wandb(
                            project_name=wandb_project_name,
                            config={"dataset":"imagenet"},
                            name=trial_name,
                            exp_dir=exp_dir,
                            )
                        
                    manage_ckpt_num(
                        save_ckpt_dir,
                        milestone_step=args.milestone_step,
                        milestone_start=args.milestone_start,
                        max_milestone_num=args.max_milestone_num
                    )

                wandb_updates = [] 
                dist.barrier()

            if args.iterations is not None and args.iterations <= train_steps:
                breaking_flag = True
                break

            if args.early_stop_iter is not None and args.early_stop_iter <= train_steps:
                breaking_flag = True
                break
        
        if breaking_flag:
            break

    dist.barrier()
    if args.save_last and rank == 0 and node_rank == 0:
        final_ckpt_dir = cd_checkpoint_dir if (use_wsd and const_end_flag) else checkpoint_dir
        last_path = f"{final_ckpt_dir}/last.pt"
        save_training_checkpoint(
            last_path,
            vq_model,
            vq_loss,
            optimizer,
            optimizer_disc,
            train_steps,
            args,
            ema=ema if args.ema else None,
            compile_model=args.compile,
        )
        logger.info(f"Saved last checkpoint to {last_path}")
    if rank == 0 and node_rank == 0:
        if metrics_dir is not None:
            plot_train_val_curves(metrics_dir)
    vq_model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    if rank == 0 and node_rank == 0 and (not args.no_wandb):
        wandb.finish()
    dist.destroy_process_group()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--data-face-path", type=str, default=None, help="face datasets to improve vq model")
    parser.add_argument("--save-path", type=str, default="results_tokenizer_image")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", action='store_true', help="finetune a pre-trained vq model")
    parser.add_argument("--ema", action='store_true', help="whether using ema training")

    # loss configs
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")

    parser.add_argument("--compile", action='store_true', default=False)

    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--json-path", type=str,
                        help="When given json path for dataset, all the images in the json will be used for training.")
    parser.add_argument("--max-images", type=int, default=0, help="Limit training images for manifest datasets. 0 means all rows.")

    # training configs
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device-backend", type=str, default="cuda", choices=["cuda", "npu"])
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.add_argument("--aux-loss-end", type=int, default=None, help="iteration to stop using auxiliary loss")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--fract-decay", type=float, default=0.2, 
                        help="fraction of the total iterations to decay the learning rate, \
                        used when lr-scheduler=wsd")


    # log and ckpts
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--val-json-path", type=str, default=None, help="JSON list of validation image paths for online reconstruction eval.")
    parser.add_argument("--val-every", type=int, default=0, help="Run online validation every N train steps. 0 disables online validation.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Per-rank validation batch size.")
    parser.add_argument("--val-num-workers", type=int, default=4, help="Validation dataloader workers per rank.")
    parser.add_argument("--val-max-images", type=int, default=0, help="Limit online validation images. 0 means all validation images.")
    parser.add_argument("--val-compute-ssim", action='store_true', help="Compute online validation SSIM. Slower than MSE/PSNR.")
    parser.add_argument("--save-best", action='store_true', help="Save checkpoints/best.pt according to online validation metric.")
    parser.add_argument("--save-last", action='store_true', help="Save checkpoints/last.pt at the end of training.")
    parser.add_argument("--best-metric", type=str, default="val_mse", choices=["val_mse", "val_mae", "val_psnr", "val_ssim"])
    parser.add_argument("--best-mode", type=str, default="min", choices=["min", "max"])
    parser.add_argument("--sub-exp-dir", type=str, default=None, help="sub experiment dir")
    parser.add_argument("--milestone-step", type=int, default=50_000, help="milestone step for checkpoint saving")
    parser.add_argument("--milestone-start", type=int, default=50_000, help="milestone start for checkpoint saving")
    parser.add_argument("--max-milestone-num", type=int, default=30, help="max milestone num for checkpoint saving")
    parser.add_argument("--wandb-project", type=str, default=None, help="wandb project name")
    parser.add_argument("--no-wandb", action='store_true', help="whether not using wandb")

    to_int_or_none = lambda x: int(x) if x != 'None' else None
    parser.add_argument("--early-stop-iter", type=to_int_or_none, default=None)

    parser.add_argument("--model-config", type=str, required=True)
    # parser.add_argument("--power-sample-T", type=float, default=None, help="temperature for power sampling")
    # probs = [(l/max(choices))**(epoch / T) for l in choices]

    args = parser.parse_args()

    wandb_project_name = args.wandb_project if args.wandb_project is not None else wandb_project_name
    main(args)
