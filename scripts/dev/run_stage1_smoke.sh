#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}}"
if [[ -f "${REPO_ROOT}/scripts/dev/source_stage1_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/scripts/dev/source_stage1_env.sh"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

echo "[stage1-smoke] DINOV2_REPO_DIR=${DINOV2_REPO_DIR:-}"
echo "[stage1-smoke] DINOV2_EXPECTED_COMMIT=${DINOV2_EXPECTED_COMMIT:-}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_RUN_PATH="${TORCH_RUN_PATH:-torchrun}"

SMOKE_MODE="${SMOKE_MODE:-train}"
MODEL_CONFIG="${MODEL_CONFIG:-configs/vq/VQ_BL256_dino_disc_hr.yaml}"
VQ_CKPT="${VQ_CKPT:-}"
export MODEL_CONFIG

SMOKE_ROOT="${SMOKE_ROOT:-${REPO_ROOT}/runs/stage1_smoke}"
SAVE_PATH="${SAVE_PATH:-${SMOKE_ROOT}/outputs}"
SUB_EXP_DIR="${SUB_EXP_DIR:-stage1_hr_smoke}"
DATA_PATH="${DATA_PATH:-}"

ITERATIONS="${ITERATIONS:-3}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
CKPT_EVERY="${CKPT_EVERY:-100000}"
LOG_EVERY="${LOG_EVERY:-1}"
PORT="${PORT:-29531}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

require_file() {
  local file_path="$1"
  local var_name="$2"
  if [[ -z "${file_path}" || ! -f "${file_path}" ]]; then
    echo "[stage1-smoke] ${var_name} must point to an existing file: ${file_path}" >&2
    exit 2
  fi
}

validate_iterations() {
  if (( ITERATIONS < 2 || ITERATIONS > 5 )); then
    echo "[stage1-smoke] ITERATIONS must be between 2 and 5, got ${ITERATIONS}" >&2
    exit 2
  fi
}

run_import_smoke() {
  echo "[stage1-smoke] mode=import"
  "${PYTHON_BIN}" - <<'PY'
import os
import yaml
import torch

from tokenizer.tokenizer_image.vq.vq_loss import high_rank_attention_loss

config_path = os.environ["MODEL_CONFIG"]
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

trainer = config["trainer"]
assert trainer["hr_on"] is True
assert trainer["hr_random_one_layer"] is True
assert float(trainer["hr_loss_weight"]) > 0
assert trainer["freeze_encoder"] is True
assert trainer["freeze_quantizer"] is True
assert trainer["freeze_codebook"] is True

attn = torch.softmax(torch.randn(1, 2, 8, 16), dim=-1)
hr_loss, uniformity = high_rank_attention_loss(attn)
print(f"[stage1-smoke] config={config_path}")
print(f"[stage1-smoke] dummy_attention_shape={tuple(attn.shape)}")
print(f"[stage1-smoke] hr_loss={hr_loss.item():.8f}")
print(f"[stage1-smoke] hr_spectrum_uniformity={uniformity.item():.8f}")
print("[stage1-smoke] import smoke passed")
PY
}

run_random_forward_smoke() {
  echo "[stage1-smoke] mode=random_forward"
  require_file "${MODEL_CONFIG}" "MODEL_CONFIG"
  require_file "${VQ_CKPT}" "VQ_CKPT"

  export MODEL_CONFIG VQ_CKPT GLOBAL_BATCH_SIZE IMAGE_SIZE
  "${PYTHON_BIN}" - <<'PY'
import os
import random
import yaml

import torch

from tokenizer.tokenizer_image.vq.vq_loss import high_rank_attention_loss
from utils.model_init import custom_load, load_model_from_config

model_config = os.environ["MODEL_CONFIG"]
vq_ckpt = os.environ["VQ_CKPT"]
batch_size = int(os.environ.get("GLOBAL_BATCH_SIZE", "1"))
image_size = int(os.environ.get("IMAGE_SIZE", "256"))

if not torch.cuda.is_available():
    raise RuntimeError("random_forward smoke requires one CUDA GPU for the B-L tokenizer")

with open(model_config, "r") as f:
    config = yaml.safe_load(f)

device = torch.device("cuda:0")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = load_model_from_config(config).to(device)
checkpoint = torch.load(vq_ckpt, map_location="cpu")
state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

def strip_prefix_if_present(state_dict, prefix):
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict

state = strip_prefix_if_present(state, "module.")
state = strip_prefix_if_present(state, "_orig_mod.")
custom_load(model, state)

trainer = config["trainer"]
model.apply_stage1_finetune_freeze(
    freeze_encoder=trainer.get("freeze_encoder", False),
    freeze_quantizer=trainer.get("freeze_quantizer", False),
    freeze_codebook=trainer.get("freeze_codebook", False),
)
model.train()

modules = [
    "encoder",
    "s2to1encoder",
    "quant_conv",
    "quantize",
    "post_quant_conv",
    "s1to2decoder",
    "decoder",
]

print("[stage1-smoke] frozen/trainable parameter summary")
for name in modules:
    module = getattr(model, name)
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    status = "trainable" if trainable > 0 else "frozen"
    print(f"[stage1-smoke] module={name} status={status} trainable={trainable} frozen={frozen}")

decoder_layers = model.s1to2decoder.num_layers
selected_layer_env = os.environ.get("SELECTED_LAYER")
if selected_layer_env is None:
    selected_layer = random.Random(1).randrange(decoder_layers)
else:
    selected_layer = int(selected_layer_env)
    if not 0 <= selected_layer < decoder_layers:
        raise ValueError(f"SELECTED_LAYER must be in [0, {decoder_layers}), got {selected_layer}")

x = torch.randn(batch_size, 3, image_size, image_size, device=device)
outputs = model(
    x,
    rec_loss=True,
    selected_decoder_layer=selected_layer,
    global_step=1,
    max_steps=5,
)

recons_imgs, inter_loss_set, attn_weights = outputs
if attn_weights is None:
    raise RuntimeError("selected decoder cross-attention weights were not returned")

hr_loss, uniformity = high_rank_attention_loss(attn_weights)
hr_loss_weight = float(trainer.get("hr_loss_weight", 0.0))
main_recon = recons_imgs[0] if isinstance(recons_imgs, (list, tuple)) else recons_imgs
dummy_recon_loss = main_recon.float().pow(2).mean()
total_loss = dummy_recon_loss + hr_loss_weight * hr_loss
total_loss.backward()

print(f"[stage1-smoke] total_loss={total_loss.item():.8f}")
print(f"[stage1-smoke] dummy_recon_loss={dummy_recon_loss.item():.8f}")
print(f"[stage1-smoke] hr_loss={hr_loss.item():.8f}")
print(f"[stage1-smoke] hr_loss_weight={hr_loss_weight:.8f}")
print(f"[stage1-smoke] selected_layer={selected_layer}")
print(f"[stage1-smoke] attention_shape={tuple(attn_weights.shape)}")
print(f"[stage1-smoke] hr_spectrum_uniformity={uniformity.item():.8f}")
print("[stage1-smoke] random forward/backward smoke passed")
PY
}

create_dummy_imagenet() {
  if [[ -n "${DATA_PATH}" ]]; then
    return
  fi

  DATA_PATH="${SMOKE_ROOT}/dummy_imagenet"
  export DATA_PATH
  mkdir -p "${DATA_PATH}/class0"

  if find "${DATA_PATH}/class0" -maxdepth 1 -type f | grep -q .; then
    return
  fi

  echo "[stage1-smoke] creating dummy ImageFolder at ${DATA_PATH}"
  "${PYTHON_BIN}" - <<'PY'
import os
from PIL import Image, ImageDraw

root = os.environ["DATA_PATH"]
class_dir = os.path.join(root, "class0")
os.makedirs(class_dir, exist_ok=True)

for idx in range(8):
    base = 32 + idx * 17
    img = Image.new("RGB", (288, 288), color=(base % 255, (base * 3) % 255, (base * 7) % 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((32, 32, 256, 256), outline=(255 - base % 255, 128, base % 255), width=6)
    draw.text((48, 48), f"smoke-{idx}", fill=(255, 255, 255))
    img.save(os.path.join(class_dir, f"{idx:03d}.png"))
PY
}

run_train_smoke() {
  echo "[stage1-smoke] mode=train"
  validate_iterations
  require_file "${MODEL_CONFIG}" "MODEL_CONFIG"
  require_file "${VQ_CKPT}" "VQ_CKPT"
  create_dummy_imagenet

  mkdir -p "${SAVE_PATH}"
  local log_file="${SMOKE_ROOT}/stage1_train_smoke.log"

  echo "[stage1-smoke] config=${MODEL_CONFIG}"
  echo "[stage1-smoke] checkpoint=${VQ_CKPT}"
  echo "[stage1-smoke] data_path=${DATA_PATH}"
  echo "[stage1-smoke] save_path=${SAVE_PATH}"
  echo "[stage1-smoke] log_file=${log_file}"

  "${TORCH_RUN_PATH}" \
    --nnodes=1 \
    --nproc_per_node=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${PORT}" \
    tokenizer/tokenizer_image/vq/vq_train.py \
    --data-path "${DATA_PATH}" \
    --dataset imagenet \
    --save-path "${SAVE_PATH}" \
    --vq-ckpt "${VQ_CKPT}" \
    --finetune \
    --iterations "${ITERATIONS}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --image-size "${IMAGE_SIZE}" \
    --mixed-precision "${MIXED_PRECISION}" \
    --log-every "${LOG_EVERY}" \
    --ckpt-every "${CKPT_EVERY}" \
    --sub-exp-dir "${SUB_EXP_DIR}" \
    --no-wandb \
    --model-config "${MODEL_CONFIG}" 2>&1 | tee "${log_file}"
}

case "${SMOKE_MODE}" in
  import)
    run_import_smoke
    ;;
  random_forward)
    run_random_forward_smoke
    ;;
  train)
    run_train_smoke
    ;;
  all)
    run_import_smoke
    run_random_forward_smoke
    run_train_smoke
    ;;
  *)
    echo "[stage1-smoke] unknown SMOKE_MODE=${SMOKE_MODE}; use import, random_forward, train, or all" >&2
    exit 2
    ;;
esac
