# Stage-1 HR Loss 云服务器环境与 Smoke Test

本文档面向当前服务器配置：

- GPU：RTX 4090 48G，单卡
- CPU：6 核
- 内存：50G
- 系统盘：200G
- 镜像：`pytorch-2.0.0-cuda12.1-ubuntu22.04`
- 仓库路径：`~/GigaTok_hr/GigaTok_Loss`

目标只覆盖 stage-1 主线：搭环境、验证官方 baseline reconstruction、验证当前 GigaTok B-L (dino disc) decoder attention HR loss smoke test。暂时不接入 TextAtlas5M，不跑正式训练，不改模型逻辑。

## 先做哪几步

1. 拉取或更新 `feat/hrloss-bl-textatlas` 分支。
2. 创建 conda 环境并安装依赖。
3. 设置缓存、checkpoint、输出目录环境变量。
4. 下载或放置官方 B-L (dino disc) tokenizer checkpoint。
5. 先跑官方 baseline reconstruction。
6. 再跑 stage-1 HR smoke test：先 `import`，再 `random_forward`，最后视 checkpoint 完整性决定是否跑 `train`。

最关键命令：

```bash
cd ~/GigaTok_hr/GigaTok_Loss
bash scripts/dev/setup_stage1_env.sh
conda activate gigatok-hr

export PROJECT_ROOT=$HOME/GigaTok_hr/GigaTok_Loss
export PERSIST_ROOT=$HOME/gigatok_persist
export CKPT_DIR=$PERSIST_ROOT/checkpoints
export VQ_CKPT=$CKPT_DIR/VQ_BL256_dino_disc.pt
export TORCH_RUN_PATH=$(which torchrun)
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

SMOKE_MODE=import bash scripts/dev/run_stage1_smoke.sh
SMOKE_MODE=random_forward GLOBAL_BATCH_SIZE=1 bash scripts/dev/run_stage1_smoke.sh
```

如果 `VQ_CKPT` 是包含 optimizer/discriminator 的完整训练 checkpoint，再跑：

```bash
SMOKE_MODE=train ITERATIONS=3 GLOBAL_BATCH_SIZE=1 NUM_WORKERS=0 bash scripts/dev/run_stage1_smoke.sh
```

## 1. 从 clone 仓库开始

如果服务器还没有仓库：

```bash
mkdir -p ~/GigaTok_hr
cd ~/GigaTok_hr
git clone -b feat/hrloss-bl-textatlas git@github.com:mayday4yl/GigaTok_Loss.git
cd GigaTok_Loss
```

如果服务器没有配置 GitHub SSH key，用 HTTPS：

```bash
mkdir -p ~/GigaTok_hr
cd ~/GigaTok_hr
git clone -b feat/hrloss-bl-textatlas https://github.com/mayday4yl/GigaTok_Loss.git
cd GigaTok_Loss
```

如果已经拉到了 `~/GigaTok_hr/GigaTok_Loss`：

```bash
cd ~/GigaTok_hr/GigaTok_Loss
git fetch origin
git switch feat/hrloss-bl-textatlas
git pull --ff-only
git status
git log -1 --oneline
```

确认关键文件存在：

```bash
ls configs/vq/VQ_BL256_dino_disc.yaml
ls configs/vq/VQ_BL256_dino_disc_hr.yaml
ls scripts/dev/run_stage1_smoke.sh
ls scripts/dev/setup_stage1_env.sh
```

## 2. 系统包与 Conda 环境

先做基础检查和系统包安装：

```bash
nvidia-smi
df -h
free -h

sudo apt-get update
sudo apt-get install -y git git-lfs wget curl unzip tmux htop libgl1 libglib2.0-0
```

然后创建 conda 环境并安装 Python 依赖。

推荐直接运行仓库脚本：

```bash
cd ~/GigaTok_hr/GigaTok_Loss
bash scripts/dev/setup_stage1_env.sh
conda activate gigatok-hr
```

脚本默认行为：

- 创建 conda 环境 `gigatok-hr`
- Python 版本：`3.9`
- 如果当前环境没有 `torch`，安装 PyTorch + CUDA 12.1 版本
- 安装 tokenizer reconstruction 和 stage-1 smoke test 需要的 Python 依赖
- 不安装 `flash-attn`，当前代码对它是可选依赖
- 默认不安装 TensorFlow，因为本文只跑 qualitative reconstruction 和 smoke test，不跑 quantitative FID

可选变量：

```bash
CONDA_ENV_NAME=gigatok-hr \
PYTHON_VERSION=3.9 \
INSTALL_TORCH=auto \
INSTALL_TF=0 \
PERSIST_ROOT=$HOME/gigatok_persist \
bash scripts/dev/setup_stage1_env.sh
```

如果后续要跑官方 quantitative reconstruction evaluator，再单独装 TensorFlow：

```bash
INSTALL_TF=1 bash scripts/dev/setup_stage1_env.sh
```

## 3. 环境变量设置

每次重新开 shell 后建议执行：

```bash
conda activate gigatok-hr

export PROJECT_ROOT=$HOME/GigaTok_hr/GigaTok_Loss
export PERSIST_ROOT=$HOME/gigatok_persist
export CKPT_DIR=$PERSIST_ROOT/checkpoints
export OUTPUT_DIR=$PERSIST_ROOT/outputs
export DATA_ROOT=$PERSIST_ROOT/datasets

export TORCH_HOME=$PERSIST_ROOT/cache/torch
export HF_HOME=$PERSIST_ROOT/cache/huggingface
export XDG_CACHE_HOME=$PERSIST_ROOT/cache/xdg
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export TORCH_RUN_PATH=$(which torchrun)
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true

mkdir -p "$CKPT_DIR" "$OUTPUT_DIR" "$DATA_ROOT" "$TORCH_HOME" "$HF_HOME" "$XDG_CACHE_HOME"
cd "$PROJECT_ROOT"
```

如果你要把缓存和 checkpoint 放到共享持久化存储，把 `PERSIST_ROOT` 换成实际挂载路径，例如：

```bash
export PERSIST_ROOT=/path/to/shared_storage/gigatok_stage1
```

## 4. Checkpoint 放置路径

本阶段需要官方 GigaTok B-L (dino disc) tokenizer checkpoint：

```bash
export VQ_CKPT=$CKPT_DIR/VQ_BL256_dino_disc.pt
```

推荐放置格式：

```text
$PERSIST_ROOT/checkpoints/VQ_BL256_dino_disc.pt
```

如果使用 `gdown` 从 Google Drive 下载官方 checkpoint：

```bash
mkdir -p "$CKPT_DIR"
gdown --fuzzy "https://drive.google.com/file/d/11bXyg6y15hIWYnIsl7YbwDtRAJMK7jjn/view?usp=drive_link" -O "$VQ_CKPT"
ls -lh "$VQ_CKPT"
```

如果 `gdown` 触发配额或权限问题，手动下载后上传到：

```text
$CKPT_DIR/VQ_BL256_dino_disc.pt
```

注意：

- 官方 release checkpoint 通常足够跑 reconstruction 和 `SMOKE_MODE=random_forward`。
- `SMOKE_MODE=train` 走现有 `vq_train.py --vq-ckpt` 路径，当前训练入口会读取 optimizer/discriminator state；如果你的 checkpoint 只有模型权重，`train` 模式可能失败。这种情况下先以 `random_forward` 作为 stage-1 最小 smoke 结论。

## 5. 先跑官方 baseline reconstruction

目的：确认官方 B-L (dino disc) tokenizer、checkpoint、CUDA、DDP、图片读写路径都能跑通。这里使用仓库自带 `tests/gt_qualitiative/` 小图片，不接 TextAtlas5M。

```bash
cd "$PROJECT_ROOT"

export TOK_CONFIG=configs/vq/VQ_BL256_dino_disc.yaml
export VQ_CKPT=$CKPT_DIR/VQ_BL256_dino_disc.pt
export DATA_PATH=$PROJECT_ROOT/tests
export SAMPLE_DIR=$OUTPUT_DIR/reconstruction_baseline_bl_dino

GPUS=1 \
PORT=29541 \
bash scripts/reconstruction.sh \
  --quant-way=vq \
  --data-path "$DATA_PATH" \
  --dataset imagenet \
  --image-size 256 \
  --sample-dir "$SAMPLE_DIR" \
  --vq-ckpt "$VQ_CKPT" \
  --model-config "$TOK_CONFIG" \
  --qualitative \
  --per-proc-batch-size 1 \
  --num-workers 0 \
  --clear-cache
```

预期：

- 终端打印 `Starting rank=0`。
- 打印 `Saving .png samples at ...`。
- `$SAMPLE_DIR` 下生成 reconstruction PNG 和 `_gt.png`。

如果要同时计算 LPIPS，可追加 `--lpips`，但首次运行会下载 LPIPS/VGG 权重，排错阶段可以先不加：

```bash
  --lpips
```

## 6. 再跑 stage-1 HR smoke test

主脚本：

```bash
scripts/dev/run_stage1_smoke.sh
```

默认配置：

```bash
configs/vq/VQ_BL256_dino_disc_hr.yaml
```

### 6.1 Import-level smoke

不需要 checkpoint，不需要数据，只验证 HR 配置和 loss 函数可 import：

```bash
cd "$PROJECT_ROOT"
SMOKE_MODE=import bash scripts/dev/run_stage1_smoke.sh
```

预期字段：

```text
[stage1-smoke] dummy_attention_shape=(1, 2, 8, 16)
[stage1-smoke] hr_loss=...
[stage1-smoke] hr_spectrum_uniformity=...
[stage1-smoke] import smoke passed
```

### 6.2 Random forward/backward smoke

需要 tokenizer checkpoint，不需要真实数据。该模式用随机图片 tensor 验证：

- B-L 模型可以 load checkpoint
- stage-1 freeze 生效
- selected decoder cross-attention weights 能返回
- HR loss 能计算
- total loss 能 backward
- 打印 `attention_shape`

```bash
cd "$PROJECT_ROOT"
export VQ_CKPT=$CKPT_DIR/VQ_BL256_dino_disc.pt

SMOKE_MODE=random_forward \
GLOBAL_BATCH_SIZE=1 \
IMAGE_SIZE=256 \
bash scripts/dev/run_stage1_smoke.sh
```

预期字段：

```text
[stage1-smoke] frozen/trainable parameter summary
[stage1-smoke] module=encoder status=frozen ...
[stage1-smoke] module=s2to1encoder status=frozen ...
[stage1-smoke] module=quant_conv status=frozen ...
[stage1-smoke] module=quantize status=frozen ...
[stage1-smoke] module=post_quant_conv status=trainable ...
[stage1-smoke] module=s1to2decoder status=trainable ...
[stage1-smoke] module=decoder status=trainable ...
[stage1-smoke] total_loss=...
[stage1-smoke] hr_loss=...
[stage1-smoke] selected_layer=...
[stage1-smoke] attention_shape=(...)
[stage1-smoke] hr_spectrum_uniformity=...
[stage1-smoke] random forward/backward smoke passed
```

### 6.3 Train-entry smoke

只有当 `VQ_CKPT` 是完整训练 checkpoint 时再跑这一层。它会复用现有 `vq_train.py` 入口，单卡跑 2 到 5 个 iteration。`DATA_PATH` 为空时，脚本会自动创建极小 `ImageFolder` dummy 数据。

```bash
cd "$PROJECT_ROOT"
export VQ_CKPT=$CKPT_DIR/VQ_BL256_dino_disc.pt

SMOKE_MODE=train \
ITERATIONS=3 \
GLOBAL_BATCH_SIZE=1 \
NUM_WORKERS=0 \
MIXED_PRECISION=bf16 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/dev/run_stage1_smoke.sh
```

如果 4090 上 `bf16` 报错，改成：

```bash
MIXED_PRECISION=fp16
```

训练入口预期字段：

```text
Stage-1 finetune freeze: freeze_encoder=True, freeze_quantizer=True, freeze_codebook=True, trainable_params=..., frozen_params=...
(Generator) ... hr_loss: ..., weighted_hr_loss: ..., selected_layer: ..., hr_spectrum_uniformity: ...
(step=0000001) Train Loss: ..., Train Steps/Sec: ..., LR: ...
Done!
```

字段对应关系：

- `Train Loss`：训练入口记录的 total loss 汇总。
- `hr_loss`：新增 high-rank loss 原值。
- `weighted_hr_loss`：`hr_loss_weight * hr_loss`。
- `selected_layer`：当前 step 被随机选中的 decoder layer。
- `hr_spectrum_uniformity`：谱均匀性指标。
- `Stage-1 finetune freeze`：全模型 frozen/trainable 参数总数。
- `attention_shape`：由 `SMOKE_MODE=random_forward` 打印；完整训练入口当前不额外打印 shape，避免为 smoke test 改训练逻辑。

## 7. DINO 依赖固定策略

当前 stage-1 训练入口涉及两类 DINO 依赖，它们不是同一件事：

- `DINOv2 repo commit`：固定 `torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")` 使用的模型定义源码。当前服务器将本地 repo 固定在 Python 3.9 可运行的官方 DINOv2 commit。
- `旧 DINO checkpoint sha256`：固定 `disc_type: "dinodisc"` 使用的 DINOv1 discriminator 权重文件 `dino_deitsmall16_pretrain.pth`。

当前约定路径：

```text
DINOv2 repo:
  /root/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main

DINOv2 ViT-B/14 weight:
  /root/gigatok_persist/cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth

DINOv1 dino-disc checkpoint:
  /root/gigatok_persist/cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth
```

标准环境加载方式：

```bash
cd /root/GigaTok_hr/GigaTok_Loss
conda activate gigatok
source scripts/dev/source_stage1_env.sh
```

`source_stage1_env.sh` 会设置当前 stage-1 所需路径，并固定：

```text
DINOV2_REPO_DIR=/root/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main
DINOV2_EXPECTED_COMMIT=4d83ffd4f4b39df112cef7ed7833de7ebf3a202e
```

`utils/model_init.py` 在 `DINOV2_REPO_DIR` 存在时会优先使用本地 repo：

```python
torch.hub.load(DINOV2_REPO_DIR, "dinov2_vitb14", source="local")
```

如果未设置 `DINOV2_REPO_DIR`，才回退到官方原行为 `torch.hub.load("facebookresearch/dinov2", ...)`。

训练或 smoke 前建议先做 DINO preflight：

```bash
cd /root/GigaTok_hr/GigaTok_Loss
conda activate gigatok
source scripts/dev/source_stage1_env.sh
bash scripts/dev/check_dino_deps.sh
```

预期关键输出：

```text
expected commit: 4d83ffd4f4b39df112cef7ed7833de7ebf3a202e
actual commit:   4d83ffd4f4b39df112cef7ed7833de7ebf3a202e
git status: clean
dino_deitsmall16_pretrain.pth: OK
embed_dim: 768
DINO dependency check passed
```

在服务器上记录 DINOv2 本地 repo 版本和状态：

```bash
cd /root/GigaTok_hr/GigaTok_Loss
conda activate gigatok

export PROJECT_ROOT=/root/GigaTok_hr/GigaTok_Loss
export PERSIST_ROOT=/root/gigatok_persist
export TORCH_HOME=$PERSIST_ROOT/cache/torch
export DINO_CACHE=$TORCH_HOME/hub/facebookresearch_dinov2_main

mkdir -p "$PERSIST_ROOT/versions"

{
  echo "repo=$DINO_CACHE"
  echo "head=$(git -C "$DINO_CACHE" rev-parse HEAD)"
  echo "oneline=$(git -C "$DINO_CACHE" log -1 --oneline)"
} > "$PERSIST_ROOT/versions/dinov2.commit"

{
  echo "repo=$DINO_CACHE"
  STATUS="$(git -C "$DINO_CACHE" status --short)"
  if [ -z "$STATUS" ]; then
    echo "git_status_short=(clean)"
  else
    printf "%s\n" "$STATUS"
  fi
} > "$PERSIST_ROOT/versions/dinov2.status"

cat "$PERSIST_ROOT/versions/dinov2.commit"
cat "$PERSIST_ROOT/versions/dinov2.status"
```

规范化旧 DINO discriminator checkpoint 的 checksum 记录：

```bash
cd /root/GigaTok_hr/GigaTok_Loss
conda activate gigatok

export PERSIST_ROOT=/root/gigatok_persist
export TORCH_HOME=$PERSIST_ROOT/cache/torch
export DINO1_CKPT=$TORCH_HOME/hub/checkpoints/dino_deitsmall16_pretrain.pth

mkdir -p "$PERSIST_ROOT/versions"

sha256sum "$DINO1_CKPT" > "$PERSIST_ROOT/versions/dino_deitsmall16_pretrain.sha256"
cat "$PERSIST_ROOT/versions/dino_deitsmall16_pretrain.sha256"
```

最小 DINOv2 本地加载测试，并保存输出：

```bash
cd /root/GigaTok_hr/GigaTok_Loss
conda activate gigatok

export PROJECT_ROOT=/root/GigaTok_hr/GigaTok_Loss
export PERSIST_ROOT=/root/gigatok_persist
export TORCH_HOME=$PERSIST_ROOT/cache/torch
export PYTHONPATH=$PROJECT_ROOT:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0

python - <<'PY' 2>&1 | tee "$PERSIST_ROOT/versions/dinov2.load_test.txt"
from utils.model_init import load_encoders

encoder, encoder_type, architecture = load_encoders("dinov2-vit-b", "cuda:0")
print("encoder_type:", encoder_type)
print("architecture:", architecture)
print("embed_dim:", encoder.embed_dim)
print("DINOv2 load through GigaTok ok")
PY
```

当前已验证的 DINOv2 固定版本：

```text
4d83ffd4f4b39df112cef7ed7833de7ebf3a202e
```

当前服务器已验证的 `SMOKE_MODE=train` 结果：

```text
mode=train
iterations=3
global_batch_size=1
mixed_precision=bf16
Using cache found in /root/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main
Stage-1 finetune freeze: trainable_params=480,559,363, frozen_params=141,077,768
step 1: hr_loss=4.8938e-04, selected_layer=4, hr_spectrum_uniformity=0.8742
step 2: hr_loss=2.2116e-04, selected_layer=1, hr_spectrum_uniformity=0.9432
step 3: hr_loss=7.3944e-04, selected_layer=7, hr_spectrum_uniformity=0.8100
Done!
```

`DDP find_unused_parameters=True` 可能打印性能 warning；当前 smoke 中不影响结论。

正式训练前不要执行以下操作，避免 DINOv2 repo 重新跟随 `main`：

```bash
git -C /root/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main pull
git -C /root/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main checkout main
rm -rf /root/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main
```

建议备份：

```text
/root/gigatok_persist/versions/dinov2.commit
/root/gigatok_persist/versions/dinov2.status
/root/gigatok_persist/versions/dinov2.load_test.txt
/root/gigatok_persist/versions/dino_deitsmall16_pretrain.sha256
/root/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main
/root/gigatok_persist/cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth
/root/gigatok_persist/cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth
```

## 8. 常见失败点

1. `ModuleNotFoundError`：重新执行 `bash scripts/dev/setup_stage1_env.sh`，并确认 `conda activate gigatok-hr`。
2. `torch.cuda.is_available()` 为 false：检查 `nvidia-smi`、镜像 CUDA、`CUDA_VISIBLE_DEVICES=0`。
3. `SMOKE_MODE=train` 加载 checkpoint 失败：官方 release checkpoint 可能没有 optimizer/discriminator state。先用 `SMOKE_MODE=random_forward` 完成最小验证。
4. LPIPS/VGG/DINO 下载失败：先不加 `--lpips` 跑 baseline reconstruction；HR `random_forward` 不需要真实数据。
5. 显存不足：保持 `GLOBAL_BATCH_SIZE=1`、`IMAGE_SIZE=256`、`NUM_WORKERS=0`。
6. `bf16` 报错：把 `MIXED_PRECISION` 改成 `fp16` 或 `none`。
7. Google Drive 下载失败：手动下载 checkpoint 后上传到 `$CKPT_DIR`。
8. 端口冲突：换端口，例如 `PORT=29601`。

## 9. 明确不做的事

- 不接入 TextAtlas5M。
- 不改 AR model。
- 不改模型逻辑。
- 不跑正式训练。
- 不做消融实验。
