# Stage-1 HR Loss 服务器最小 Smoke Test

本文档用于在云服务器上验证当前 GigaTok B-L (dino disc) tokenizer 的 decoder cross-attention high-rank loss 改动是否能正常运行。范围只覆盖 stage-1 主线：单卡、已有 checkpoint finetune、2 到 5 个 iteration、不接入 TextAtlas5M、不跑全量训练。

主脚本：

```bash
scripts/dev/run_stage1_smoke.sh
```

默认使用配置：

```bash
configs/vq/VQ_BL256_dino_disc_hr.yaml
```

## 需要准备

1. 一张 CUDA GPU。
2. 已安装项目训练依赖，包括 `torch`、`torchvision`、`timm`、`Pillow`、`yaml`、`wandb` 等原训练入口依赖。
3. 一个已有 GigaTok tokenizer checkpoint。stage-1 freeze 会要求从已有权重开始，不能从零训练。
4. 如果运行完整 `train` smoke，最好提前准备 DINO / DINOv2 / LPIPS 等缓存权重；当前 HR 配置沿用 `dinodisc` 和 distill 设置，训练入口可能触发这些权重加载。

## Checkpoint 路径格式

推荐放置格式：

```text
/path/to/checkpoints/0008000.pt
```

`SMOKE_MODE=train` 复用现有 `vq_train.py --vq-ckpt` 入口，checkpoint 最好是完整训练 checkpoint，至少包含：

```python
{
    "model": ...,
    "optimizer": ...,
    "discriminator": ...,
    "optimizer_disc": ...,
    "steps": ...,
}
```

如果 checkpoint 只有 `"model"` 权重，可以先跑 `SMOKE_MODE=random_forward`。这个模式不加载 optimizer/discriminator state，只验证模型 forward/backward、attention weights、HR loss 和 freeze 状态。

## 环境变量

必需：

```bash
export VQ_CKPT=/path/to/checkpoints/0008000.pt
```

常用可选项：

```bash
export CUDA_VISIBLE_DEVICES=0
export SMOKE_MODE=all
export ITERATIONS=3
export GLOBAL_BATCH_SIZE=1
export NUM_WORKERS=0
export MIXED_PRECISION=bf16
export MODEL_CONFIG=configs/vq/VQ_BL256_dino_disc_hr.yaml
export SMOKE_ROOT=$PWD/runs/stage1_smoke
export SAVE_PATH=$SMOKE_ROOT/outputs
export DATA_PATH=
export TORCH_RUN_PATH=torchrun
export MASTER_ADDR=127.0.0.1
export PORT=29531
```

缓存路径可按服务器情况设置：

```bash
export TORCH_HOME=/path/to/cache/torch
export HF_HOME=/path/to/cache/huggingface
export XDG_CACHE_HOME=/path/to/cache
```

说明：

- `ITERATIONS` 必须在 `2` 到 `5` 之间。
- `DATA_PATH` 为空时，脚本会在 `$SMOKE_ROOT/dummy_imagenet/class0/` 自动生成 8 张 dummy 图片，并按 `ImageFolder` 方式跑现有训练入口。
- 不会下载或接入 TextAtlas5M。
- `--no-wandb` 默认启用，避免 smoke test 写远端日志。

## 推荐执行顺序

第一步，先做 import-level smoke，不需要 checkpoint 和数据：

```bash
SMOKE_MODE=import bash scripts/dev/run_stage1_smoke.sh
```

预期看到：

```text
[stage1-smoke] dummy_attention_shape=(1, 2, 8, 16)
[stage1-smoke] hr_loss=...
[stage1-smoke] hr_spectrum_uniformity=...
[stage1-smoke] import smoke passed
```

第二步，用 checkpoint 做随机输入 forward/backward，不需要真实数据：

```bash
VQ_CKPT=/path/to/checkpoints/0008000.pt \
SMOKE_MODE=random_forward \
GLOBAL_BATCH_SIZE=1 \
bash scripts/dev/run_stage1_smoke.sh
```

预期看到：

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

第三步，跑现有训练入口 2 到 5 个 iteration：

```bash
VQ_CKPT=/path/to/checkpoints/0008000.pt \
SMOKE_MODE=train \
ITERATIONS=3 \
GLOBAL_BATCH_SIZE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/dev/run_stage1_smoke.sh
```

也可以一次串起来：

```bash
VQ_CKPT=/path/to/checkpoints/0008000.pt \
SMOKE_MODE=all \
ITERATIONS=3 \
GLOBAL_BATCH_SIZE=1 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/dev/run_stage1_smoke.sh
```

## 训练入口预期日志

`SMOKE_MODE=train` 会走：

```bash
torchrun --nproc_per_node=1 tokenizer/tokenizer_image/vq/vq_train.py ...
```

关键预期字段：

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
- `attention_shape`：由 `SMOKE_MODE=random_forward` 打印实际 selected decoder cross-attention weight shape；完整训练入口当前不额外打印该 shape，避免本次为 smoke test 改训练逻辑。

## 最可能失败的点

1. `VQ_CKPT` 不是完整训练 checkpoint：`SMOKE_MODE=train` 可能在加载 optimizer 或 discriminator state 时失败。先用 `SMOKE_MODE=random_forward` 判断模型权重本身是否可用。
2. 服务器没有缓存 DINO / DINOv2 / LPIPS 权重：完整训练入口可能在加载 distill teacher、`dinodisc` 或 perceptual loss 时触发下载或失败。
3. GPU 显存不足：B-L tokenizer + selected attention SVD 比 import smoke 更重。先用 `GLOBAL_BATCH_SIZE=1`，必要时只跑 `SMOKE_MODE=random_forward`。
4. `MIXED_PRECISION=bf16` 不被当前 GPU 支持：可改成 `MIXED_PRECISION=fp16` 或 `MIXED_PRECISION=none`。
5. dummy 数据目录格式错误：如果手动设置 `DATA_PATH`，必须是 `ImageFolder` 格式，例如 `$DATA_PATH/class0/xxx.png`。
6. 端口冲突：设置新的 `PORT`，例如 `PORT=29601`。
