# Stage-1 服务器迁移计划

本文只覆盖当前 stage-1 四子集方案的迁移，不改变模型逻辑、训练目标、tokenizer 结构或数据集范围。

当前允许的数据子集固定为：

- `CleanTextSynth`
- `StyledTextSynth`
- `LongWordsSubset-M`
- `TextScenesHQ`

当前不加入 `TextVisionBlend`、`PPT2Structured`、`Paper2Text`。

## 迁移目标

把旧服务器上已经构建好的 stage-1 数据、manifest、checkpoint、版本固定信息和可复现状态迁移到新服务器。迁移完成后，新服务器应能在相同四子集方案下通过 manifest 检查和 DINO/checkpoint preflight，再进入后续短训练。

## 必须转移的内容

这些内容直接决定实验可复现性，建议从旧服务器复制到新服务器：

- 当前 Git 分支和提交：
  - `feat/hrloss-bl-textatlas`
  - 迁移时记录 `git rev-parse HEAD`
- stage-1 pilot manifest 目录：
  - `train_image_paths.json`
  - `val_image_paths.json`
  - `train_manifest.jsonl`
  - `val_manifest.jsonl`
  - `counts.json`
  - `build_config.json`
  - `bad_rows.jsonl`
  - `manifest.sha256`
  - `image_paths.sha256`
- materialized PNG 图片目录：
  - `build_config.json` 中的 `image_root`
  - 或 rich manifest 中 `image_path` 指向的图片根目录
- tokenizer 起点 checkpoint：
  - 默认 `VQ_CKPT=$PERSIST_ROOT/checkpoints/VQ_BL256_dino_disc.pt`
  - 必须记录 sha256
- DINO 固定信息：
  - `DINOV2_REPO_DIR`
  - `DINOV2_EXPECTED_COMMIT`
  - DINOv2 local repo 当前 commit/status
  - DINOv2 / DINOv1 权重 sha256
- 旧服务器环境状态导出：
  - Python/conda/pip 版本
  - CUDA/GPU 信息
  - stage-1 路径变量
  - disk usage

## 可以在新服务器重下的内容

这些内容可以重下，但重下后必须再次验证版本或 checksum：

- Git 仓库代码。
- Conda/pip 依赖。
- Hugging Face 原始 TextAtlas parquet。
- Hugging Face cache。
- DINOv2 repo 和权重文件。
- DINOv1 `dino_deitsmall16_pretrain.pth`。

如果平台网络不稳定，可以继续使用运行时环境变量：

```bash
HF_ENDPOINT=https://hf-mirror.com
```

如果新平台 IPv6 不通，可临时使用 `/tmp/python_force_ipv4/sitecustomize.py` 方案；该文件不进仓库。

## 可以重建但必须保留索引/manifest 的内容

materialized PNG 图片理论上可以在新服务器重新从 Hugging Face 构建，但必须保留旧 manifest 作为审计基准：

- `subset`
- `hf_split`
- `hf_row_idx`
- `hf_image_path`
- `split`
- `image_path`
- `text`
- `text_source`
- `raw_annotation`

更稳妥的做法是直接迁移旧服务器生成的 PNG 图片和 manifest。这样 baseline/HR 会使用完全相同的本地图像文件。

如果确实选择重建图片，必须：

1. 使用相同四个 subset。
2. 使用相同 `train_per_subset` / `val_per_subset` / `start_offset`。
3. 使用相同 `save_format=png`。
4. 构建后对照旧 rich manifest 的 `subset + split + hf_row_idx`。
5. 重新运行 `check_textatlas_manifest.py`。

## 旧服务器绝对路径审计

### `scripts/dev/source_stage1_env.sh`

该脚本已改为默认从脚本位置自动推导 `PROJECT_ROOT`，并把 `PERSIST_ROOT` 默认设为 `$HOME/gigatok_persist`。迁移到新服务器时仍建议显式指定：

```bash
export PROJECT_ROOT=/new/path/GigaTok_Loss
export PERSIST_ROOT=/new/persist/gigatok_persist
source scripts/dev/source_stage1_env.sh
```

### `train_image_paths.json` / `val_image_paths.json`

这两个文件是训练实际读取的文件。它们是 JSON 字符串数组，数组元素是图片绝对路径。若新服务器路径不同，必须处理。

可选方案：

1. 在新服务器保持相同目录结构，例如仍使用 `/root/gigatok_persist/...`。
2. 使用 `scripts/dev/rewrite_manifest_paths.py` 重写路径并重算 sha256。
3. 在新服务器重新构建 manifest。

推荐优先级：

```text
保持相同目录结构 > 路径重写 > 重新构建
```

### rich manifest

`train_manifest.jsonl` / `val_manifest.jsonl` 中的 `image_path` 同样是绝对路径。若重写 image path list，rich manifest 必须同步重写，否则 check 会失败。

### `docs/server_stage1_setup.md`

该文档中包含旧服务器示例路径，例如 `/root/GigaTok_hr/GigaTok_Loss` 和 `/root/gigatok_persist`。迁移时把它们视为示例，不要直接照抄；新服务器应使用实际 `PROJECT_ROOT` / `PERSIST_ROOT`。

### 其他脚本

部分旧 composite 脚本仍有项目历史路径或默认 `PROJECT_ROOT` 示例。当前 stage-1 数据桥接和迁移检查只依赖：

- `scripts/dev/source_stage1_env.sh`
- `scripts/dev/check_dino_deps.sh`
- `scripts/stage1/check_textatlas_manifest.py`
- `scripts/stage1/build_textatlas_image_manifest.py`
- `scripts/dev/export_env_state.sh`
- `scripts/dev/check_migration_ready.sh`
- `scripts/dev/rewrite_manifest_paths.py`

训练前若使用其他 composite 脚本，需要单独审计其 `PROJECT_ROOT`、`DATA_ROOT`、`VQ_CKPT`、`save-path`。

## 旧服务器迁移导出顺序

先确认 pilot manifest 已完成并通过 check：

```bash
cd /root/GigaTok_hr/GigaTok_Loss
conda activate gigatok
export PROJECT_ROOT=/root/GigaTok_hr/GigaTok_Loss
export PERSIST_ROOT=/root/gigatok_persist
source scripts/dev/source_stage1_env.sh

python scripts/stage1/check_textatlas_manifest.py \
  --manifest-root "$OUTPUT_DIR/textatlas_stage1_pilot/manifest" \
  --expected-train-per-subset 10000 \
  --expected-val-per-subset 500
```

导出环境状态：

```bash
bash scripts/dev/export_env_state.sh \
  --manifest-root "$OUTPUT_DIR/textatlas_stage1_pilot/manifest"
```

建议复制的目录：

```text
$OUTPUT_DIR/textatlas_stage1_pilot/
$CKPT_DIR/
$PERSIST_ROOT/versions/
$PERSIST_ROOT/migration_state/
```

如果新服务器可以重下 DINO 和 HF cache，可以不复制：

```text
$PERSIST_ROOT/cache/huggingface/
$PERSIST_ROOT/cache/torch/
```

如果希望完全减少新服务器下载，可以连同 cache 一起复制。

推荐使用 `rsync`，保留文件时间和目录结构：

```bash
rsync -aH --info=progress2 \
  "$OUTPUT_DIR/textatlas_stage1_pilot/" \
  user@new-server:/new/persist/gigatok_persist/outputs/textatlas_stage1_pilot/

rsync -aH --info=progress2 \
  "$CKPT_DIR/" \
  user@new-server:/new/persist/gigatok_persist/checkpoints/

rsync -aH --info=progress2 \
  "$PERSIST_ROOT/migration_state/" \
  user@new-server:/new/persist/gigatok_persist/migration_state/
```

## 新服务器恢复顺序

克隆代码并切到主实验分支：

```bash
git clone git@github.com:mayday4yl/GigaTok_Loss.git /new/path/GigaTok_Loss
cd /new/path/GigaTok_Loss
git fetch origin
git switch feat/hrloss-bl-textatlas
```

准备环境。若使用仓库脚本：

```bash
PERSIST_ROOT=/new/persist/gigatok_persist \
CONDA_ENV_NAME=gigatok \
bash scripts/dev/setup_stage1_env.sh
```

以后每个 shell 进入：

```bash
cd /new/path/GigaTok_Loss
conda activate gigatok
export PROJECT_ROOT=/new/path/GigaTok_Loss
export PERSIST_ROOT=/new/persist/gigatok_persist
source scripts/dev/source_stage1_env.sh
```

如果新服务器路径和旧服务器不同，重写 manifest 中的绝对图片路径：

```bash
python scripts/dev/rewrite_manifest_paths.py \
  --manifest-root "$OUTPUT_DIR/textatlas_stage1_pilot/manifest" \
  --old-prefix /root/gigatok_persist/outputs/textatlas_stage1_pilot/images \
  --new-prefix "$OUTPUT_DIR/textatlas_stage1_pilot/images" \
  --in-place
```

如果新服务器保持旧路径结构，则不需要重写。

## 迁移完成后的验证顺序

先做不训练的迁移检查：

```bash
bash scripts/dev/check_migration_ready.sh \
  --manifest-root "$OUTPUT_DIR/textatlas_stage1_pilot/manifest" \
  --expected-train-per-subset 10000 \
  --expected-val-per-subset 500 \
  --max-image-checks 0 \
  --forbid-prefix /root/gigatok_persist
```

如果新服务器沿用 `/root/gigatok_persist`，不要传 `--forbid-prefix`。

再做 DINO preflight：

```bash
bash scripts/dev/check_migration_ready.sh \
  --manifest-root "$OUTPUT_DIR/textatlas_stage1_pilot/manifest" \
  --expected-train-per-subset 10000 \
  --expected-val-per-subset 500 \
  --max-image-checks 100 \
  --run-dino-preflight
```

最后确认 Git 干净：

```bash
git status
```

只有这些检查通过后，才进入后续 200-step gate。迁移阶段不跑训练。
