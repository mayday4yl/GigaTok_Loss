# TextAtlas Stage-1 Fixed-Count Bridge

## 目的

当前 stage-1 只验证 HR loss 是否改善 tokenizer 的文字重建效果。训练输入仍是 image-only，不做 text injection，不做 OCR loss，不改模型逻辑。

训练阶段只读取本地 `train_image_paths.json` / `val_image_paths.json`，不走 Hugging Face streaming。

## 当前数据方案

默认使用 5 个 subset：

- `CleanTextSynth`
- `StyledTextSynth`
- `TextVisionBlend`
- `TextScenesHQ`
- `LongWordsSubset-A`

默认 split：

- train：`TextScenesHQ` 抽 `40,000` 张，其余 4 个 subset 各抽 `50,000` 张，总计 `240,000`。
- val：每个 subset `2,000` 张，总计 `10,000`。
- hold-out：每个 subset `500` 张，总计 `2,500`，默认只冻结 source manifest，不提前落盘。

`TextAtlasEval` 只留作最终独立 benchmark，不参与训练内 val 或调参。

## Rich Manifest

随机抽样路径中，`build_textatlas_fixed_manifest.py` 输出 source-only manifest：

- `exact_count.json`
- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `holdout_manifest.jsonl`
- `manifest.sha256`
- `build_config.json`

每行保留 `dataset`、`subset`、`split`、`hf_split`、`hf_row_idx`、`source_key`、`selection_seed`、`selection_strategy`，用于复现和回溯。

文本字段只在 materialize 后审计保存，不参与训练。当前字段映射：

| subset | text_source |
| --- | --- |
| `CleanTextSynth` | `annotation` |
| `StyledTextSynth` | `annotation` |
| `TextVisionBlend` | `annotation` |
| `TextScenesHQ` | `raw_text` |
| `LongWordsSubset-A` | `annotation` |

## 构建命令

### 首选：顺序 parquet 落盘

如果 ModelArts 上 `datasets` streaming 很慢，但 `curl` 直接下载 Hugging Face converted parquet 较快，优先使用顺序 parquet 落盘。该路径不做随机抽样；每个 subset 从前往后连续取样，直接写出 source manifest、materialized manifest 和训练需要的 image paths：

```bash
unset http_proxy https_proxy ftp_proxy all_proxy no_proxy
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY NO_PROXY
export HTTP_PROXY=http://proxy-notebook.modelarts.com:8083
export HTTPS_PROXY=http://proxy-notebook.modelarts.com:8083
export HF_ENDPOINT=https://huggingface.co

rm -rf "$OUTPUT_DIR/textatlas_stage1_fixed_240k/images"
rm -rf "$PERSIST_ROOT/cache/textatlas_parquet_test"
rm -rf "$PERSIST_ROOT/cache/textatlas_parquet_seq"

python scripts/stage1/materialize_textatlas_sequential_parquet.py \
  --output-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest" \
  --image-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/images" \
  --parquet-cache-root "$PERSIST_ROOT/cache/textatlas_parquet_seq" \
  --splits train val \
  --preprocess resize-pad \
  --image-size 256 \
  --overwrite-manifest
```

该脚本默认：

- `TextScenesHQ=40,000 train`
- 其余 4 个 subset 各 `50,000 train`
- 每 subset `2,000 val`
- 每 subset `500 hold-out`，写 source manifest，但默认不落盘图片
- 从 Hugging Face `refs/convert/parquet` 下载连续 parquet shard

下载和落盘完成后再校验：

```bash
python scripts/stage1/check_textatlas_fixed_manifest.py \
  --manifest-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest" \
  --materialized-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest"
```

如果需要节省磁盘，可在确认可断点重跑策略后增加 `--delete-parquet-after-materialize`；默认保留 parquet cache，便于失败后续跑和排查。

### 备选：随机 manifest + streaming

```bash
python scripts/stage1/build_textatlas_fixed_manifest.py \
  --output-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest" \
  --exact-counts-json "$PERSIST_ROOT/versions/textatlas5m_5subset_exact_counts.json" \
  --overwrite
```

校验 source-only manifest：

```bash
python scripts/stage1/check_textatlas_fixed_manifest.py \
  --manifest-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest"
```

落盘 fixed train/val：

```bash
HF_ENDPOINT=https://hf-mirror.com \
python scripts/stage1/materialize_textatlas_local.py \
  --manifest-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest" \
  --image-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/images" \
  --output-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest" \
  --splits train val \
  --preprocess resize-pad \
  --image-size 256
```

校验 materialized 文件：

```bash
python scripts/stage1/check_textatlas_fixed_manifest.py \
  --manifest-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest" \
  --materialized-root "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest"
```

## 训练入口

训练继续复用 `dataset.openimage.MixedDatasetJson`：

```bash
--dataset imagenet_openimage \
--json-path "$OUTPUT_DIR/textatlas_stage1_fixed_240k/manifest/train_image_paths.json"
```

`train_image_paths.json` 是 JSON 字符串数组，里面每个元素都是本地图片绝对路径。

## 边界

- 不修改模型逻辑。
- 不修改 HR loss 公式。
- 不修改 tokenizer 主结构。
- 不修改 AR model。
- 不保留旧的大规模分块训练数据脚本。
