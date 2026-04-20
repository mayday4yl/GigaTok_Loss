# TextAtlas5M Stage-1 Image-Only Bridge

## 目的

当前 stage-1 pilot 只训练 GigaTok tokenizer reconstruction 路径，输入是图像，不做 text injection，不做 OCR 评测，也不基于文本计算额外 loss。TextAtlas5M 的文本字段只保留在 rich manifest 中，用于审计、复现和后续扩展。

## 训练实际读取的文件

训练入口复用现有 `dataset.openimage.MixedDatasetJson`：

```bash
--dataset imagenet_openimage \
--json-path /path/to/manifest/train_image_paths.json
```

`train_image_paths.json` 必须是 JSON 字符串数组：

```json
[
  "/abs/path/to/image_000001.png",
  "/abs/path/to/image_000002.png"
]
```

`MixedDatasetJson` 会把每个元素当成完整路径，直接执行 `Image.open(path).convert("RGB")`。因此训练不会读取 rich manifest，也不会读取 text 字段。

## Rich Manifest

构建脚本同时输出：

- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `train_image_paths.json`
- `val_image_paths.json`
- `counts.json`
- `manifest.sha256`
- `image_paths.sha256`
- `bad_rows.jsonl`

rich manifest 每行保留：

```json
{
  "dataset": "CSU-JPG/TextAtlas5M",
  "subset": "CleanTextSynth",
  "split": "train",
  "hf_split": "train",
  "hf_row_idx": 0,
  "hf_image_path": "000004f933f14f65bfcd6ee1d54d4e69.png",
  "image_path": "/abs/output/images/CleanTextSynth/train/0000000000.png",
  "text": "...",
  "text_source": "annotation",
  "raw_annotation": "..."
}
```

文本字段映射只做字段复制：

| subset | text_source | text | raw_annotation |
| --- | --- | --- | --- |
| `CleanTextSynth` | `annotation` | `annotation` | `annotation` |
| `StyledTextSynth` | `annotation` | `annotation` | `annotation` |
| `LongWordsSubset-M` | `annotation` | `annotation` | `annotation` |
| `TextScenesHQ` | `raw_text` | `raw_text` | `annotation` |

这些字段当前不参与训练。以后如果要做 text conditioning、OCR 指标或文本专项分析，可以直接用 `subset`、`hf_row_idx`、`hf_image_path`、`text`、`text_source`、`raw_annotation` 回溯样本和文本来源。

## 默认规模

默认只构建 pilot 所需样本，不下载全量 TextAtlas5M：

- 每个 subset `10,000` train
- 每个 subset `500` val
- 四个 subset 总计 `40,000` train + `2,000` val

允许 subset 固定为：

- `CleanTextSynth`
- `StyledTextSynth`
- `LongWordsSubset-M`
- `TextScenesHQ`

## 构建命令

所有输出路径都必须显式传入，避免写死系统盘路径：

```bash
python3 scripts/stage1/build_textatlas_image_manifest.py \
  --dataset-name CSU-JPG/TextAtlas5M \
  --output-root /path/to/runs/stage1_pilot_YYYYMMDD/manifest \
  --image-root /path/to/data/textatlas_stage1_pilot/images \
  --train-per-subset 10000 \
  --val-per-subset 500 \
  --seed 0 \
  --save-format png
```

构建脚本使用 Hugging Face `datasets` streaming 读取目标 subset，按顺序 materialize 当前 pilot 需要的图像。`seed` 会写入 `build_config.json` 作为复现元信息；当前为了保持 `hf_row_idx` 精确，默认选择策略是 sequential streaming。

## 校验命令

```bash
python3 scripts/stage1/check_textatlas_manifest.py \
  --manifest-root /path/to/runs/stage1_pilot_YYYYMMDD/manifest \
  --expected-train-per-subset 10000 \
  --expected-val-per-subset 500
```

校验内容：

- `train_image_paths.json` / `val_image_paths.json` 是 `MixedDatasetJson` 兼容的字符串数组。
- image path list 与 rich manifest 中的 `image_path` 顺序完全一致。
- subset 只包含四个允许值。
- 每个 subset 的 train/val 数量符合预期。
- 同一 subset 内 train / val 的 `hf_row_idx` 没有交集。
- 本地图像存在，并使用 `PIL.Image.open(...).convert("RGB")` 后调用 `load()` 验证可读。
- `manifest.sha256` 中记录的文件校验值匹配。
- `bad_rows.jsonl` 存在，若非空则每行是合法 JSONL 错误记录。
- `counts.json`、`manifest.sha256`、`image_paths.sha256` 均存在并通过一致性检查。

## 磁盘估计

默认 42,000 张图像，优先保存为 RGB PNG，避免 materialize 阶段引入额外 JPEG 有损压缩，从而减少对 rFID / LPIPS / PSNR / SSIM 的污染。PNG 落盘会明显大于 JPEG，粗略预留 `50-120 GB` 更稳妥，实际取决于图像内容和压缩率。`counts.json` 会记录实际图片字节数和估算 GB。

## 边界

- 不修改模型逻辑。
- 不修改 tokenizer 主结构。
- 不修改 AR model。
- 不做 OCR 评测。
- 不做 text injection。
- 训练不读取 rich manifest。
- 不下载全量 TextAtlas5M。
