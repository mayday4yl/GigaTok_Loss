# Text Feature Cache Plan

## Goal

当前 text encoder 是 frozen。训练时每个 step 重新 tokenize 和 encode text 会浪费时间。

目标是提前把 text feature 编码好，训练时直接读取 cache，减少训练开销，并让 text feature layer / feature mode 实验更可控。

## Cache Inputs

cache 生成脚本需要输入：

- manifest jsonl，例如 `cleantextsynth_ocr_readable_50_train.jsonl`；
- `text_conditioning` config；
- backend：`glyph_byt5` 或 `t5`；
- feature mode：
  - `mapped`
  - `byt5_last`
  - `byt5_last_k_avg`
- `max_length`；
- tokenizer path / model path；
- Glyph special token 配置；
- 是否包含 attention mask；
- dtype，例如 `fp16` / `bf16` / `fp32`。

## Cache Outputs

建议优先 cache **text hidden states**，而不是只 cache tokenized input。

每个样本保存：

```text
sample_id / source_key / image_path
text
input_ids
attention_mask
feature
feature_mode
feature_shape
config_hash
```

当前 mapped 模式下：

```text
feature.shape = [T, 2048]
attention_mask.shape = [T]
```

如果缓存 projected 后的 decoder memory，则 shape 是：

```text
projected_feature.shape = [T, 1024]
```

但不建议第一版 cache projected feature，因为 `text_projection` 是 trainable。

第一版更稳的是 cache frozen text encoder output：

```text
Glyph/ByT5 feature -> trainable text_projection 仍在训练时计算
```

## Cache Key

cache key 必须能检测配置变化。

建议 key 包含：

- `image_path`
- `source_key`
- `text` 的 hash；
- backend；
- encoder path；
- tokenizer path；
- max_length；
- feature mode；
- mapper checkpoint path；
- ByT5 checkpoint path；
- tokenizer vocab size；
- special token config；
- relevant `text_conditioning` config hash。

示例：

```text
cache_key = sha1(json.dumps({
  "image_path": image_path,
  "source_key": source_key,
  "text_sha1": sha1(text),
  "backend": "glyph_byt5",
  "feature_mode": "mapped",
  "max_length": 1024,
  "encoder_name": ".../Glyph-SDXL-v2",
  "byt5_base_name": ".../google_byt5-small",
  "byt5_model_path": "checkpoints/byt5_model.pt",
  "byt5_mapper_path": "checkpoints/byt5_mapper.pt",
  "tokenizer_vocab_size": len(tokenizer),
}, sort_keys=True))
```

## Storage Format

推荐两种：

### Option 1: Sharded `.pt`

```text
text_feature_cache/
  metadata.json
  shard_00000.pt
  shard_00001.pt
```

每个 shard 存一个 list/dict：

```python
{
    "keys": [...],
    "source_keys": [...],
    "image_paths": [...],
    "texts": [...],
    "input_ids": tensor[N, T],
    "attention_mask": tensor[N, T],
    "features": tensor[N, T, C],
}
```

优点：PyTorch 读取简单。

缺点：单 shard 太大时随机读不方便。

### Option 2: `.npz` per shard

适合只存 numpy arrays。

缺点：metadata 和 variable length 支持不如 `.pt` 方便。

第一版建议 `.pt` sharded。

## Training Read Path

两种接入方式：

### Option A: dataset 返回 text_feature

dataset 根据 manifest row 的 cache key 读取 feature，返回：

```python
{
    "image": image,
    "text": text,
    "text_feature": feature,
    "text_attention_mask": attention_mask,
}
```

训练 loop 如果检测到 `text_feature`，就跳过 text_encoder forward。

优点：清晰。

缺点：dataset 需要持有 cache index，multi-worker 随机读要处理好。

### Option B: training loop 根据 sample id 查 cache

dataset 返回 `cache_key` 或 `source_key`，训练 loop 里统一查 cache。

优点：对 dataset 改动小。

缺点：训练 loop 会更复杂。

建议第一版用 Option A，但先只做 skeleton，不接主线。

## Multi-layer Feature Cache

如果未来要 cache 多层 text features：

```python
features = {
    "mapped": tensor[T, 2048],
    "byt5_last": tensor[T, d_model],
    "byt5_last_k_avg_k4": tensor[T, d_model],
}
```

或者按 layer 存：

```python
features_by_layer = {
    "layer_20": tensor[T, d_model],
    "layer_21": tensor[T, d_model],
    "layer_22": tensor[T, d_model],
    "layer_23": tensor[T, d_model],
}
```

训练 config 的 `layer_pairs` 必须和 cache 内暴露的 layer keys 对齐。

## Invalidation Rules

以下任一变化都必须让 cache 失效：

- `text` 内容变化；
- tokenizer vocab 或 special tokens 变化；
- `max_length` 变化；
- `encoder_backend` 变化；
- `glyph_feature_mode` 变化；
- `glyph_last_k` 变化；
- Glyph mapper checkpoint 变化；
- ByT5 checkpoint 变化；
- dtype 策略变化；
- 是否使用 mapper 变化。

## Skeleton Command

未来可以新增脚本：

```bash
python3 scripts/stage1/cache_text_features.py \
  --manifest-jsonl /path/to/train.jsonl \
  --model-config configs/vq/xxx.yaml \
  --output-dir /path/to/text_feature_cache \
  --feature-mode mapped \
  --batch-size 16 \
  --dtype bf16
```

今晚不接入训练主线。
