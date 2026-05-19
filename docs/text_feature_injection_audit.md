# Text Feature Injection Audit

## Current Answer

1. **当前 backend 是 `glyph_byt5`，不是普通 `t5`。**

   当前 OCR/Text 模板里：

   ```yaml
   text_conditioning:
     enabled: true
     encoder_backend: glyph_byt5
   ```

   训练时 `vq_train.py` 调用 `load_text_encoder_from_config()`，`encoder_backend=glyph_byt5` 会进入 `load_glyph_byt5_text_encoder()`。

2. **当前 config 里的 text layer 0，不是普通意义上的 ByT5 第 0 层。**

   config 里写的是：

   ```yaml
   text_recon_conditioning:
     layer_pairs:
       - [0, 6]
       - [0, 12]
       - [0, 18]
   ```

   但 `GlyphByT5Encoder.forward()` 返回的 `hidden_states` 不是 ByT5 每层 hidden states，而是只包含一个 mapped output：

   ```python
   encoder_outputs = self.encoder(..., output_hidden_states=False)
   mapped = self.mapper(encoder_outputs.last_hidden_state, attention_mask)
   return SimpleNamespace(last_hidden_state=mapped, hidden_states=(mapped,))
   ```

   所以这里的 text layer 0 实际上是：

   ```text
   ByT5 encoder last_hidden_state -> Glyph mapper -> mapped hidden state
   ```

   它不是 ByT5 encoder 的第 0 层，也不是 embedding 层。

3. **`GlyphByT5Encoder.forward()` 当前只暴露一个 mapped hidden state。**

   是的。它固定返回：

   ```python
   hidden_states=(mapped,)
   ```

   同时 `TextEncoderLoadResult.num_layers=1`。

4. **当前 layer 0 应该解释成什么？**

   当前 layer 0 应该解释为：

   ```text
   Glyph-ByT5 mapped text feature
   ```

   更完整地说：

   ```text
   冻结 ByT5 encoder 的最后输出，经过 Glyph-SDXL 的 ByT5 mapper 后得到的 2048 维文本特征。
   ```

   因此，师姐说“一般不是用最浅层，而是用最后一层或倒数几层”这个提醒是合理的，但当前代码里的 `0` 不代表最浅层；它代表 wrapper 暴露出来的唯一一层 mapped output。

5. **当前 text feature shape 是多少？**

   在 text encoder 输出后：

   ```text
   t5_layer_states[0].shape = [B, T, 2048]
   ```

   其中：

   - `B` 是 batch size；
   - `T` 是 tokenizer 后的 text length，当前 `max_length=1024`；
   - `2048` 来自 config 里的 `glyph_byt5.mapper_output_dim: 2048`。

   进入 GigaTok decoder 前，还会经过：

   ```python
   text_projection = Linear(2048, decoder_width) + LayerNorm(decoder_width)
   ```

   当前 decoder 是 `large`，`decoder_width=1024`，所以投影后：

   ```text
   text_memory.shape = [B, T, 1024]
   ```

6. **当前 text encoder 是否 frozen？**

   是 frozen。

   config 里：

   ```yaml
   text_conditioning:
     freeze: true
   ```

   `load_glyph_byt5_text_encoder()` 里会执行：

   ```python
   if text_cfg.get("freeze", True):
       wrapped.requires_grad_(False)
   wrapped.eval().to(device)
   ```

   所以 ByT5 encoder 和 Glyph mapper 都冻结。训练的主要新增可学习部分是 GigaTok 侧的 `text_projection`、residual cross-attention、projection 和 scale。

## Current Injection Path

当前文本注入方式是 `residual_cross_attn_visual_mask`：

```yaml
text_recon_conditioning:
  enabled: true
  mode: residual_cross_attn_visual_mask
  layers: [6, 12, 18]
  layer_pairs:
    - [0, 6]
    - [0, 12]
    - [0, 18]
```

真实流程：

1. `texts` 先经过 frozen Glyph-ByT5。
2. `get_text_layer_states()` 对 `glyph_byt5` 只返回 `(mapped,)`。
3. `vq_train.py` 根据 `layer_pairs` 把这个 mapped feature 分配给 decoder layer 6/12/18。
4. `vq_vit_model.py` 用 `text_projection` 把 `[B, T, 2048]` 投影到 `[B, T, 1024]`。
5. `blocks.py` 在 decoder 第 6/12/18 层做 residual cross-attention：

   ```python
   text_context = CrossAttention(
       query=latent_tokens,
       key=text_memory,
       value=text_memory,
   )
   latent_tokens = latent_tokens + scale * projection(text_context)
   ```

这里的 residual 是加到 decoder 中间层的 image latent tokens 上，不是加到最终图片像素上。

## Options For Text Feature Layer Changes

### 方案 A：保持当前 Glyph-ByT5 mapped output

这是当前行为。

优点：

- 不改训练主线；
- 继续复用 Glyph-SDXL 已训练好的 ByT5 mapper；
- `feature_dim=2048`，现有 `text_projection` 兼容；
- 当前 config 里的 layer 0 可以解释为“唯一暴露的 mapped output”，不是最浅层。

建议：

- 先把文档和命名讲清楚，避免误以为用了 ByT5 第 0 层。
- 如果继续跑更长训练，可以先保持这个模式，减少变量。

### 方案 B：暴露 ByT5 encoder last_hidden_state

新增 config：

```yaml
text_conditioning:
  glyph_feature_mode: mapped
  # candidates: mapped, byt5_last
```

行为：

- `mapped`：保持当前行为，返回 Glyph mapper output，`feature_dim=2048`。
- `byt5_last`：返回 ByT5 encoder 的 `last_hidden_state`，不经过 Glyph mapper，`feature_dim=encoder.config.d_model`。

需要改的代码：

1. `glyph_byt5.py`
   - `load_glyph_byt5_text_encoder()` 读取 `glyph_feature_mode`。
   - `GlyphByT5Encoder` 保存 `feature_mode`。
   - `forward()` 中如果 `feature_mode=="byt5_last"`，返回 `encoder_outputs.last_hidden_state`。
   - `TextEncoderLoadResult.feature_dim` 对应改为 `encoder.config.d_model`。

2. `vq_train.py`
   - 不需要大改逻辑，但要确保 `text_feature_dim` 来自 `TextEncoderLoadResult.feature_dim`。
   - 当前已经是这样。

3. config
   - 如果 `byt5_last`，`text_projection` 仍可用，但输入维度会从 2048 变成 ByT5 d_model。
   - checkpoint 兼容要注意：`text_projection.0.weight` shape 会变，不能直接 strict load 旧 text projection。

风险：

- 失去 Glyph mapper 的字体/颜色/多语言适配特征；
- 需要单独 smoke，确认 checkpoint loading 对新 projection shape 的处理符合预期。

### 方案 C：暴露 ByT5 hidden_states 的 last-k average

新增 config：

```yaml
text_conditioning:
  glyph_feature_mode: byt5_last_k_avg
  glyph_last_k: 4
```

行为：

- ByT5 encoder forward 时设置 `output_hidden_states=True`；
- 取最后 `k` 层 hidden states；
- 对它们求平均：

  ```python
  feature = torch.stack(hidden_states[-k:], dim=0).mean(dim=0)
  ```

需要改的代码：

1. `GlyphByT5Encoder.forward()`
   - 对 `byt5_last_k_avg` 开启 `output_hidden_states=True`；
   - 读取最后 k 层；
   - 返回 averaged feature。

2. `load_glyph_byt5_text_encoder()`
   - 读取 `glyph_last_k`；
   - 校验 `glyph_last_k > 0` 且不超过 encoder layer 数；
   - `feature_dim=encoder.config.d_model`。

3. `get_text_layer_states()`
   - 如果仍然只返回一个 averaged feature，可以保持 `hidden_states=(feature,)`。
   - 如果未来要让 config 选择多个 ByT5 层，则需要让 `hidden_states` 真正暴露多层，并同步修改 `layer_pairs` 语义。

建议：

- 方案 C 可以先设计，今晚不建议直接接主训练。
- 如果要做，先做 isolated smoke：只检查 forward shape、projection shape、2-step training 是否正常。

## Minimal Recommendation

今晚建议先采用：

1. 文档上明确当前 layer 0 是 Glyph mapped output；
2. 只准备 `glyph_feature_mode` 改造方案，不直接上大训练；
3. 下一步如要实现，先做方案 B 的最小代码改动和 2-step smoke；
4. 方案 C 等方案 B 跑通后再做。
