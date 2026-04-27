# Text-HR v2 代码导览

这份文件用于快速说明当前分支相对原生 GigaTok 的主要改动位置。代码里统一使用 `Text-HR v2` 注释标记，打开文件后可以直接搜索这个关键词。

## 核心入口

- `configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml`
  - `text_conditioning`: 开启 frozen T5，取 T5 hidden states 作为 decoder text memory。
  - `text_hr`: 开启 image-to-text attention HR loss，配置 layer pairs、`image_token_len`、padding mask 和 SVD 方式。
- `configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml`
  - 原生 GigaTok baseline。
  - 没有 `text_conditioning`，没有 `text_hr`。

## 模型侧

- `tokenizer/tokenizer_image/vq/vq_vit_model.py`
  - `configure_text_conditioning(...)`: 构造 T5 feature 到 decoder width 的 projection。
  - `project_text_memory(...)`: 把选中 T5 layer hidden states 投影成 decoder memory tokens。
  - `decode(...)`: 只在 `selected_decoder_layer` 注入 text memory，并返回该层 cross-attention weights。
  - `forward(...)`: 透传 `selected_decoder_layer`、`decoder_text_features` 和 `decoder_text_key_padding_mask`。

- `tokenizer/tokenizer_image/vq/blocks.py`
  - `TransformerDecoderLayer.forward_post(...)`
  - `TransformerDecoderLayer.forward_pre(...)`
  - 增加 `return_cross_attn_weights`，从 PyTorch `nn.MultiheadAttention` 取 post-softmax attention weights。

## Loss 侧

- `tokenizer/tokenizer_image/vq/vq_loss.py`
  - `high_rank_image_text_attention_loss(...)`
    - 输入选中 decoder 层 attention `[B, H, Q, K]`。
    - 从 key 维裁出 text token slice。
    - 用 `text_attention_mask` 去掉 padding token。
    - reshape 成 `[H * Q, T_valid]` 后用 float32 做 SVD。
    - 记录 `text_hr_loss`、`text_hr_sigma_mean`、`valid_text_tokens` 和 skipped samples。
  - `VQLoss.forward(...)`
    - 将 `weighted_text_hr_loss` 加到 generator 总 loss。
    - 保持 reconstruction / perceptual / feature / codebook loss 原有记录。

## 训练侧

- `tokenizer/tokenizer_image/vq/vq_train.py`
  - 读取 `text_conditioning` 和 `text_hr` config。
  - 加载 frozen T5 tokenizer / encoder。
  - 每个 step 随机选择一个 `[t5_layer, decoder_layer]` pair。
  - 把选中 T5 hidden state 传给 VQVitModelPlus。
  - 把返回的 decoder cross-attention weights 和 text mask 传给 `VQLoss`。
  - 对旧 GigaTok checkpoint 兼容新增的 `text_projection` 和 `text_type_embedding` 参数。

## 单图诊断

- `scripts/stage1/single_image_debug/diagnose_single_image.py`
  - 对训练前、训练中、训练后的 checkpoint 做逐层诊断。
  - 输出 `target_256.png`、`recon_layerXX_textXX.png`、`diff_layerXX_textXX.png`、`per_layer_metrics.csv` 和奇异值 `.npy`。

- `scripts/stage1/single_image_debug/run_single_image_overfit.sh`
  - `MODE=hr`: text + HR。
  - `MODE=baseline`: 原生 GigaTok tokenizer baseline。
