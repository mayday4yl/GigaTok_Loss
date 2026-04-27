# 相对原版 GigaTok 的主要修改

这份文档以官方 GigaTok 作为基线，说明当前分支的主要改动范围。

## 对比基线

- 原版仓库：`https://github.com/SilentView/GigaTok.git`
- 本地基线分支：`upstream/master`
- merge-base：`baf3de042034318aa762c28db4d14231efce8b0c`
- 当前分支：`codex/text-hr-decoder`
- 当前提交可用 `git rev-parse HEAD` 查看。

查看完整差异：

```bash
git diff --stat upstream/master...HEAD
git diff --name-status upstream/master...HEAD
```

当前总量约为：

```text
44 files changed, 9432 insertions(+), 114 deletions(-)
```

## 一、核心方法改动：Text-HR v2

这部分是最应该给师姐看的代码。

### 1. Decoder cross-attention 可返回 attention weights

文件：

- `tokenizer/tokenizer_image/vq/blocks.py`

主要改动：

- 在 `TransformerDecoderLayer.forward_post(...)` 和 `forward_pre(...)` 增加 `return_cross_attn_weights`。
- 使用 PyTorch `nn.MultiheadAttention(..., need_weights=True, average_attn_weights=False)` 返回 post-softmax attention。
- 只在选中的 decoder layer 打开 attention 返回，避免每层都额外存 attention。

目的：

- 为 HR loss 提供选中 decoder cross-attention 矩阵 `[B, H, Q, K]`。

### 2. VQ tokenizer 支持 frozen T5 text feature 注入 decoder

文件：

- `tokenizer/tokenizer_image/vq/vq_vit_model.py`

主要改动：

- 新增 `configure_text_conditioning(...)`：
  - 构造 T5 hidden state 到 GigaTok decoder width 的 projection。
  - 当前默认是 `Linear + LayerNorm`。
- 新增 `project_text_memory(...)`：
  - 将选中 T5 layer hidden states `[B, T, d_t5]` 投影成 decoder memory `[B, T, d_dec]`。
- 修改 `decode(...)` / `forward(...)`：
  - 透传 `selected_decoder_layer`、`decoder_text_features`、`decoder_text_key_padding_mask`。
  - 只在选中 decoder layer 注入 text memory。
  - 返回该层 cross-attention weights 给 loss。

目的：

- 在不改 encoder、quantizer、codebook、CNN decoder 的前提下，让 stage-1 decoder 能接收文本条件。

### 3. Image-to-text attention 的 HR/SVD loss

文件：

- `tokenizer/tokenizer_image/vq/vq_loss.py`

主要改动：

- 新增 `high_rank_image_text_attention_loss(...)`。
- 输入选中 decoder 层 post-softmax attention `[B, H, Q, K]`。
- 从 key 维裁出 text token slice：

```text
image keys: [0 : image_token_len]
text keys : [image_token_len : image_token_len + T]
```

- 使用 `text_attention_mask` 去掉 padding text tokens。
- 对每个样本构造：

```text
[H * Q, T_valid]
```

- 用 `torch.linalg.svdvals(matrix.float())` 计算奇异值。
- 将 `weighted_text_hr_loss` 加入 generator 总 loss。
- 单独记录：
  - `text_hr_loss`
  - `weighted_text_hr_loss`
  - `selected_decoder_layer`
  - `selected_text_layer`
  - `text_hr_sigma_mean`
  - `text_hr_valid_text_tokens_mean`
  - `text_hr_skipped_samples`

目的：

- 约束 decoder image query 对 text token 的 cross-attention 谱结构。

### 4. Stage-1 训练主线接入 T5 和 Text-HR

文件：

- `tokenizer/tokenizer_image/vq/vq_train.py`

主要改动：

- 读取新增配置块：
  - `text_conditioning`
  - `text_hr`
- 加载 frozen T5 tokenizer / encoder。
- 每个训练 step 随机选择一个 `[t5_layer, decoder_layer]` pair。
- 将选中 T5 hidden state 传给 VQ tokenizer decoder。
- 将 decoder 返回的 attention weights 和 `text_attention_mask` 传给 `VQLoss`。
- 支持从原版 GigaTok checkpoint fine-tune：
  - 对新增的 `text_projection`、`text_type_embedding` 允许随机初始化。
  - 跳过不兼容 optimizer state。
- `text_conditioning.enabled=False` 且 `text_hr.enabled=False` 时，仍保持原生 GigaTok image-only 路径。

目的：

- 把 Text-HR 方法接入原来的 stage-1 tokenizer 训练循环。

### 5. 新增方法配置

文件：

- `configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml`
- `configs/vq/VQ_BL256_dino_disc_text_baseline_v2.yaml`
- `configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml`

用途：

- `VQ_BL256_dino_disc_text_hr_v2.yaml`
  - 我们的方法：T5 text conditioning + Text-HR loss。
- `VQ_BL256_dino_disc_text_baseline_v2.yaml`
  - 有 text conditioning，但不加 HR。
- `VQ_BL256_dino_disc_stage1_baseline.yaml`
  - 原生 GigaTok tokenizer baseline，不加 T5、不加 HR。

## 二、数据接入：TextAtlas JSONL 数据集

文件：

- `dataset/textatlas.py`
- `dataset/build.py`

主要改动：

- 新增 `textatlas_image_text` dataset。
- 从 materialized JSONL 读取：
  - `image_path`
  - `text`
  - `source_key`
  - `subset`
- 返回 `(image, text)`，供 T5 encoder 使用。
- 对空文本、raw prompt、failed extraction 做检查。
- 对 TextAtlas 单图和验证保持 resize+pad 到 `256x256`。

目的：

- 让 GigaTok stage-1 训练能使用 TextAtlas 图片和对应渲染文字。

## 三、训练与验证能力增强

文件：

- `tokenizer/tokenizer_image/vq/vq_train.py`
- `utils/distributed.py`
- `utils/model_init.py`
- `utils/resume_log.py`

主要改动：

- 支持 `--device-backend npu`，适配 Ascend NPU / HCCL。
- 支持 online validation：
  - `Val MSE`
  - `Val MAE`
  - `Val PSNR`
  - 可选 `Val SSIM`
- 支持 `--no-wandb` 在 wandb 环境损坏时继续训练。
- 支持本地 DINOv2 / torch hub cache 加载策略。
- 支持从旧 checkpoint 加载到新增 text 参数的模型。

目的：

- 让 stage-1 在服务器/NPU 环境可跑、可验证、可续训。

## 四、TextAtlas 构建、评估和单图诊断工具

### 1. TextAtlas manifest / materialization

文件：

- `scripts/stage1/textatlas_manifest_utils.py`
- `scripts/stage1/build_textatlas_fixed_manifest.py`
- `scripts/stage1/check_textatlas_fixed_manifest.py`
- `scripts/stage1/refresh_textatlas_rendered_text.py`
- `scripts/stage1/materialize_textatlas_local.py`
- `scripts/stage1/materialize_textatlas_sequential_parquet.py`

用途：

- 构建当前固定 TextAtlas subset：
  - `CleanTextSynth`
  - `StyledTextSynth`
  - `TextVisionBlend`
  - `TextScenesHQ`
  - `LongWordsSubset-A`
- 生成 train / val manifest。
- materialize 图像到本地持久化目录。
- 检查 manifest 图片和文本字段有效性。

### 2. 重建评估

文件：

- `scripts/stage1/evaluate_textatlas_reconstruction.py`

用途：

- 从 checkpoint 导出重建图。
- 计算重建指标。
- 支持 text-conditioned checkpoint。
- 后续加入 OCR 指标入口，但当前服务器没有安装 OCR 引擎时不会跑 OCR。

### 3. 单图 overfit 和逐层诊断

文件：

- `scripts/stage1/single_image_debug/make_single_image_manifests.py`
- `scripts/stage1/single_image_debug/run_single_image_overfit.sh`
- `scripts/stage1/single_image_debug/diagnose_single_image.py`
- `scripts/stage1/single_image_debug/README.md`

用途：

- 构造 dense / medium / sparse 单图 manifest。
- 单图 overfit HR 或原生 baseline。
- 对 checkpoint 做逐层诊断：
  - `target_256.png`
  - `recon_layerXX_textXX.png`
  - `diff_layerXX_textXX.png`
  - `per_layer_metrics.csv`
  - `singular_values_layerXX_textXX.npy`
- 诊断每层：
  - reconstruction MSE / PSNR
  - decoder layer feature stats
  - text attention mass
  - singular value distribution
  - effective rank
  - rank90 / rank95

目的：

- 回答“这个结构在一张图上是否能 overfit、HR loss 是否真的进入并改变 attention 谱”。

## 五、实验管理和服务器辅助脚本

文件：

- `scripts/dev/check_dino_deps.sh`
- `scripts/dev/check_disk_budget.sh`
- `scripts/dev/cleanup_stage1_artifacts.sh`
- `scripts/dev/prune_stage1_eval_outputs.py`
- `scripts/dev/rotate_stage1_checkpoints.py`
- `scripts/dev/setup_stage1_env.sh`
- `scripts/dev/source_stage1_env.sh`
- `scripts/train_vq_npu.sh`
- `scripts/stage1/plot_stage1_training_curves.py`
- `scripts/stage1/prepare_t5_encoder.py`

用途：

- 检查 DINO / T5 / checkpoint 依赖。
- 清理磁盘和 checkpoint。
- 画训练曲线。
- 下载或检查本地 T5。
- 简化 NPU 启动命令。

这些不是方法核心，但用于保证服务器实验能复现。

## 六、文档和记录

文件：

- `docs/codex_worklog.md`
- `docs/text_hr_code_map.md`
- `docs/stage1_text_hr_flow.svg`
- `docs/textatlas_notes.md`
- `docs/textatlas_stage1_bridge.md`
- `docs/hr_project_brief.md`
- `AGENTS.md`

用途：

- 记录设计、实验、失败原因、单图验证结果。
- 画 Text-HR 流程图。
- 给代码审阅提供入口说明。

## 给师姐看的最小文件列表

如果只看方法主体，建议发这几个：

```text
tokenizer/tokenizer_image/vq/blocks.py
tokenizer/tokenizer_image/vq/vq_vit_model.py
tokenizer/tokenizer_image/vq/vq_loss.py
tokenizer/tokenizer_image/vq/vq_train.py
configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml
configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml
dataset/textatlas.py
scripts/stage1/single_image_debug/diagnose_single_image.py
docs/original_gigatok_diff_map.md
```

如果只想快速定位我们方法相关改动，在代码里搜索：

```text
Text-HR v2
```

## 一句话总结

相对原版 GigaTok，当前分支主要做了三件事：

1. 在 stage-1 tokenizer decoder 中接入 frozen T5 文本特征，并只在选中 decoder layer 做 text cross-attention。
2. 从该层 post-softmax image-to-text attention 中计算 SVD/HR loss，并作为额外 generator loss 训练。
3. 增加 TextAtlas 数据、NPU 训练、online validation、单图 overfit 和逐层 SVD 诊断工具，用于验证这个方法是否真的生效。
