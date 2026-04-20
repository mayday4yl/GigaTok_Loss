# 2026-04-19 代码检查：GigaTok B-L dino disc high-rank loss 计划

## 本轮范围
- 只做只读代码检查和实现计划；未修改代码，未跑训练。
- 目标限定为 GigaTok tokenizer B-L (dino disc)，不涉及 AR model。

## 关键定位
- 目标配置：`configs/vq/VQ_BL256_dino_disc.yaml`，模型类为 `VQVitModelPlus`，`encoder_size: "base"`，`decoder_size: "large"`，`num_latent_tokens: 256`，`disc_type: "dinodisc"`。
- tokenizer 模型：`tokenizer/tokenizer_image/vq/vq_vit_model.py`
  - CNN encoder：`VQVitModelPlus.encoder`
  - transformer encoder：`VQVitModelPlus.s2to1encoder`
  - transformer decoder：`VQVitModelPlus.s1to2decoder`
  - final CNN decoder：`VQVitModelPlus.decoder`
  - quantizer/codebook：`VQVitModelPlus.quantize`
- transformer decoder 实现：`tokenizer/tokenizer_image/vq/blocks.py`
  - `ViTDecoder` 创建 `self.transformer = nn.ModuleList([... TransformerDecoderLayer ...])`
  - B-L dino disc 默认走 `ViTDecoder`，不是 `ViTDecoder_V2`
  - decoder layer 中有 self-attention 和 cross-attention；第一版建议对 `TransformerDecoderLayer.multihead_attn` 的 cross-attention 做 high-rank loss。
- 当前 attention 权重状态：
  - B-L dino disc 未启用 `use_qk_norm/use_flash_attn/use_rope`，因此 decoder layer 使用 `torch.nn.MultiheadAttention`。
  - 当前代码只取 attention 输出 `[0]`，没有保留 weights。
  - 可在选中的 decoder layer 调用 `need_weights=True, average_attn_weights=False`，得到 per-head post-softmax attention weights，形状预计为 `[B, H, 256, 256]`。
- generator total loss 组装：`tokenizer/tokenizer_image/vq/vq_loss.py` 的 `VQLoss.forward(... optimizer_idx=0 ...)`，原始 loss 在 generator 分支内组合。
- 训练入口：
  - 普通 tokenizer 训练入口：`scripts/train_vq.sh` -> `tokenizer/tokenizer_image/vq/vq_train.py`
  - 显式 checkpoint / finetune：`vq_train.py` 的 `--vq-ckpt` 和 `--finetune`
  - auto resume：`vq_train.py` 从 `experiment_dir/checkpoints` 找最新 checkpoint
  - config init checkpoint：`config["model"]["init_ckpt"]`
  - checkpoint 保存：`vq_train.py` 保存 `model/optimizer/discriminator/optimizer_disc/steps/args`
  - FSDP 入口存在，但 `vq_train_fsdp.py` 对显式 `--vq-ckpt` 当前直接 `NotImplementedError`，第一阶段不作为主线。

## 最小改动方案
- `configs/vq/VQ_BL256_dino_disc.yaml`
  - 增加可关闭的 high-rank 配置开关，例如 `trainer.high_rank_loss.enabled/weight/eps/target`。
  - 增加 decoder-only finetune 开关，例如 `trainer.decoder_only_finetune: true`。
- `tokenizer/tokenizer_image/vq/blocks.py`
  - 给 `TransformerDecoderLayer` 增加可选返回 cross-attention weights 的参数。
  - 给 `ViTDecoder.forward` 增加 `hr_layer_idx` 参数，只在选中的 1 个 layer 请求 attention weights。
  - attention weights 使用 `average_attn_weights=False` 保留 head 维度。
  - 不碰 final CNN `Decoder` / `AttnBlock` 的 attention。
- `tokenizer/tokenizer_image/vq/vq_vit_model.py`
  - 在 `decode/forward` 中透传 `hr_layer_idx`，计算 `hr_loss`。
  - high-rank loss 按定义：`torch.linalg.svdvals(attn.float())`，归一化奇异值后与均匀分布做 MSE，并在 batch/head 维度平均。
  - 增加 decoder-only finetune helper：freeze `encoder/s2to1encoder/quant_conv/quantize`，保留 `post_quant_conv/s1to2decoder/decoder` 可训练。
- `tokenizer/tokenizer_image/vq/vq_loss.py`
  - generator 分支增加可选 `hr_loss` 和权重，保持原始 loss 不变，在总 loss 上额外加 `hr_weight * hr_loss`。
  - 单独 log `hr_loss`、`hr_loss_weight`、`hr_layer_idx`。
- `tokenizer/tokenizer_image/vq/vq_train.py`
  - 读取 high-rank 和 decoder-only finetune 配置。
  - 每个 step 随机但可复现地选择 1 个 decoder layer，例如用 `train_steps` 作为 seed，保证各 rank 同步选择。
  - 将 `hr_layer_idx` 传入 tokenizer forward，将 `hr_loss` 传入 `VQLoss.forward`。
  - 在加载 checkpoint 后、DDP 包装前应用 decoder-only freeze；保留现有 `--vq-ckpt --finetune` 语义。

## 冻结计划
- 冻结 tokenizer 编码/量化侧：
  - `vq_model.encoder`
  - `vq_model.s2to1encoder`
  - `vq_model.quant_conv`
  - `vq_model.quantize`，包含 codebook/embedding
- 保持 tokenizer 解码侧可训练：
  - `vq_model.post_quant_conv`
  - `vq_model.s1to2decoder`
  - `vq_model.decoder`
  - decoder distill MLP 如存在也保持可训练
- `vq_loss.discriminator` 按原有训练逻辑保留；是否更新由 `disc_start` 和原训练流程决定。

## smoke test 方案
- 不做长训练，不接 TextAtlas5M。
- 先跑静态 import / 语法检查。
- 构造小 batch 或小模型前向，确认开启 HR 后能返回 `hr_loss`、`hr_layer_idx`，且 `hr_loss` 为有限标量。
- 对 B-L config 做一次最短 forward/backward smoke test；如显存不足，先用 tiny/small decoder 参数做功能测试，再用 B-L 仅检查模型构建和冻结参数统计。
- 检查关闭开关时返回行为与原路径兼容。

# 2026-04-19 实现：decoder cross-attention high-rank loss

## 本轮改动
- 按边界调整实现：模型侧只返回选中的 transformer decoder cross-attention weights，high-rank loss 放在 `vq_loss.py` 里计算。
- 只修改 tokenizer 主线相关文件；未修改 `autoregressive/`。
- `configs/vq/VQ_BL256_dino_disc.yaml` 增加 stage-1 配置：`hr_on`、`hr_loss_weight`、`hr_random_one_layer`、`freeze_encoder`、`freeze_quantizer`、`freeze_codebook`。
- `blocks.py` 在 `ViTDecoder`/`TransformerDecoderLayer` 上增加可选 selected layer attention 返回；默认不传 selected layer 时返回路径保持原样。
- `vq_vit_model.py` 只透传 `selected_decoder_layer` 并返回 attention weights；新增 stage-1 freeze helper。
- `vq_loss.py` 新增 `high_rank_attention_loss`，使用 post-softmax attention weights 做 float32 SVD，并记录 `hr_loss`、`selected_layer`、`hr_spectrum_uniformity`。
- `vq_train.py` 读取配置，每步按 `train_steps + global_seed` 随机选 1 个 decoder layer；加载 checkpoint 后、DDP 前应用 decoder-only freeze。

## 验证记录
- 已运行：`python3 -m py_compile tokenizer/tokenizer_image/vq/blocks.py tokenizer/tokenizer_image/vq/vq_vit_model.py tokenizer/tokenizer_image/vq/vq_loss.py tokenizer/tokenizer_image/vq/vq_train.py`
- 结果：通过。
- 已运行：`git diff --check`
- 结果：通过。
- 尝试运行 tiny decoder 前向/反向 smoke test 时，当前默认 `/usr/bin/python3` 环境缺少 `torch`，且未发现 `torchrun` 或本地 conda/venv Python；动态 smoke test 未能执行。
- 未跑全量训练。

# 2026-04-19 实现后审查与小修正

## Git 状态
- 当前处于 detached HEAD：`git status -sb` 显示 `## HEAD (no branch)`。
- 最新提交：`0613a0d clean workspace`。
- 当前改动只在 worktree，未提交。
- `configs/vq/VQ_BL256_dino_disc.yaml` 已恢复为 HEAD 原样；HR 配置改为新增 `configs/vq/VQ_BL256_dino_disc_hr.yaml`。

## Strict decoder-only finetune 检查
- `freeze_encoder=True` 会冻结 `vq_model.encoder`、`vq_model.s2to1encoder`、`vq_model.quant_conv`。
- `freeze_quantizer=True` 会冻结 `vq_model.quantize`。
- `freeze_codebook=True` 会额外冻结 `vq_model.quantize.embedding.weight`；在 `freeze_quantizer=True` 时已经被覆盖，但保留显式语义。
- `vq_model.post_quant_conv`、`vq_model.s1to2decoder`、`vq_model.decoder` 保持可训练。
- `quant_conv` 已纳入 stage-1 freeze。原因：`quant_conv` 属于 encoder output 到 codebook embedding 空间的 mapping；若它可训练，token mapping 会变化，不再是干净的 decoder-only finetune。

## HR 路径复核
- HR loss 只从 `ViTDecoder` 选中层的 `TransformerDecoderLayer.multihead_attn` cross-attention weights 计算。
- `need_weights=True, average_attn_weights=False` 只在选中层打开，使用 post-softmax per-head attention weights。
- 最后的 CNN `Decoder` 和 `AttnBlock` 没有接入 HR loss。
- `hr_on=False` 时 `selected_decoder_layer=None`，模型 forward 返回原始结构，`VQLoss` 不接收 `hr_attn_weights`，路径回到原行为。

## 验证记录
- 已运行：`git diff --check`，通过。
- 已运行：`python3 -m py_compile tokenizer/tokenizer_image/vq/blocks.py tokenizer/tokenizer_image/vq/vq_vit_model.py tokenizer/tokenizer_image/vq/vq_loss.py tokenizer/tokenizer_image/vq/vq_train.py`，通过。
- 未跑全量训练。

# 2026-04-19 训练稳定性调整

## 修改
- 将 `configs/vq/VQ_BL256_dino_disc_hr.yaml` 中 `hr_loss_weight` 从 `1.0` 调整为 `0.05`。

## 原因
- HR loss 是本阶段新引入项，初期不应过强干扰原有 tokenizer finetune loss。
- 较小权重有助于保证训练稳定，先观察 `hr_loss`、重建质量和 GAN/感知 loss 的相互影响。
- 后续可在短训稳定后逐步调大 `hr_loss_weight`。

# 2026-04-19 云服务器 stage-1 smoke test 方案

## 本轮改动
- 新增 `scripts/dev/run_stage1_smoke.sh`，用于单卡 stage-1 smoke test。
- 新增 `docs/server_stage1_setup.md`，记录服务器环境变量、checkpoint 格式、运行命令、预期日志字段和常见失败点。
- 不修改模型逻辑，不接入 TextAtlas5M，不跑全量训练。

## smoke test 分层
- `SMOKE_MODE=import`：不需要 checkpoint 或数据，只验证 HR 配置可读、`high_rank_attention_loss` 可 import 并能对 dummy attention 计算 `hr_loss` 和谱均匀性。
- `SMOKE_MODE=random_forward`：需要 tokenizer checkpoint，不需要数据；用随机图片 tensor 验证 B-L 模型 forward/backward、selected decoder cross-attention weights、`attention_shape`、`hr_loss` 和 freeze summary。
- `SMOKE_MODE=train`：复用现有 `tokenizer/tokenizer_image/vq/vq_train.py` 入口，单卡跑 2 到 5 个 iteration；未提供 `DATA_PATH` 时自动生成极小 ImageFolder dummy 数据。

## 静态验证
- 已运行：`bash -n scripts/dev/run_stage1_smoke.sh`
- 结果：通过。

# 2026-04-19 云服务器环境搭建方案补充

## 本轮改动
- 扩展 `docs/server_stage1_setup.md`，补齐从 clone 仓库、conda 环境创建、依赖安装、环境变量、checkpoint 放置、官方 baseline reconstruction 到 stage-1 HR smoke test 的完整云服务器操作流程。
- 新增 `scripts/dev/setup_stage1_env.sh`，用于创建 `gigatok-hr` conda 环境并安装 smoke test / reconstruction 所需依赖。
- 未修改模型逻辑，未接入 TextAtlas5M，未跑训练。

## 方案重点
- 先用 `configs/vq/VQ_BL256_dino_disc.yaml` + 官方 B-L dino disc checkpoint 跑 qualitative reconstruction，验证官方 tokenizer 基线和服务器环境。
- 再用 `configs/vq/VQ_BL256_dino_disc_hr.yaml` 跑 `SMOKE_MODE=import` 和 `SMOKE_MODE=random_forward`，验证 HR loss、selected decoder cross-attention、freeze summary 和 backward。
- `SMOKE_MODE=train` 仅在 checkpoint 包含 optimizer/discriminator state 时运行；官方 release checkpoint 若只有模型权重，则以 `random_forward` 作为最小 stage-1 smoke 结论。

# 2026-04-19 DINO 依赖固定策略文档化

## 本轮改动
- 更新 `docs/server_stage1_setup.md`，增加 DINO 依赖固定策略章节。
- 明确区分 DINOv2 repo commit 固定和旧 DINO discriminator checkpoint checksum 固定：前者固定 `torch.hub` 使用的 DINOv2 源码，后者固定 `disc_type: "dinodisc"` 使用的 DINOv1 权重文件。
- 增加服务器侧记录命令，生成：
  - `/root/gigatok_persist/versions/dinov2.commit`
  - `/root/gigatok_persist/versions/dinov2.status`
  - `/root/gigatok_persist/versions/dinov2.load_test.txt`
  - `/root/gigatok_persist/versions/dino_deitsmall16_pretrain.sha256`
- 记录当前已验证的 DINOv2 Python 3.9 兼容 commit：`4d83ffd4f4b39df112cef7ed7833de7ebf3a202e`。

## 边界
- 未修改模型逻辑。
- 未修改训练入口。
- 未跑全量训练。

# 2026-04-19 Stage-1 train-entry smoke 通过

## 服务器验证结果
- 已运行 `SMOKE_MODE=train ITERATIONS=3 GLOBAL_BATCH_SIZE=1 NUM_WORKERS=0 MIXED_PRECISION=bf16 bash scripts/dev/run_stage1_smoke.sh`。
- 使用固定 DINOv2 cache：`/root/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main`。
- DINOv2 teacher 输出维度：`out_inner_dim=768`。
- LPIPS 权重从 `tokenizer/tokenizer_image/cache/vgg.pth` 加载成功。
- 官方 checkpoint `/root/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt` 以 finetune 方式 resume 成功。
- Stage-1 freeze summary：`trainable_params=480,559,363`，`frozen_params=141,077,768`。
- 3 个 step 均记录到 `hr_loss`、`weighted_hr_loss`、`selected_layer`、`hr_spectrum_uniformity`，最后输出 `Done!`。

## 边界
- 使用 dummy ImageFolder 8 张图片。
- 未接入 TextAtlas5M。
- 未跑全量训练。

# 2026-04-19 DINOv2 本地 repo 优先加载与 preflight 脚本

## 本轮改动
- 新增 `scripts/dev/source_stage1_env.sh`，集中设置 stage-1 服务器环境变量，并默认固定：
  - `DINOV2_REPO_DIR=/root/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main`
  - `DINOV2_EXPECTED_COMMIT=4d83ffd4f4b39df112cef7ed7833de7ebf3a202e`
- 新增 `scripts/dev/check_dino_deps.sh`，用于训练前检查 DINOv2 commit、repo clean 状态、DINOv1 checksum，并执行轻量本地 `torch.hub.load` 测试。
- `scripts/dev/run_stage1_smoke.sh` 会 source stage-1 环境并打印 `DINOV2_REPO_DIR`、`DINOV2_EXPECTED_COMMIT`。
- `utils/model_init.py` 在 `DINOV2_REPO_DIR` 存在时优先使用本地 repo 加载 DINOv2；否则回退到原 torch.hub 行为。
- DINOv2 加载后增加 `embed_dim == 768` 的运行时安全检查。

## 边界
- 未修改 `vq_train.py`。
- 未修改 `vq_loss.py`。
- 未修改 tokenizer forward/decoder HR loss 逻辑。
- 未接入 TextAtlas5M。
- 未跑全量训练。

# 2026-04-20 TextAtlas image-only 数据桥接实现

## 实施前确认
- 已确认 `dataset/openimage.py` 中 `MixedDatasetJson` 的输入格式：`json_path` 必须指向一个 JSON 字符串数组，每个元素是完整本地图片路径。
- `MixedDatasetJson.getdata()` 会直接对数组元素执行 `Image.open(image_path_full).convert('RGB')`，因此 `train_image_paths.json` 不能是对象数组，也不能是 rich manifest JSONL。

## 本轮改动
- 新增 `scripts/stage1/build_textatlas_image_manifest.py`。
  - 只 materialize 当前 stage-1 pilot 允许的四个 subset：`CleanTextSynth`、`StyledTextSynth`、`LongWordsSubset-M`、`TextScenesHQ`。
  - 默认规模为每 subset `10000` train + `500` val，总计 `40000` train + `2000` val。
  - 输出训练实际读取的 `train_image_paths.json` / `val_image_paths.json`，格式为 JSON 字符串数组，完全兼容 `MixedDatasetJson`。
  - 同时输出 `train_manifest.jsonl` / `val_manifest.jsonl` rich manifest，用于审计和复现。
  - 图像保存和校验均使用 RGB 路径；保存后校验调用 `PIL.Image.open(...).convert("RGB")` 和 `load()`。
- 新增 `scripts/stage1/check_textatlas_manifest.py`。
  - 校验 image path list 与 rich manifest 的 `image_path` 顺序一致。
  - 校验 subset 白名单、计数、字段完整性、text_source 映射、本地图片存在且可 RGB decode/load。
  - 校验 `manifest.sha256` 中记录的文件 digest。
- 新增 `docs/textatlas_stage1_bridge.md`。
  - 记录当前 image-only 决策、训练实际读取文件、rich manifest 字段、构建/校验命令、磁盘估计和边界。

## Rich manifest 字段
- 保留：`dataset`、`subset`、`split`、`hf_split`、`hf_row_idx`、`hf_image_path`、`image_path`、`text`、`text_source`、`raw_annotation`。
- 当前训练不使用 `text`、`text_source`、`raw_annotation`。
- text 映射只做字段复制：
  - `CleanTextSynth` / `StyledTextSynth` / `LongWordsSubset-M` 使用 `annotation`。
  - `TextScenesHQ` 使用 `raw_text`，并把 `annotation` 保留到 `raw_annotation`。

## 边界
- 未修改模型逻辑。
- 未修改 tokenizer 主结构。
- 未修改 AR model。
- 未加 OCR 评测。
- 未做 text injection。
- 未下载全量 TextAtlas5M。

# 2026-04-20 TextAtlas materialize 默认格式修正

## 修改
- 将 `scripts/stage1/build_textatlas_image_manifest.py` 的默认 `--save-format` 从 `jpeg` 改为 `png`。
- 构建脚本新增 `image_paths.sha256`，单独记录 `train_image_paths.json` 和 `val_image_paths.json` 的 sha256。
- `manifest.sha256` 现在同时覆盖 rich manifest、image path list、`counts.json`、`build_config.json`、`bad_rows.jsonl` 和 `image_paths.sha256`。
- `scripts/stage1/check_textatlas_manifest.py` 新增校验：
  - `counts.json`、`manifest.sha256`、`image_paths.sha256`、`bad_rows.jsonl` 必须存在。
  - 同一 subset 内 train / val 的 `hf_row_idx` 不得有交集。
  - `bad_rows.jsonl` 若非空，每行必须是合法 JSON 对象，并包含 `dataset`、`subset`、`hf_split`、`hf_row_idx`、`error`。
  - `counts.json` 中 train/val 总数和各 subset 计数必须与 manifest 一致。
- 更新 `docs/textatlas_stage1_bridge.md`，将推荐构建命令改为 `--save-format png`，并说明 PNG 无损保存是为了避免二次 JPEG 压缩污染 reconstruction 指标。

## 边界
- 训练仍然只读取 `train_image_paths.json` / `val_image_paths.json` 字符串数组。
- rich manifest 中的 text 字段仍仅用于审计和复现。
- 未修改模型逻辑、tokenizer 主结构或 AR model。
- 未加 OCR 评测。
