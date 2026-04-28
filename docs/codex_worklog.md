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

# 2026-04-20 TextAtlas 早期数据方案归档

## 结论
- 早期 TextAtlas pilot、服务器迁移审计和路径重写方案已废弃。
- 相关可执行脚本和旧服务器文档已在后续清理中删除。
- 当前 TextAtlas 数据入口以固定数量 5-subset 方案为准。

# 2026-04-21 Stage-1 存储控制与清理策略工具

## 本轮目标
- 只实现磁盘与大文件控制工具。
- 不修改模型逻辑、HR loss 公式、tokenizer 主结构或 AR model。
- 当前数据路线以固定数量 5-subset 本地 materialize 为准。

## 大文件风险排序
1. materialized train images：最容易快速增长。
2. full resume checkpoint：包含 model、optimizer、discriminator、optimizer_disc、steps、args，必须 rotation。
3. eval 重建输出：禁止默认保留全量 reconstruction PNG / GT PNG / `.npz`。
4. torch / HF / DINO / LPIPS cache：DINO pinned 信息和必需权重不能自动清理；HF 临时下载和 pip cache 可按需清理。
5. fixed val images：HR / baseline 共用，长期保留但纳入预算。
6. model-only checkpoint：用于 eval / 汇报 / baseline 对齐，体积小于 full resume。
7. logs / metrics / manifests / run_state：体积小，默认长期保留。

## 本轮新增
- `scripts/dev/check_disk_budget.sh`
  - 只读打印 `df -h`、关键目录 `du -sh` 和 normal / warning / hard / emergency 水位线状态。
  - 默认阈值：warning `75%` 或 `<350GB`，hard `85%` 或 `<200GB`，emergency `90%` 或 `<100GB`。
- `scripts/dev/cleanup_stage1_artifacts.sh`
  - 默认 dry-run；只有显式传 `--execute` 才删除。
  - 清理 eval bulky 输出，并调用 checkpoint rotation。
  - 保护 manifests、fixed train/val、latest/best/final checkpoint。
- `scripts/dev/rotate_stage1_checkpoints.py`
  - 管理 `checkpoints/full_resume` 和 `checkpoints/model_only`。
  - full resume 默认保留最近 2 份，model-only 默认保留最近 3 份。
  - 永远保护 `latest.pt`、`best.pt`、`final.pt` 和 `.protected` 标记文件。
- `scripts/dev/prune_stage1_eval_outputs.py`
  - eval 后只保留 metrics / config / grid 类输出。
  - 默认最多保留 32 张非 grid sample image，并将 eval 目录控制在 1GB 以内。
  - 删除超额图片、`.npz`、`.npy` 和 `eval/tmp` 内容。

## 不可删除对象
- `train_manifest.jsonl`
- `val_manifest.jsonl`
- `manifest.sha256`
- `exact_count.json`
- `run_state.json`
- 当前 latest 可恢复 full checkpoint
- fixed train images
- `val_image_paths.json`
- fixed val images 和 fixed val manifest
- 起点 checkpoint `VQ_BL256_dino_disc.pt`
- DINO pinned repo/version 信息和必需权重
- 当前 run config、实际命令记录和本 worklog

## 建议目录策略
- `checkpoints/full_resume/`：最近 2 份 + `latest.pt`。
- `checkpoints/model_only/`：最近 3 份 + `latest.pt` / `best.pt` / `final.pt`。
- `eval/latest/`：只保留 metrics 和 sample grid。
- `eval/history/`：只保留 metrics。
- `eval/tmp/`：eval 后清空。
- eval 输出硬上限：默认 `MAX_SAMPLES=32`，`MAX_DIR_SIZE=1GB`。

## 使用边界
- 这些脚本不启动训练。
- 这些脚本不修改模型、loss、tokenizer 或数据集范围。
- 清理脚本默认只 dry-run，训练 runner 集成时必须先看 dry-run 输出，再决定是否传 `--execute`。

# 2026-04-21 Stage-1 最小 NPU 训练入口兼容

## 本轮目标
- 只做训练入口设备后端兼容，支持新服务器 Ascend NPU baseline smoke。
- 保留 CUDA 原路径。
- 不修改模型逻辑、HR loss 公式、tokenizer 主结构或 AR model。
- 不修改当前四子集 image-only 数据方案。

## 本轮改动
- `utils/distributed.py`
  - `init_distributed_mode(args)` 根据 `args.device_backend` 选择设备模块。
  - `cuda` 使用原 `nccl` backend。
  - `npu` import `torch_npu`，使用 `torch.npu` 和 `hccl` backend。
- `tokenizer/tokenizer_image/vq/vq_train.py`
  - 新增 `--device-backend {cuda,npu}`，默认 `cuda`。
  - CUDA 路径保留原设备检查、DDP 和 AMP 行为。
  - NPU 路径使用 `torch_npu` / `torch.npu`、`torch.device("npu:<local_rank>")`。
  - `torch.cuda.synchronize()` 改为按后端同步。
  - autocast 增加后端分支；baseline smoke 推荐先使用 `--mixed-precision none`，避免 NPU AMP 兼容风险。
- `scripts/train_vq_npu.sh`
  - 新增 NPU 专用启动脚本。
  - 默认 `GPUS=1`，用于单卡 NPU baseline smoke。
  - 调用 `vq_train.py --device-backend npu`。

## Smoke 边界
- baseline smoke 第一版推荐单卡 NPU、`--mixed-precision none`、`--no-wandb`。
- 当前不跑长训练。
- 当前不启用多卡 HCCL 性能优化。

# 2026-04-21 NPU baseline smoke: DINO discriminator device 修复

## 现象
- 新服务器单卡 NPU baseline smoke 已进入 HCCL 初始化、gate dataloader、DINOv2 teacher load 和 VQ model 构造。
- 随后在 `VQLoss(...disc_type="dinodisc")` 构造 DINO discriminator 时失败：
  - `DINODiscriminator` 默认 `device="cuda"`。
  - NPU 环境的 PyTorch 未编译 CUDA，因此 `.to("cuda")` 报 `Torch not compiled with CUDA enabled`。

## 本轮修复
- `tokenizer/tokenizer_image/vq/vq_loss.py`
  - 给 `VQLoss` 新增 `discriminator_device` 参数，默认仍为 `"cuda"`，保持 CUDA 原路径。
  - `disc_type="dinodisc"` 时把该设备传给 `DINODiscriminator`。
- `tokenizer/tokenizer_image/vq/vq_train.py`
  - 构造 `VQLoss` 时传入当前训练设备 `device`。

## 边界
- 不修改 DINO discriminator 结构。
- 不修改 loss 公式。
- 不修改 HR loss、tokenizer 主结构或 AR model。
- 目标只是让 NPU baseline smoke 不再强制走 CUDA。

# 2026-04-21 大规模数据方案归档

## 结论
- 大规模 source-only manifest、分块落盘和独立 fixed-val 方案已废弃。
- 相关可执行脚本已删除。
- 当前只维护固定数量 5-subset manifest + 本地 train/val materialize 路线。

# 2026-04-21 Stage-1 固定数量文字重建验证数据方案

## 最新目标
- 当前不再执行大规模原始比例训练，也不按 TextAtlas 原始比例抽样。
- 目标改为用固定数量的本地 materialized 数据，验证：
  - baseline = `original_gigatok_loss`
  - HR = `original_gigatok_loss + hr_loss_weight * hr_loss`
- baseline 和 HR 必须共用同一份 `train / val / optional hold-out`、同一起点 checkpoint、batch、lr、steps 和 seed。
- 训练时只读本地 `*_image_paths.json`，不走 Hugging Face streaming。

## 数据设计
- 当前 5 个 subset：
  - `CleanTextSynth`
  - `StyledTextSynth`
  - `TextVisionBlend`
  - `TextScenesHQ`
  - `LongWordsSubset-A`
- `TextScenesHQ`：抽 `40000` 张 train。
- 其余 4 个 subset：每个抽 `50000` 张 train。
- 总 train 为 `240000` 张；balanced val 为 `10000` 张；optional hold-out 为 source-only `2500` 张。
- 空间估算只作为可选步骤，不再作为构建 manifest 的前置条件。
- 当前采用 `LongWordsSubset-A`，因为更重视 256 下可读性和预处理稳定性。

## Val / Test
- 自建 balanced val：每个 subset `2000` 张，总计 `10000`。
- 额外冻结 source-only hold-out：每个 subset `500` 张，总计 `2500`，默认不提前 materialize。
- `TextAtlasEval` 只作为最终独立 benchmark/test，不参与调参或 checkpoint 选择；它不覆盖 `LongWordsSubset-A`，不能替代当前 balanced val。

## 本轮实现
- `scripts/stage1/textatlas_manifest_utils.py`
  - 新增当前 5-subset 默认常量。
  - 新增 `resize-pad` 预处理 helper，支持按长边缩放并 pad 到 `256x256`。
- `scripts/stage1/build_textatlas_fixed_manifest.py`
  - 根据 exact counts 构建固定 source-only manifest。
  - 默认 `TextScenesHQ=40000 train`，其他 4 个 subset 各 `50000 train`。
  - 输出 `exact_count.json`、`train_manifest.jsonl`、`val_manifest.jsonl`、`holdout_manifest.jsonl`、`manifest.sha256`。
- `scripts/stage1/materialize_textatlas_local.py`
  - 从 source-only manifest 一次性 materialize fixed local train/val。
  - 输出 `train_image_paths.json`、`val_image_paths.json`、materialized manifests、`materialized.sha256`。
- `scripts/stage1/check_textatlas_fixed_manifest.py`
  - 校验 5-subset source-only manifest、train/val/hold-out disjoint、balanced val、固定 train count 和 materialized 图片可读性。
- 删除旧的大规模训练和独立 val 脚本，避免后续误用旧方案。

## 推荐执行顺序
1. `build_textatlas_fixed_manifest.py` 构建固定数量 manifest。
2. `check_textatlas_fixed_manifest.py` 校验 source-only manifest。
3. `materialize_textatlas_local.py --splits train val --preprocess resize-pad` 落盘本地 fixed train/val。
4. `check_textatlas_fixed_manifest.py --materialized-root ...` 校验本地图像路径和可读性。
5. 再跑 baseline smoke、HR smoke，最后进入 baseline/HR 正式对比训练。

# 2026-04-21 Stage-1 固定数量数据方案简化与脚本清理

## 本轮修正
- 进一步简化固定数量数据方案：不再通过单独空间采样反推 `N_main`。
- `TextScenesHQ` 默认 train count 固定为 `40000`。
- `CleanTextSynth`、`StyledTextSynth`、`TextVisionBlend`、`LongWordsSubset-A` 默认每个 train count 固定为 `50000`。
- balanced val 仍为每 subset `2000`，optional hold-out 仍为每 subset `500`。

## 代码影响
- 保留当前方案需要的 stage-1 数据脚本：
  - `scripts/stage1/build_textatlas_fixed_manifest.py`
  - `scripts/stage1/materialize_textatlas_local.py`
  - `scripts/stage1/check_textatlas_fixed_manifest.py`
  - `scripts/stage1/textatlas_manifest_utils.py`
- 删除旧方案残留脚本，只保留当前固定数量方案所需入口。
- `scripts/stage1/build_textatlas_fixed_manifest.py`
  - 新增默认参数：
    - `--textsceneshq-train-count 40000`
    - `--same-train-count 50000`
  - 构建 manifest 时直接按固定数量抽样，仍保证 val/hold-out/train 无交集。
- `scripts/stage1/check_textatlas_fixed_manifest.py`
  - 新增默认检查：
    - `--expected-textsceneshq-train-count 40000`
    - `--expected-same-train-count 50000`

## 边界
- 不修改模型逻辑、HR loss 公式、tokenizer 主结构或 AR model。
- 不保留单独空间采样脚本；如需估算空间，直接用 materialize smoke 或 `du` 观测。
- 不引入 OCR/CER/F1 作为阻塞指标。
- 不再维护“小训练集调参 + 大训练集正式跑”的两阶段数据集。

# 2026-04-21 Stage-1 旧方案残留清理复查

## 清理结论
- 当前 stage-1 可执行数据路径只保留固定数量方案：
  - `scripts/stage1/build_textatlas_fixed_manifest.py`
  - `scripts/stage1/materialize_textatlas_local.py`
  - `scripts/stage1/check_textatlas_fixed_manifest.py`
  - `scripts/stage1/textatlas_manifest_utils.py`
- 继续删除旧服务器迁移和旧 smoke 残留，避免误调用已删除脚本。
- `AGENTS.md` 已同步到当前 5-subset 固定数量方案，避免后续 agent 继续按旧 4-subset 约束工作。

## 当前边界
- 之前的大规模分块训练方案不再有可执行脚本。
- 之前的容量反推方案不再有可执行脚本。
- 现行方案以后续固定数量章节为准。

# 2026-04-21 Stage-1 persist root 默认路径修正

## 修正
- `scripts/dev/source_stage1_env.sh` 的默认 `PERSIST_ROOT` 从 `${HOME}/gigatok_persist` 改为仓库同级目录下的 `gigatok_persist`。
- 在 ModelArts 路径 `/home/ma-user/work/GigaTok_hr/GigaTok_Loss` 下，默认会解析到 `/home/ma-user/work/GigaTok_hr/gigatok_persist`。

## 原因
- 当前服务器约定所有数据和 cache 放在 `/home/ma-user/work/GigaTok_hr/gigatok_persist`。
- 避免后台 materialize 误写到 `/home/ma-user/gigatok_persist`。

# 2026-04-21 Stage-1 顺序 parquet 落盘路径

## 背景
- ModelArts 上 `datasets` streaming + 随机行抽样会频繁触发远端 parquet seek，实际下载很慢且容易超时。
- 直接通过 ModelArts 默认代理 `proxy-notebook.modelarts.com:8083` 用 `curl` 下载 Hugging Face converted parquet shard 可达到十几 MB/s，明显更稳定。

## 本轮修正
- 新增 `scripts/stage1/materialize_textatlas_sequential_parquet.py`。
- 新路径不做随机抽样；每个 subset 从前往后连续取样：
  - `val`: 前 `2000` 行。
  - `hold-out`: 接下来的 `500` 行，仅写 source manifest。
  - `train`: 后续 `TextScenesHQ=40000` 行，其余 subset 各 `50000` 行。
- 脚本直接下载 `refs/convert/parquet` 下的连续 parquet shard，解码图片并按 `resize-pad` 落盘到训练需要的 `train_image_paths.json` / `val_image_paths.json`。
- 默认保留 parquet cache，便于失败后断点续跑；如确认空间紧张，可显式传 `--delete-parquet-after-materialize`。

## 验证
- 本地通过 `py_compile` 检查新增脚本和现有 stage-1 数据脚本。
- 本地用伪 parquet 对象做了小型顺序 materialize smoke，验证连续行号、split 跳过 hold-out、图片路径命名和 `resize-pad` 保存逻辑。

## 边界
- 不修改模型逻辑、HR loss 公式、tokenizer 主结构或 AR model。
- 顺序抽样会牺牲随机覆盖面，但 baseline/HR 仍使用同一份本地数据，满足当前 pilot 的严格可比性。

# 2026-04-22 Stage-1 310k 训练配置准备

## 数据状态
- 当前固定数据集为 `textatlas_stage1_fixed_310k`。
- train 共 `300000` 张：`TextScenesHQ=40000`，其余 4 个 subset 各 `65000`。
- val 共 `10000` 张：每个 subset `2000`。
- `check_textatlas_fixed_manifest.py --materialized-root ...` 校验通过，`errors=[]`。

## 训练配置
- 新增 `configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml`。
- baseline 配置与 HR 配置保持同样的 decoder-only finetune 边界：
  - `freeze_encoder=True`
  - `freeze_quantizer=True`
  - `freeze_codebook=True`
- baseline 只关闭 HR：
  - `hr_on=False`
  - `hr_loss_weight=0.0`
- HR 继续使用 `configs/vq/VQ_BL256_dino_disc_hr.yaml`，当前 `hr_loss_weight=0.05`。

## 下一步
- 先用 310k train JSON 跑短 smoke，确认 dataloader、checkpoint、DINO distill 和 NPU 后端都能闭环。
- smoke 通过后跑 baseline 与 HR 两个严格可比实验，除 HR 开关/权重外，其余训练预算和数据保持一致。

# 2026-04-22 Stage-1 TextAtlas 重建评估脚本

## 背景
- 4 NPU baseline 与 4 NPU HR 训练已跑完 `10000` steps；两者 train loss 基本重合。
- 下一步不能只看 train loss，需要在固定 val 图像上比较重建指标和可视化结果。
- 仓库已有 `reconstruction_ddp.py` 主要是 CUDA/NCCL 和内置 dataset 路线，不适合直接用于当前 NPU + 本地 `val_image_paths.json`。

## 本轮新增
- 新增 `scripts/stage1/evaluate_textatlas_reconstruction.py`。
- 功能：
  - 读取本地 `val_image_paths.json` 和可选 `val_manifest.jsonl`。
  - 加载一个或多个 `NAME:CONFIG:CKPT` tokenizer run。
  - 使用 VQ encode/decode 重建图像。
  - 输出 `metrics.json`，包含 `mse`、`mae`、`psnr`、`ssim` 的 overall 和 subset 统计。
  - 输出 `comparison_grid.png`，按 `GT / baseline / HR` 并排可视化。
  - 默认不保存全量 reconstruction PNG，避免评估产物占用过多空间。

## 验证
- 本地已运行：`python3 -m py_compile scripts/stage1/evaluate_textatlas_reconstruction.py`，通过。

## 边界
- 不修改训练代码、模型结构、HR loss 或数据抽样。
- 第一轮评估使用当前 materialized val 作为固定测试/验证集；若后续要独立 hold-out 图片，需要先 materialize hold-out。

# 2026-04-22 Stage-1 训练曲线可视化脚本

## 背景
- 当前 baseline 与 HR 训练已完成，日志里已有 `Train Loss` 和 generator 组件 loss。
- 需要把日志曲线画出来，便于判断训练是否进入平台期，以及汇报 baseline/HR 的训练动态。

## 本轮新增
- 新增 `scripts/stage1/plot_stage1_training_curves.py`。
- 功能：
  - 读取一个或多个 `NAME:/path/to/train.log`。
  - 解析 `Train Loss`、`steps/sec`、学习率，以及 generator 行里的 `rec_loss`、`direct_rec_loss`、`perceptual_loss`、`feature_rec_loss`、`hr_loss` 等字段。
  - 输出 `train_loss.png`、`reconstruction_components.png`、`hr_metrics.png`。
  - 同时输出 `training_curves.csv` 和 `training_summary.json`，便于后续写报告或画其他图。

## 边界
- 只读训练日志，不修改训练代码、checkpoint、数据集或评估逻辑。

# 2026-04-23 Stage-1 在线 validation 训练准备

## 背景
- 第一轮 `HR=0.05` 与 baseline 训练曲线和重建效果基本重合，说明 HR 接入稳定但权重偏弱。
- `HR=50` 的 4 NPU `gbs48` 短测可运行，`weighted_hr_loss` 达到可见量级。
- 下一轮计划重跑 baseline 与 HR 强权重版本，并在训练过程中在线验证、保存 best checkpoint。

## 本轮新增
- `tokenizer/tokenizer_image/vq/vq_train.py` 增加在线验证参数：
  - `--val-json-path`
  - `--val-every`
  - `--eval-batch-size`
  - `--val-num-workers`
  - `--val-max-images`
  - `--val-compute-ssim`
  - `--save-best`
  - `--save-last`
  - `--best-metric`
  - `--best-mode`
- 在线验证每隔固定 step 在 `torch.no_grad()` 下计算 `val_mse`、`val_mae`、`val_psnr`，可选计算 `val_ssim`，不参与反传，不调用判别器。
- 在线 validation 的 MSE/MAE/PSNR/SSIM 均按 `[0,1]` 图像范围统计，和离线重建评估脚本保持一致。
- rank 0 写出：
  - `metrics/train_metrics.csv`
  - `metrics/val_metrics.csv`
  - `metrics/train_val_curves.png`
- 当验证指标更优时保存 `checkpoints/best.pt`。
- 新增 `configs/vq/VQ_BL256_dino_disc_hr_w50.yaml`，只将 `hr_loss_weight` 改为 `50.0`，其余 stage-1 decoder-only finetune 边界保持不变。

## 边界
- 不修改 tokenizer 架构、AR model、数据抽样或判别器启动策略。
- 在线 validation 当前只记录重建数值指标；最终视觉对比仍使用独立重建评估脚本生成 grid。

# 2026-04-23 Stage-1 best checkpoint 轮转修复

## 背景
- 在线 validation 保存 `checkpoints/best.pt` 后，原有 checkpoint 轮转逻辑会把所有 `*.pt` 都当成数字步数 checkpoint。
- 在保存 `0001000.pt` 后调用 `manage_ckpt_num()` 时，`best.pt` 被解析成整数 `best`，触发 `ValueError` 并中断训练。

## 修复
- `utils/resume_log.py` 新增数字前缀 checkpoint 过滤 helper。
- `manage_ckpt_num()`、`manage_fsdp_ckpt_num()` 和 WSD checkpoint 查找只处理 `0001000.pt` 这类数字前缀文件。
- `tokenizer/tokenizer_image/vq/vq_train.py` 自动恢复也只从数字前缀 checkpoint 中选择 latest，避免 `best.pt`、`last.pt` 参与 step 恢复排序。

## 边界
- `best.pt`、`last.pt` 仍正常保留。
- 不改变 checkpoint 保存内容、验证指标或训练损失。

# 2026-04-25 Text-conditioned decoder HR v2 代码管理

## 分支
- 从 `feat/hrloss-bl-textatlas` 当前本地 HEAD 切出新分支 `codex/text-hr-decoder`。
- 当前分支保留旧分支领先远端的 1 个本地提交作为 v2 基线。
- 旧分支不继续混入 v2 文本注入改动。

## 方案归档
- v1 = image-only decoder attention HR。
  - 保留旧 decoder cross-attention HR loss。
  - 保留旧 `hr_on` / `hr_loss_weight` / `hr_random_one_layer` 配置语义。
  - 保留旧 baseline / HR / HR=50 配置。
- v2 = T5 text-conditioned decoder + image-to-text HR。
  - 新增独立配置 `configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml`。
  - v2 使用 `text_conditioning.enabled` 和 `text_hr.enabled`，不复用旧 `hr_on` 作为主开关。

## v2 默认边界
- 冻结 GigaTok encoder、`s2to1encoder`、`quant_conv`、quantizer/codebook、`post_quant_conv` 和 T5。
- 只训练 `s1to2decoder`、final CNN `decoder`、新增 text projection 和可学习 `text_type_embedding`。
- 不加 `text_position_embedding`；T5 hidden state 提供文本序列位置信息，GigaTok decoder image queries 提供 16x16 图像位置信息。
- 默认层策略为中层显式 pair：decoder/T5 block `8..15`。
- v2 HR loss 目标矩阵为选中层 post-softmax cross-attention 中裁出的 image-to-text 子矩阵。
- padding text token 必须在 SVD 前移除。
- 默认 SVD 策略为 per-sample all-head stack，并除以 `sqrt(num_heads)` 保持奇异值尺度稳定。

## 合并安全规则
- 后续实现只给现有 forward/loss/train 路径增加可选参数；文本参数为 `None` 或开关关闭时保持旧 image-only 行为。
- v1 HR loss 函数保留；v2 image-to-text HR loss 使用独立函数。
- merge 前需要分别跑旧 config smoke 和 v2 config smoke，确认 v1 未破坏、v2 文本注入与 HR loss 可用。

# 2026-04-25 Text-conditioned decoder HR v2 实现

## 本轮新增
- 新增 `dataset/textatlas.py` 和 `textatlas_image_text` dataset 入口，读取 `image_path/text` JSONL manifest，并支持相对图片路径按 manifest 所在目录解析。
- 新增 v2 配置：
  - `configs/vq/VQ_BL256_dino_disc_text_baseline_v2.yaml`：T5 text-conditioned decoder baseline，不启用 text HR。
  - `configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml`：同样文本注入路径，启用 image-to-text HR loss。
- `VQVitModelPlus` 新增可选 text conditioning：
  - `configure_text_conditioning()` 创建 T5 hidden state 到 decoder width 的 projection。
  - `text_type_embedding` 为可学习参数。
  - `post_quant_conv` 可通过 `freeze_post_quant_conv` 冻结。
- `ViTDecoder.forward()` 新增可选 `text_memory` / `text_key_padding_mask`：
  - 只在选中的 decoder layer 把 memory 扩展为 `[image_tokens; text_tokens]`。
  - padding text token 通过 `memory_key_padding_mask` 在 attention softmax 前 mask。
- `vq_train.py` 新增 v2 训练路径：
  - 冻结 T5 encoder，`output_hidden_states=True`。
  - T5 embedding 输出 `hidden_states[0]` 不参与选层；配置中的 `t5_layer=k` 对应 `hidden_states[k + 1]`。
  - 默认按显式中层 pair `[8,8]..[15,15]` 每 step 随机选一对。
  - text-conditioned 数据预处理改为等比 resize + pad 到 256，不做随机裁剪和水平翻转。

## v2 HR loss
- 新函数：`high_rank_image_text_attention_loss()`。
- 输入为选中 decoder 层的 post-softmax cross-attention weights `[B, H, Q, K]`。
- 裁剪方式：
  - image keys 为前 `image_token_len=256` 列。
  - text keys 为后续 `T` 列。
  - HR 矩阵使用 `A[:, :, :, 256:256+T]` 的有效 text token 子矩阵。
- SVD 策略：
  - 不在 batch 维度先求均值。
  - 每个样本单独处理，按 `text_attention_mask` 移除 padding 列。
  - 将 heads 垂直堆叠为 `[H*Q, valid_T]`，再除以 `sqrt(H)`。
  - `torch.linalg.svdvals(matrix.float())` 用 float32 计算奇异值。
- loss：
  - `mean(abs(sigma - tau))`，默认 `tau=1.0`。
  - `valid_T < 2` 的样本跳过，避免 padding 或极短文本制造无意义奇异值。

## 兼容性
- 旧 v1 `hr_on/hr_loss_weight` 和 `high_rank_attention_loss()` 保留。
- `text_conditioning.enabled=False` 且 `text_hr.enabled=False` 时，旧 image-only forward/loss/train 路径保持不变。
- v1 HR 与 v2 text HR 不允许同时开启，避免同一 attention 权重被两套语义混用。

## 验证
- 已运行 `python3 -m py_compile` 检查新增/修改的 dataset、decoder、model、loss、train 文件，语法通过。
- 已用 Ruby YAML parser 检查旧 v1 config 与新增 v2 baseline/HR config，解析通过。
- 已运行 `git diff --check`，无 whitespace error。
- 当前本地 Python 环境缺少 `torch`，无法在此环境执行张量级 runtime smoke；需要在训练环境补跑旧 config smoke 与 v2 config smoke。

# 2026-04-25 TextAtlas rendered text 与 T5 本地加载修补

## 背景
- v2 的 `text` 必须是图片里实际出现的文字内容，不是 caption，也不是完整 generation prompt。
- TextAtlas 不同 subset 的 `annotation` 格式不完全一致，不能简单把整段 `annotation` 透传给 T5。
- 当前 workspace 没有已下载的 `*_materialized_manifest.jsonl`，也没有预拉取 T5 权重；训练环境需要重新 materialize 并准备 T5 cache。

## 本轮修补
- `scripts/stage1/textatlas_manifest_utils.py`
  - 新增 annotation 文本抽取逻辑。
  - `CleanTextSynth`、`StyledTextSynth`、`TextVisionBlend`、`LongWordsSubset-A` 默认从 `annotation` 中解析 rendered text。
  - `TextScenesHQ` 优先使用 `raw_text`；如果缺失则 fallback 到 annotation 解析。
  - materialized manifest 新增 `text_extraction_status`，保留 `raw_annotation`，最终训练字段仍为 `text`。
  - 如果抽取后 `text` 为空，materialize 直接报错，避免错误数据进入训练。
- `dataset/textatlas.py`
  - 训练读取 manifest 时检查 `text` 非空。
  - 拒绝看起来仍是 raw prompt 的 `text`。
  - annotation-based row 若 `text == raw_annotation` 且不是 `plain_annotation`，直接报错。
- `scripts/stage1/check_textatlas_fixed_manifest.py`
  - materialized manifest 校验新增 `text_extraction_status`。
  - 检查空 text、抽取失败、raw prompt 残留、错误的 annotation 透传。
- `tokenizer/tokenizer_image/vq/vq_train.py`
  - T5 `from_pretrained()` 支持 `cache_dir` 和 `local_files_only`。
  - `encoder_name` 可以是 Hugging Face model id，也可以是本地目录。
- 新增 `scripts/stage1/prepare_t5_encoder.py`
  - 用于在训练环境提前下载或验证 T5 tokenizer + encoder cache。

## 注意
- v2 训练必须使用 `train_materialized_manifest.jsonl`，不能使用旧的 `train_image_paths.json`。
- 如果已有旧 materialized manifest，里面没有 `text_extraction_status`，且 `text` 可能是完整 `annotation`；需要用新脚本重新 materialize。
- 离线训练时建议在 config 里设置：
  - `text_conditioning.encoder_name: /path/to/local/t5`
  - `text_conditioning.local_files_only: True`

# 2026-04-25 TextAtlas v2 text manifest refresh

## 服务器数据确认
- 已通过 SSH 检查 ModelArts：
  - `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/images/train` 已存在，约 `20G`。
  - `manifest/train_materialized_manifest.jsonl` 和 `manifest/val_materialized_manifest.jsonl` 已存在。
  - `manifest_holdout_eval/holdout_materialized_manifest.jsonl` 已存在。
- 旧 manifest 统计：
  - train `300000` 行；val `10000` 行。
  - annotation-based 四个子集的旧 `text` 全部等于 `raw_annotation`。
  - 旧 manifest 没有 `text_extraction_status`。
  - parquet cache 已不存在，但旧 manifest 保留 `raw_annotation`，因此不需要重下图片。
- 服务器 Hugging Face cache 当前没有 T5 权重。

## 本轮新增
- 新增 `scripts/stage1/refresh_textatlas_rendered_text.py`。
- 功能：
  - 输入旧 `*_materialized_manifest.jsonl`。
  - 保留原始行顺序、`image_path`、`hf_row_idx`、`source_key` 和 `materialize_order`。
  - 只刷新 `text`、`text_source`、`raw_annotation`、`text_extraction_status`。
  - 输出新 manifest，默认文件名增加 `_v2text` 后缀。
  - 如果 `materialize_order` 与行号不一致、抽取为空、仍像 raw prompt、或抽取失败，则不落最终文件并报错。

## 子集抽取规则
- `CleanTextSynth`：提取 `displaying the text:` 后面的内容。
- `StyledTextSynth`：提取所有 `the text : '...'` / `the text : "..."` 文本块，按出现顺序用换行连接。
- `TextVisionBlend`：只解析 `For text elements` 里的编号文本项，按编号顺序用换行连接。
- `TextScenesHQ`：保留原 `raw_text`。
- `LongWordsSubset-A`：按 `with text reading ...`、`we note ... visible`、`along with visible ...`、`and ... text`、`and ... clearly shown` 等模式提取可见文字列表。

## 云端刷新命令
```bash
python scripts/stage1/refresh_textatlas_rendered_text.py \
  --manifest-root /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest \
  --splits train val

python scripts/stage1/refresh_textatlas_rendered_text.py \
  --manifest-root /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest_holdout_eval \
  --splits holdout
```
- v2 train 应使用：
  - `train_materialized_manifest_v2text.jsonl`
  - `val_materialized_manifest_v2text.jsonl`
  - 可选 `holdout_materialized_manifest_v2text.jsonl`

## 验证
- 本地样例测试覆盖 `CleanTextSynth`、`StyledTextSynth`、`TextVisionBlend`、`TextScenesHQ`、`LongWordsSubset-A`，抽取结果符合预期。
- 已运行 `python3 -m py_compile` 检查相关脚本，语法通过。
- 已运行 `git diff --check`，无 whitespace error。

# 2026-04-25 TextAtlas rendered-text parser hardening

## 修补内容
- `LongWordsSubset-A` 的 annotation 模板比初版覆盖面更散，已改成 marker-based parser，覆盖 `plus`、`adding`、`including`、`seeing`、`discern`、`we find`、`with ...`、`, text ...` 等可见文字列表模板。
- `StyledTextSynth` 增补：
  - `with text ''...''`
  - narrative 形式的 `text reads/states/continues`
  - 新闻标题类的多段双引号
  - 极少数 blank poster / empty text 标记为 `styled_no_explicit_text`
- `TextVisionBlend` 中没有独立 `For text elements` 条目的少量样本标记为 `textvision_no_explicit_text`，不伪造文本。
- Dataset 和 manifest checker 只允许上述 no-explicit-text 状态为空文本；其它空文本仍然报错。

## 服务器验证和产物
- 已在 ModelArts 对旧 manifest 运行 dry-run：
  - train `300000` 行，`error_count=0`
  - val `10000` 行，`error_count=0`
  - holdout `2500` 行，`error_count=0`
- 已正式生成：
  - `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/train_materialized_manifest_v2text.jsonl`
  - `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/val_materialized_manifest_v2text.jsonl`
  - `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest_holdout_eval/holdout_materialized_manifest_v2text.jsonl`

# 2026-04-25 T5 encoder 下载网络排障

## 现象
- `prepare_t5_encoder.py` 使用 `HF_ENDPOINT=https://huggingface.co` 加 ModelArts proxy 时，`requests/urllib3` 在 CONNECT proxy 阶段 timeout 或返回 `503 Service Unavailable`。
- `HF_ENDPOINT=https://hf-mirror.com` 且完全 unset proxy 时，服务器直连 `hf-mirror.com` timeout。
- Ascend toolkit owner warning 与本问题无关。

## 排查结论
- ModelArts 登录环境会默认注入大小写 proxy 变量，proxy 主机解析到 `192.168.0.33:8083`。
- `curl -k` 通过 proxy 可以访问 Hugging Face API，并确认 `google/t5-v1_1-xl` 文件列表中 PyTorch 权重为 `pytorch_model.bin`，大小约 `11.4GB`。
- Python `requests` 通过同一 proxy 访问 `https://hf-mirror.com/google/t5-v1_1-xl/...` 可以返回 `307` 和 repo commit。
- 当前推荐组合是保留 ModelArts proxy，并设置 `HF_ENDPOINT=https://hf-mirror.com`；不要在 mirror 路径下 unset proxy 后直连。

## 下一步命令
```bash
export HTTP_PROXY=http://proxy-notebook.modelarts.com:8083
export HTTPS_PROXY=http://proxy-notebook.modelarts.com:8083
export http_proxy=http://proxy-notebook.modelarts.com:8083
export https_proxy=http://proxy-notebook.modelarts.com:8083
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ETAG_TIMEOUT=120
export HF_HUB_DOWNLOAD_TIMEOUT=600

python scripts/stage1/prepare_t5_encoder.py \
  --config-yaml configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml \
  --cache-dir /home/ma-user/work/GigaTok_hr/gigatok_persist/cache/huggingface

python scripts/stage1/prepare_t5_encoder.py \
  --config-yaml configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml \
  --cache-dir /home/ma-user/work/GigaTok_hr/gigatok_persist/cache/huggingface \
  --local-files-only
```

## 追加结论
- 2026-04-25 11:59 继续使用 `HF_ENDPOINT=https://hf-mirror.com` 加 ModelArts proxy 时，`prepare_t5_encoder.py` 仍在 CONNECT 阶段返回 `503 Service Unavailable`。
- 后续不再优先反复尝试 `transformers.from_pretrained()` 联网路径。
- 推荐 fallback：用 `curl -k -L -C -` 逐文件断点下载 `google/t5-v1_1-xl` 到本地目录，再用 `--encoder-name /path/to/local/t5 --local-files-only` 验证，训练 config 也指向同一个本地目录。

# 2026-04-25 Stage-1 text HR 矢量流程图

## 本轮产物
- 新增 `docs/stage1_text_hr_flow.svg`。
- 图中记录 v2 text-conditioned HR 主流程：
  - T5 tokenizer / encoder 冻结。
  - T5 `hidden_states` 为 embedding output + 24 layer outputs，实际选层使用 `hidden_states[layer + 1]`。
  - GigaTok large transformer decoder 为 24 层；每个 step 从 `[8,8]..[15,15]` 中随机选 1 个 T5/decoder 中层 layer pair。
  - T5 原始特征 `[B,T,2048]` 经 `Linear + LayerNorm` 投影到 `[B,T,1024]`，并加 learnable text type embedding。
  - decoder 选中层 cross-attention 权重为 `[B,16,256,256+T]`，HR loss 只取 text slice `[B,16,256,T]`。
  - 每个 valid sample 去除 padding text token 后堆叠 heads，构造 `[16*256,T_valid] / sqrt(16)`，做 float32 SVD。
  - 当前 v2 loss 公式为 `mean_i |sigma_i - tau|`，默认 `tau=1.0`，再对 valid samples 取均值。
- 修订：去掉重复的 matched-layer 模块，将 layer pair sampling 合并为单一说明框，避免遮挡并明确 `l_t` 选 T5 hidden state、`l_d` 选 decoder cross-attention layer。
- 修订：将 image branch 中的 `post_quant_conv` 单独标为 frozen post-quant projection，明确它位于 quantizer 和 transformer decoder 之间，负责 `codebook dim 8 -> token dim 256`，不属于 trainable decoder 主体。

# 2026-04-25 v2 text-conditioned checkpoint load 修补

## 现象
- ModelArts baseline smoke 已成功加载本地 T5 encoder，并进入 GigaTok 模型初始化。
- 使用旧 `VQ_BL256_dino_disc.pt` 初始化 v2 text-conditioned 模型时报错：
  - `text_type_embedding`
  - `text_projection.*`
- 原因是这些参数是 v2 新增模块，旧 stage-1 checkpoint 中不存在。

## 修补
- `custom_load()` 新增 `ignore_missing_keys` 可选参数，默认行为不变。
- 仅在 `args.finetune and text_conditioning_on` 时，允许 v2 新增的 text projection / text type embedding 参数从随机初始化开始训练。
- 其它 missing keys 仍然报错，避免掩盖 checkpoint 结构不匹配。

## 验证
- 已运行：
  - `PYTHONPYCACHEPREFIX=/tmp/gigatok_pycache python3 -m py_compile utils/model_init.py tokenizer/tokenizer_image/vq/vq_train.py`

# 2026-04-25 ModelArts v2 smoke 结果

## 已跑通
- baseline smoke：
  - config: `configs/vq/_smoke_text_baseline_local.yaml`
  - output: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_smoke/baseline_2step_nodistill`
  - 2 steps 完成，最终日志 `Done!`
- HR smoke：
  - config: `configs/vq/_smoke_text_hr_local.yaml`
  - output: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_smoke/hr_2step_nodistill`
  - 2 steps 完成，最终日志 `Done!`

## 关键验证点
- 本地 T5 encoder 正常加载：
  - `/home/ma-user/work/GigaTok_hr/gigatok_persist/models/google_t5-v1_1-xl`
- v2 新增参数从旧 checkpoint 随机初始化：
  - `text_type_embedding`
  - `text_projection.*`
- decoder-only finetune 冻结项生效：
  - `freeze_encoder=True`
  - `freeze_quantizer=True`
  - `freeze_codebook=True`
  - `freeze_post_quant_conv=True`
- HR smoke 日志出现：
  - `text_hr_loss`
  - `weighted_text_hr_loss`
  - `selected_decoder_layer`
  - `selected_text_layer`
  - `text_hr_valid_text_tokens_mean`
  - `text_hr_skipped_samples=0`

## smoke 与正式理想实验的差异
- 当前 smoke 为 2-step 可行性验证，不是正式训练。
- `max_images=8`、`global_batch_size=1`，仅用于检查代码路径。
- `distill_loss=False`，绕过 DINOv2 teacher 在线加载；正式实验如需保持完整原设定，需要预缓存 DINOv2 repo/code 或修复本地加载路径。
- `--no-wandb`，绕过当前环境中 wandb/protobuf 依赖冲突；正式实验可继续 no-wandb 或修复 wandb 环境。
- `ckpt_every=999999`，没有保存 checkpoint。
- `val_every=0`，没有跑验证集。
- 因复用同一个 `sub-exp-dir`，`train_metrics.csv` 中追加了历史 smoke 行；正式实验应使用全新的输出目录或先清空对应 smoke 目录。

## 残留
- 正常 smoke 产物：
  - `log.txt`
  - `metrics/train_metrics.csv`
  - `metrics/train_val_curves.png`
- 未生成新的 `.pt/.pth` checkpoint。
- 服务器 repo 中有 Python import 产生的 `__pycache__` / `.pyc` 工作区痕迹；不影响训练，但需要保持 git clean 时可以清理。
- `/tmp/gigatok_pycache` 是手动 `py_compile` 使用的临时 pycache，大小约 `824K`。

# 2026-04-25 DINOv2 distill + text HR smoke

## 已确认
- ModelArts 已存在 DINOv2 本地缓存：
  - repo: `/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main`
  - weights: `/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`
- 使用：
  - `TORCH_HOME=/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch`
  - `DINOV2_REPO_DIR=/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main`
- 离线加载 `torch.hub.load(..., "dinov2_vitb14", source="local")` 成功，`embed_dim=768`。

## 完整训练链路 smoke
- baseline + distill smoke 跑通：
  - output: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_smoke/baseline_2step_distill`
  - `distill_loss=True`
  - 2 steps 完成，最终日志 `Done!`
- HR + distill smoke 跑通：
  - output: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_smoke/hr_2step_distill`
  - `distill_loss=True`
  - `text_hr.enabled=True`
  - 2 steps 完成，最终日志 `Done!`
  - 日志出现 `text_hr_loss`、`weighted_text_hr_loss`、`selected_decoder_layer`、`selected_text_layer`。

## 发现并修补的问题
- 旧 online validation 使用 `JsonImageDataset`，只返回 image，不返回 text。
- v2 text-conditioned validation 因此会走 image-only decode，和正式 v2 推理路径不一致。
- 已在本地修补：
  - 当 `text_conditioning.enabled=True` 且 `dataset=textatlas_image_text` 时，validation 使用 TextAtlas manifest 读取 `(image, text)`。
  - `compute_reconstruction_metrics()` 支持用固定的第一个 layer pair 做 deterministic text-conditioned validation。
- 第一次服务器 text-aware validation smoke 暴露 NaN：
  - 训练路径使用 bf16 autocast，validation 的 VQ forward 未传 `dtype=ptdtype`，在 NPU 上默认落到 fp16。
  - 已修补 validation VQ forward 使用同样的 bf16 autocast dtype。

## validation / checkpoint 验证
- HR + distill + text-aware validation smoke 跑通：
  - output: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_smoke/hr_2step_distill_textval_bf16`
  - `val_every=1`
  - `val_max_images=2`
  - step 1: `Val MSE: 0.027725`, `Val PSNR: 15.5712`
  - step 2: `Val MSE: 0.034865`, `Val PSNR: 14.5761`
- checkpoint save smoke 跑通：
  - output: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_smoke/hr_1step_distill_save_last`
  - saved: `checkpoints/last.pt`
  - size: `6.1G`
- checkpoint load smoke 跑通：
  - loaded from saved v2 `last.pt`
  - output: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_smoke/hr_1step_distill_load_last`
  - 1 step 完成，最终日志 `Done!`

# 2026-04-26 text HR 100-step pilot

## 设置
- output root: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_pilot_100`
- checkpoint: `/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt`
- train manifest: `train_materialized_manifest_v2text.jsonl`
- val manifest: `val_materialized_manifest_v2text.jsonl`
- `iterations=100`
- `global_batch_size=1`
- `max_images=1000`
- `val_every=50`
- `val_max_images=64`
- `mixed_precision=bf16`
- `distill_loss=True`
- baseline:
  - `text_conditioning.enabled=True`
  - `text_hr.enabled=False`
  - `text_hr.hr_loss_weight=0.0`
- HR:
  - `text_conditioning.enabled=True`
  - `text_hr.enabled=True`
  - `text_hr.hr_loss_weight=1.0`
  - `tau=1.0`
  - 每 step 随机选 1 个 `[T5 layer, decoder layer]` pair。

## 结果
- baseline 跑通：
  - output: `baseline_100step_b1_seed0`
  - step 50: `Val MSE: 0.047492`, `Val PSNR: 13.2338`
  - step 100: `Val MSE: 0.039176`, `Val PSNR: 14.0698`
  - 稳定阶段约 `2.0 steps/s`
- HR 跑通：
  - output: `hr_100step_b1_seed0`
  - 日志出现 `text_hr_loss`、`weighted_text_hr_loss`、`selected_decoder_layer`、`selected_text_layer`
  - `text_hr_skipped_samples=0`
  - step 50: `Val MSE: 0.048406`, `Val PSNR: 13.1510`
  - step 100: `Val MSE: 0.039680`, `Val PSNR: 14.0143`
  - 稳定阶段约 `1.9 steps/s`

## 结论
- HR loss 已进入总 loss；step 100 的 HR 总 loss 相比 baseline 高约 `0.95`，与 `weighted_text_hr_loss=0.9518` 量级一致。
- 当前 per-sample SVD + 16 heads stack 没有造成明显速度灾难，100-step pilot 中约 5% 开销。
- 100 step 只能证明可行性，不能判断最终质量差异；下一步建议跑 500 或 1000 step pilot。

# 2026-04-26 text HR 500-step pilot

## 设置
- output root: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_pilot_500`
- checkpoint: `/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt`
- train manifest: `train_materialized_manifest_v2text.jsonl`
- val manifest: `val_materialized_manifest_v2text.jsonl`
- `iterations=500`
- `global_batch_size=1`
- `max_images=5000`
- `val_every=100`
- `val_max_images=500`
- `mixed_precision=bf16`
- `distill_loss=True`
- baseline:
  - output: `baseline_500step_b1_seed0`
  - `text_conditioning.enabled=True`
  - `text_hr.enabled=False`
- HR:
  - output: `hr_500step_b1_seed0`
  - `text_conditioning.enabled=True`
  - `text_hr.enabled=True`
  - `text_hr.hr_loss_weight=1.0`
  - `tau=1.0`

## validation
| step | baseline MSE | baseline PSNR | HR MSE | HR PSNR |
| --- | ---: | ---: | ---: | ---: |
| 100 | 0.037492 | 14.2606 | 0.037046 | 14.3126 |
| 200 | 0.029645 | 15.2804 | 0.028779 | 15.4093 |
| 300 | 0.025466 | 15.9404 | 0.025169 | 15.9913 |
| 400 | 0.024867 | 16.0437 | 0.024752 | 16.0639 |
| 500 | 0.023550 | 16.2800 | 0.023455 | 16.2976 |

## 观察
- HR 在 500-step pilot 的所有 validation checkpoint 上 MSE/PSNR 均略好于 baseline，但差距很小，不能作为最终质量结论。
- step 500:
  - baseline train loss: `2.0098`
  - HR train loss: `2.9481`
  - HR `text_hr_loss`: `0.96194`
  - 二者 train loss 差值约 `0.9383`，与 `weighted_text_hr_loss` 量级一致，说明 HR 项已进入总 loss。
- 稳定阶段速度：
  - baseline 大约 `2.0 steps/s`
  - HR 大约 `1.9-2.0 steps/s`
  - 当前设置下 HR SVD 开销较小。

## 当前判断
- 代码路径可行，HR loss 确实参与优化。
- 500 step 仍偏短；后续需要更长 pilot、可视化重建结果，或 OCR/文本识别指标来判断是否提升文字质量。

# 2026-04-26 OCR 文本重建评估

## 目标
- 将文字内容重建从训练 loss 中解耦出来，作为 validation / test 后处理评估指标。
- 先不把 OCR 放进训练约束，避免引入不可微、慢速和 OCR 偏置问题。

## 实现
- 扩展 `scripts/stage1/evaluate_textatlas_reconstruction.py`：
  - 支持直接从 TextAtlas materialized manifest 读取 `image_path`、`subset`、`text`。
  - 对 `text_conditioning.enabled=True` 的 v2 config，使用 T5 text feature 做 text-conditioned reconstruction。
  - 默认使用 deterministic `text_layer_pair_index=0`，即第一个配置 pair，便于 baseline / HR 可比。
  - 支持 OCR backend：
    - `pytesseract`
    - `paddleocr`
    - `easyocr`
  - 输出图像重建指标：
    - `mse`
    - `mae`
    - `psnr`
    - `ssim`
  - 输出文字重建指标：
    - `ocr_exact_acc`
    - `ocr_exact_acc_ci`
    - `ocr_cer`
    - `ocr_ned_similarity`
    - `ocr_edit_distance`
  - 可保存逐样本 OCR 结果到 `ocr_predictions.jsonl`。

## 检查
- 本地已通过语法检查：
  - `python3 -m py_compile scripts/stage1/evaluate_textatlas_reconstruction.py`
- 本地 Python 没有安装 `torch`，因此完整 import / reconstruction / OCR 需要在 ModelArts 环境验证。

# 2026-04-27 单图过拟合诊断工具

## 目标
- 将单图测试相关代码集中到独立目录，避免继续把临时诊断逻辑散落到主训练脚本。
- 用 dense / medium / sparse 三张 train 样本做单图 overfit，验证当前 text-HR 结构是否能在单样本上降 loss。
- 对单图 checkpoint 做逐层 decoder cross-attention SVD 和特征统计，排查 HR 分支是否真正生效。

## 新增目录
- `scripts/stage1/single_image_debug/`

## 新增文件
- `README.md`
  - 记录单图 manifest 生成、单图 overfit、逐层诊断命令。
- `make_single_image_manifests.py`
  - 从当前 train manifest 抽取三张指定样本。
  - 默认目标：
    - dense: `CleanTextSynth:train:2731`
    - medium: `TextScenesHQ:train:2803`
    - sparse: `LongWordsSubset-A:train:3149`
- `run_single_image_overfit.sh`
  - 单图 HR overfit 启动脚本。
  - 默认输出到 `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug`。
- `diagnose_single_image.py`
  - 加载指定 checkpoint，对单图逐层 forward。
  - 输出 `per_layer_metrics.csv`、`summary.json`、重建图、diff 图和每层 SVD 奇异值。
  - 诊断指标包括当前代码使用的 `tau` 版本 HR loss，以及 normalized spectrum loss 作为参考指标。

## 检查
- 本地已通过语法检查：
  - `python3 -m py_compile scripts/stage1/single_image_debug/make_single_image_manifests.py scripts/stage1/single_image_debug/diagnose_single_image.py`
  - `bash -n scripts/stage1/single_image_debug/run_single_image_overfit.sh`
- 本地没有 ModelArts NPU / torch-npu 环境，完整运行需要在服务器验证。

## 2026-04-27 自查修正
- 修复 `diagnose_single_image.py` 中的 encoder feature 读取：
  - 原先 `model.encode(image)` 会按默认 `return_code=True` 返回 `(quant, emb_loss, info)`。
  - 现在显式使用 `model.encode(image, return_code=False)`，与训练 forward 一致，第三项才是 `encoder_spatial`。
- 为 `diagnose_single_image.py` 增加 `--seed`，默认 `0`：
  - 诊断原始 VQ checkpoint 时，`text_projection` / `text_type_embedding` 缺失并会重新初始化。
  - 固定 seed 可以让这部分初始化可复现。
- 修复 `run_single_image_overfit.sh` 默认 config：
  - HR 默认使用仓库内 `configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml`。
  - baseline 默认使用仓库内 `configs/vq/VQ_BL256_dino_disc_text_baseline_v2.yaml`。
  - 仍支持在服务器通过 `CONFIG=configs/vq/_pilot_text_hr_local.yaml` 使用本地 T5 配置。
- `README.md` 已补充：
  - `MODE=hr` 和 `MODE=baseline` 的同图 overfit 命令。
  - 向师姐汇报“每一层”时使用 `--layer-mode all_decoder`。
- `diagnose_single_image.py` 的 `--layer-mode` 默认值已改为 `all_decoder`，避免误只诊断 8-15 配置层。

## 2026-04-27 dense 单图验证结果

### 运行样本
- dense: `CleanTextSynth:train:2731`
- manifest: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/manifests/dense.jsonl`
- 图像: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/images/train/CleanTextSynth/0000002731.png`
- 文本: `In August of the same year`
- T5 valid tokens: 7

### 单图 overfit
- HR:
  - 输出目录: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/dense_hr_1img_1000step`
  - 1000 step train: `rec_loss=0.0003`, `direct_rec_loss=0.0003`, `feature_rec_loss=0.0034`
  - 1000 step val: `MSE=0.000076`, `MAE=0.005945`, `PSNR=41.2188`
  - HR 相关: `text_hr_loss=0.9890`, `text_hr_sigma_mean=0.01097`
- baseline:
  - 输出目录: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/dense_baseline_1img_1000step`
  - 1000 step train: `rec_loss=0.0003`, `direct_rec_loss=0.0002`, `feature_rec_loss=0.0040`
  - 1000 step val: `MSE=0.000065`, `MAE=0.005311`, `PSNR=41.8767`
- 结论:
  - 单图重建路径能 overfit，说明 decoder-only finetune、T5 条件输入、checkpoint 加载和 validation 主链路没有明显阻塞 bug。
  - dense 单图上，HR 没有带来更好的重建 PSNR；baseline 略高。
  - HR loss 本身没有明显下降，后期仍接近 `1.0`，需要重点检查当前 HR 目标和实际优化方向。

### 逐层诊断
- 诊断输出:
  - pretrain: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/diagnostics/dense_pretrain_hrconfig`
  - HR step 100: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/diagnostics/dense_hr_step0100`
  - HR step 500: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/diagnostics/dense_hr_step0500`
  - HR last: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/diagnostics/dense_hr_last`
  - baseline last: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/diagnostics/dense_baseline_last`
- 全 24 层平均:
  - pretrain: `PSNR=21.01`, `text_mass=0.0178`, `tau_loss=0.9163`, `effective_rank=4.51`
  - HR step 100: `PSNR=24.57`, `text_mass=0.0308`, `tau_loss=0.9088`, `effective_rank=3.71`
  - HR step 500: `PSNR=40.83`, `text_mass=0.0173`, `tau_loss=0.8968`, `effective_rank=4.09`
  - HR last: `PSNR=43.58`, `text_mass=0.0117`, `tau_loss=0.9358`, `effective_rank=4.23`
  - baseline last: `PSNR=43.72`, `text_mass=0.0113`, `tau_loss=0.9466`, `effective_rank=4.29`
- 当前 HR 实际训练层 8-15 平均:
  - HR step 500: `text_mass=0.00323`, `tau_loss=0.9835`, `sigma_mean=0.0165`
  - HR last: `text_mass=0.00236`, `tau_loss=0.9915`, `sigma_mean=0.00848`
  - baseline last: `text_mass=0.00244`, `tau_loss=0.9914`, `sigma_mean=0.00863`
- 结论:
  - 8-15 层中，HR last 与 baseline last 的 `text_mass`、`tau_loss`、`sigma_mean` 几乎相同。
  - 这说明当前 HR 项虽然被计算和记录，但在这张 dense 单图上没有把被选层的 text cross-attention 谱推向预期目标。
  - 更可疑的是训练层的 text attention mass 在后期接近 0，导致 SVD 奇异值整体很小，`abs(sigma - tau)` 形式的 loss 长期接近 `tau=1`。

### 诊断脚本修正
- `diagnose_single_image.py` 中 `model.encode(image, return_code=False)` 返回的 `diff` 在当前模型里可能是 list。
- 增加 `scalar_mean()`，递归汇总 tensor/list/dict 形式的辅助 loss，避免诊断 summary 写入时报错。

## 2026-04-27 medium / sparse 单图验证结果

### 运行样本
- medium: `TextScenesHQ:train:2803`
  - manifest: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/manifests/medium.jsonl`
  - 图像: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/images/train/TextScenesHQ/0000002803.png`
  - 文本: `performancebehave standard Loperation Mode business funclion perform procedure practice operate Manner affair working act Method shutterstock.com151811543`
  - T5 valid tokens: 30
- sparse: `LongWordsSubset-A:train:3149`
  - manifest: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/manifests/sparse.jsonl`
  - 图像: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/images/train/LongWordsSubset-A/0000003149.png`
  - 文本: `read TDT, DETHE, TASSE, IN, SIP, EVERY, SYMPHONY words`
  - T5 valid tokens: 26

### 单图 overfit 终点
- dense:
  - HR: `Val MSE=0.000076`, `Val MAE=0.005945`, `Val PSNR=41.2188`, `text_hr_loss=0.9890`, `text_hr_sigma_mean=0.01097`
  - baseline: `Val MSE=0.000065`, `Val MAE=0.005311`, `Val PSNR=41.8767`
- medium:
  - HR: `Val MSE=0.000124`, `Val MAE=0.006862`, `Val PSNR=39.0490`, `text_hr_loss=0.9965`, `text_hr_sigma_mean=0.00355`
  - baseline: `Val MSE=0.000121`, `Val MAE=0.006790`, `Val PSNR=39.1773`
- sparse:
  - HR: `Val MSE=0.000081`, `Val MAE=0.006054`, `Val PSNR=40.8910`, `text_hr_loss=0.9958`, `text_hr_sigma_mean=0.00419`
  - baseline: `Val MSE=0.000071`, `Val MAE=0.005656`, `Val PSNR=41.4754`

### 逐层诊断汇总
- 统计口径:
  - `all_psnr`: 全 24 个 decoder layer 的诊断重建 PSNR 平均。
  - `mid_*`: 当前 HR 实际训练层 8-15 的平均。
- dense:
  - HR: `all_psnr=43.5802`, `mid_psnr=43.6929`, `mid_text_mass=0.002355`, `mid_tau_loss=0.991517`, `mid_norm_loss=0.028420`, `mid_eff_rank=3.9874`, `mid_sigma_mean=0.008483`
  - baseline: `all_psnr=43.7233`, `mid_psnr=43.8281`, `mid_text_mass=0.002440`, `mid_tau_loss=0.991372`, `mid_norm_loss=0.030773`, `mid_eff_rank=3.8841`, `mid_sigma_mean=0.008629`
- medium:
  - HR: `all_psnr=38.9960`, `mid_psnr=39.3435`, `mid_text_mass=0.015595`, `mid_tau_loss=1.006255`, `mid_norm_loss=0.009909`, `mid_eff_rank=8.3071`, `mid_sigma_mean=0.022276`
  - baseline: `all_psnr=39.2355`, `mid_psnr=39.4534`, `mid_text_mass=0.016286`, `mid_tau_loss=1.007053`, `mid_norm_loss=0.009828`, `mid_eff_rank=8.4986`, `mid_sigma_mean=0.022796`
- sparse:
  - HR: `all_psnr=43.1849`, `mid_psnr=43.3017`, `mid_text_mass=0.008067`, `mid_tau_loss=1.001221`, `mid_norm_loss=0.012869`, `mid_eff_rank=6.2981`, `mid_sigma_mean=0.017622`
  - baseline: `all_psnr=43.8528`, `mid_psnr=43.9312`, `mid_text_mass=0.003596`, `mid_tau_loss=0.996961`, `mid_norm_loss=0.010407`, `mid_eff_rank=6.8000`, `mid_sigma_mean=0.003039`

### 当前判断
- 三张单图都能 overfit 到较低重建误差，说明当前数据读取、T5 本地加载、decoder-only finetune、checkpoint 兼容加载、validation 和图像保存主链路没有明显阻塞问题。
- HR 版本在三张图的最终重建 PSNR 都没有超过 baseline，且差距方向一致为 baseline 略好。
- HR loss 在训练末期仍接近 `1.0`，没有出现“被优化项持续下降”的现象。
- 当前实际训练层 8-15 的 `text_mass` 普遍偏低，`sigma_mean` 很小；这解释了为什么当前 `abs(sigma - tau)` 形式的 HR loss 会长期接近 `tau=1`。
- HR 与 baseline 在 8-15 层的 `tau_loss`、`effective_rank`、`normalized_spectrum_loss` 差别很小，说明当前 HR 分支虽然被计算和反传，但没有稳定改变被约束层的 attention 谱。

### 下一步建议
- 不建议直接扩大训练步数来证明 HR 有效；单图已经显示 HR 约束本身没有明显下降。
- 下一步应做最小代码级核查：
  - 在训练日志中同时记录当前代码的 `tau_loss` 和论文定义的 normalized spectrum loss。
  - 检查 HR loss 是否应该改回 `p = sigma / sum(sigma)` 后的均匀谱约束，而不是直接约束未归一化奇异值接近 `tau=1`。
  - 同时记录被选层的 `text_attention_mass`，防止模型通过降低 text attention mass 绕开 HR 约束。

## 2026-04-27 单图 baseline 覆盖为原生 tokenizer

## 背景
- 前一轮单图对比里的 `baseline` 是 text baseline：
  - 有 T5 text encoder。
  - 有 `text_projection`。
  - 有 decoder text cross-attention。
  - 只是不启用 HR loss。
- 这不能代表原生 GigaTok tokenizer。
- 现在将 `MODE=baseline` 覆盖为原生 tokenizer baseline，不再默认提供 text baseline。

## 修改
- `scripts/stage1/single_image_debug/run_single_image_overfit.sh`
  - 将 `MODE=baseline` 的默认配置改为 `configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml`。
  - 该配置不包含 `text_conditioning` 和 `text_hr`，因此不加 T5、不加 text projection、不加 decoder text cross-attention、不加 HR。
  - `MODE=original` 仅保留为兼容别名，推荐统一使用 `MODE=baseline`。
- `scripts/stage1/single_image_debug/README.md`
  - 明确两条线：
    - `MODE=baseline`: 原生 tokenizer baseline。
    - `MODE=hr`: text + HR。
  - 说明 baseline 没有 text cross-attention，不能用当前 SVD 诊断脚本看 text attention 谱，只比较重建日志和图像。

## baseline 运行命令
```bash
MODE=baseline TAG=dense ITERS=1000 ASCEND_RT_VISIBLE_DEVICES=0 \
  bash scripts/stage1/single_image_debug/run_single_image_overfit.sh

MODE=baseline TAG=medium ITERS=1000 ASCEND_RT_VISIBLE_DEVICES=0 \
  bash scripts/stage1/single_image_debug/run_single_image_overfit.sh

MODE=baseline TAG=sparse ITERS=1000 ASCEND_RT_VISIBLE_DEVICES=0 \
  bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
```

## 指标
- baseline 看：
  - `rec_loss`
  - `direct_rec_loss`
  - `feature_rec_loss`
  - `Val MSE`
  - `Val MAE`
  - `Val PSNR`
- baseline 不看：
  - `text_hr_loss`
  - `text_hr_sigma_mean`
  - `text_attention_mass`
  - text attention SVD
- 原因：baseline 没有 text 分支，也没有 decoder text cross-attention。

## 2026-04-27 修复原生 baseline 单图验证集读取

## 背景
- 服务器运行 `MODE=baseline TAG=sparse ITERS=1000` 时失败：
  - `TypeError: unhashable type: 'slice'`
  - 位置在 `JsonImageDataset.__init__` 的 `image_paths[:max_images]`。
- 原因：
  - 原生 baseline 的 `text_conditioning.enabled=False`。
  - 旧逻辑在 validation 分支退回到 `JsonImageDataset`。
  - `JsonImageDataset` 只支持 JSON 数组图片路径，不支持 TextAtlas materialized JSONL。
  - 单图 manifest 是 TextAtlas JSONL，因此 validation 初始化失败。

## 修改
- `tokenizer/tokenizer_image/vq/vq_train.py`
  - 对 `args.dataset == "textatlas_image_text"`，无论是否启用 text conditioning，validation 都使用 `build_dataset(...)` 读取 JSONL。
  - 对 `textatlas_image_text`，无论是否启用 text conditioning，训练和验证 transform 都使用 resize+pad 到 `256x256`。

## 影响
- `MODE=baseline` 仍然是原生 GigaTok tokenizer：
  - 不加载 T5。
  - 不创建 text projection。
  - 不使用 decoder text cross-attention。
  - 不计算 HR loss。
- 但数据读取和预处理与 HR 单图实验保持一致，适合做严格单图对照。

## 2026-04-27 标注 Text-HR v2 主要改动位置

## 背景
- 为了方便师姐快速审阅当前分支相对原生 GigaTok 的改动，需要在代码里标注关键改动点。

## 修改
- 新增 `docs/text_hr_code_map.md`：
  - 汇总 config、模型、loss、训练、单图诊断各自的改动入口。
  - 说明可以在代码里搜索 `Text-HR v2` 快速定位。
- 在核心代码添加轻量注释标记：
  - `tokenizer/tokenizer_image/vq/vq_vit_model.py`
  - `tokenizer/tokenizer_image/vq/blocks.py`
  - `tokenizer/tokenizer_image/vq/vq_loss.py`
  - `tokenizer/tokenizer_image/vq/vq_train.py`
  - `configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml`
  - `configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml`

## 影响
- 仅增加注释和导览文档，不改变训练逻辑、模型结构、loss 公式或默认参数。

## 2026-04-27 补充原版 GigaTok 对比导览

## 背景
- 需要明确当前主要修改是相对于官方原版 GigaTok，而不是相对于中间实验分支。
- 本地已有官方 remote：
  - `upstream=https://github.com/SilentView/GigaTok.git`
  - `upstream/master=baf3de042034318aa762c28db4d14231efce8b0c`

## 修改
- 新增 `docs/original_gigatok_diff_map.md`。
- 该文档以 `upstream/master...HEAD` 为口径，整理：
  - 核心 Text-HR v2 方法改动。
  - TextAtlas 数据接入。
  - NPU / validation / checkpoint 兼容。
  - 单图 overfit 与逐层诊断工具。
  - 给师姐看的最小文件列表。

## 统计口径
```bash
git diff --stat upstream/master...HEAD
git diff --name-status upstream/master...HEAD
```

当前总量约：
```text
44 files changed, 9432 insertions(+), 114 deletions(-)
```

## 2026-04-27 dense HR 单图重跑结果复查

## 背景
- 之前 dense 样本选错，实际只有 7 个 T5 token，不能代表 dense text。
- 服务器已重新生成并训练 dense HR：
  - run: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/dense_hr_1img_1000step`
  - manifest: `CleanTextSynth:train:2713`
  - text chars: 408
  - valid T5 tokens: 82
- 新训练 checkpoint:
  - `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/dense_hr_1img_1000step/checkpoints/last.pt`

## 训练结果
- dense HR 1000 step 可以 overfit：
  - Val MSE: `0.064851 -> 0.000164`
  - Val PSNR: `11.88 -> 37.84`
  - Train Loss: `82.44 -> 0.5199`
- 但 HR 分支没有明显学动：
  - `text_hr_loss` 首次约 `0.9703`，末尾约 `0.9990`
  - `text_hr_sigma_mean` 首次约 `0.0297`，末尾约 `0.0010`
  - `text_hr_valid_text_tokens_mean=82`
  - `text_hr_skipped_samples=0`

## 新逐层诊断
- 已对新 `last.pt` 重新跑逐层诊断：
  - output: `/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/diagnostics/dense_hr_last_rerun_1719`
  - config: `configs/vq/_pilot_text_hr_local.yaml`
  - layer mode: `all_decoder`
- 训练实际使用层 `8-15` 的聚合：
  - mean MSE: `1.496e-4`
  - mean PSNR: `38.25`
  - mean `text_attention_mass_mean`: `0.00393`
  - mean `sigma_top1_ratio`: `0.424`
  - mean `effective_rank`: `16.34 / 82`
  - mean `current_code_tau_loss`: `0.99917`
- 结论：
  - dense token 数已足够，不再是 7-token 误选样本。
  - 训练层的重建最好，但 text attention mass 极低，说明 decoder 主要仍走 image/code 路径。
  - 当前 v2 tau HR loss 没有把 image-to-text attention 推成高秩，反而在后期表现为 text attention 奇异值整体变小。

## 2026-04-27 三张单图训练前后 SVD 对比

## 操作
- 为避免旧 dense 7-token 诊断混入，重新对当前三张图的原始 checkpoint 跑训练前诊断。
- 只扫 HR 实际训练层 `8-15`：
  - `dense_pretrain_current_configpairs`
  - `medium_pretrain_current_configpairs`
  - `sparse_pretrain_current_configpairs`
- 训练后对比：
  - dense: `dense_hr_last_rerun_1719`
  - medium: `medium_hr_last`
  - sparse: `sparse_hr_last`

## 8-15 层平均结果
| image | tokens | phase | text mass | sigma_mean | sigma_max | top1_ratio | effective_rank | tau_loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| dense | 82 | pre | 16.56% | 0.02851 | 0.76891 | 0.291 | 24.12 | 0.9772 |
| dense | 82 | post | 0.39% | 0.000827 | 0.02879 | 0.424 | 16.34 | 0.9992 |
| medium | 30 | pre | 7.29% | 0.05074 | 0.57955 | 0.356 | 12.26 | 0.9578 |
| medium | 30 | post | 1.56% | 0.02228 | 0.59517 | 0.527 | 8.31 | 1.0063 |
| sparse | 26 | pre | 6.64% | 0.04662 | 0.37843 | 0.347 | 10.94 | 0.9534 |
| sparse | 26 | post | 0.81% | 0.01762 | 0.40678 | 0.562 | 6.30 | 1.0012 |

## 解释
- 训练后重建 MSE 下降，但 text attention mass 在 HR 训练层显著下降。
- dense 训练后绝对奇异值整体从约 `0.7689/0.2605/0.1551/...` 降到约 `0.0288/0.0081/0.0025/...`。
- medium/sparse 的平均 `sigma_max` 没明显下降，是因为 layer 9 训练后出现单一大奇异值：
  - medium layer 9: `sigma_max=4.42377`, `top1_ratio=0.956`, `effective_rank=1.32`
  - sparse layer 9: `sigma_max=2.95964`, `top1_ratio=0.962`, `effective_rank=1.25`
- 这不是高秩变好，而是典型 top1 塌缩。
- 结论：当前单图训练后，HR 实际训练层没有形成更高秩的 image-to-text attention；模型更倾向降低 text attention mass 或集中到单一奇异方向。

## 2026-04-27 dense 三种师姐 Frobenius HR loss 单图 probe 结果

## 设置
- 三个 run 都使用当前 dense 单图 `CleanTextSynth:train:2713`，`valid_t5_tokens=82`。
- 训练 1000 step：
  - `dense_sigma_mean_1000step`
  - `dense_gram_identity_1000step`
  - `dense_log_rank_1000step`
- 对三个 `last.pt` 用 `diagnose_single_image.py --layer-mode config_pairs` 重扫 decoder layer `8-15`。
- 同时用新诊断脚本重扫旧 tau run：
  - `dense_old_tau_last_newmetrics`

## 8-15 层平均结果
| run | MSE | PSNR | text mass | raw fro norm | participation ratio | effective rank | energy top1 | log PR loss | gram loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pretrain | 0.063203 | 11.99 | 16.56% | 3.520 | 0.125 | 3.51 | 0.678 | -2.219 | 7.53e-05 |
| old tau | 0.000150 | 38.25 | 0.39% | 0.122 | 0.066 | 1.75 | 0.882 | -1.631 | 1.16e-04 |
| sigma_mean_mse | 0.000164 | 37.85 | 0.92% | 1.697 | 0.078 | 1.99 | 0.862 | -1.693 | 1.11e-04 |
| gram_scaled_identity | 0.000152 | 38.18 | 0.51% | 0.217 | 0.200 | 10.49 | 0.229 | -2.759 | 1.81e-05 |
| log_participation_ratio | 0.000144 | 38.43 | 0.93% | 1.693 | 0.076 | 2.00 | 0.856 | -1.651 | 1.10e-04 |

## 结论
- 三种新方法都能完成单图重建，PSNR 都在 `37.85-38.43`。
- `gram_scaled_identity` 是目前唯一明显改善归一化谱形状的方法：
  - participation ratio: `0.066(old tau) -> 0.200`
  - effective rank: `1.75(old tau) -> 10.49`
  - energy top1: `0.882(old tau) -> 0.229`
  - gram loss: `1.16e-04(old tau) -> 1.81e-05`
- `sigma_mean_mse` 和 `log_participation_ratio` 虽然重建正常，但 8-15 层平均谱仍偏低秩：
  - participation ratio 约 `0.076-0.078`
  - effective rank 约 `2`
  - energy top1 约 `0.86`
- 重要问题仍未解决：
  - 三种方法训练后的 text attention mass 都很低，`gram_scaled_identity` 只有约 `0.51%`，`sigma_mean/log_rank` 约 `0.92-0.93%`。
  - 因此 `gram_scaled_identity` 可以说改善了 Frobenius-normalized spectrum，但还不能说明模型真正依赖 text；下一步需要跑 medium/sparse 以及 correct/empty/wrong text sensitivity。

## 2026-04-27 给师姐看的原版 GigaTok 行号说明

## 背景
- 需要给师姐看当前代码相对官方原版 GigaTok 的主要修改点，并标注每个修改点在当前分支中的具体行号范围。
- 对比口径继续使用：
  - `upstream=https://github.com/SilentView/GigaTok.git`
  - `upstream/master`
  - merge-base: `baf3de042034318aa762c28db4d14231efce8b0c`

## 修改
- 新增 `docs/original_gigatok_line_review.md`。
- 文档按模块列出当前分支行号：
  - decoder cross-attention 返回 post-softmax attention。
  - VQ tokenizer 接入 frozen T5 text feature。
  - image-to-text attention SVD / HR loss。
  - stage-1 训练主线接入。
  - HR 与原生 baseline config。
  - TextAtlas 数据读取。
  - 单图 overfit 与逐层诊断脚本。
  - NPU / validation / checkpoint 兼容相关工程改动。

## 影响
- 仅增加说明文档，不改变训练代码、模型结构或实验配置。

## 2026-04-27 更新仓库首页 README

## 背景
- 原根目录 `README.md` 仍是官方 GigaTok 发布页，不利于从 GitHub 分支首页快速理解当前 Text-HR stage-1 pilot。
- 需要让 `codex/text-hr-decoder` 分支主页直接展示当前最新做法、训练命令和诊断入口。

## 修改
- 将根目录 `README.md` 改为当前分支说明：
  - Text-HR v2 方法概述。
  - 核心代码文件和 config。
  - TextAtlas 数据口径。
  - 服务器环境变量。
  - 4 卡 baseline / 4 卡 HR 训练命令。
  - 单图 overfit 与逐层诊断命令。
  - 当前已知结论和后续优先事项。
  - 给师姐看的最小材料列表。

## 影响
- 仅修改文档，不改变训练代码、模型结构或实验配置。

## 2026-04-27 新增 HR-only probe 配置

## 背景
- 需要验证如果去掉 reconstruction / perceptual / codebook / distill 等非 HR 目标，只优化 Text-HR loss，`text_hr_loss` 是否能单独降下去。
- 这个实验用于判断当前 HR loss 和 attention 分支本身是否可优化，不作为主实验配置。

## 修改
- 修正 `tokenizer/tokenizer_image/vq/vq_loss.py` 中 `codebook_weight` 未参与总 loss 的问题：
  - 现有主配置 `codebook_weight=1.0`，行为不变。
  - 新 HR-only 配置可设置 `codebook_weight=0.0`，确保 codebook / commit / entropy 项不进入总 loss。
- 新增 `configs/vq/VQ_BL256_dino_disc_text_hr_only_v2.yaml`：
  - 保留 frozen T5 text conditioning 和 Text-HR。
  - 关闭 reconstruction / perceptual / codebook / feature distill / projection / GAN loss。
  - 保持 decoder-only finetune 和随机选择 8-15 层 pair。

## 建议验证
- 先跑 2 step smoke，确认日志中的 `Train Loss` 基本等于 `weighted_text_hr_loss`。
- 再跑 dense 单图 1000 step，观察：
  - `text_hr_loss`
  - `weighted_text_hr_loss`
  - `text_hr_sigma_mean`
  - `text_attention_mass_mean`
  - `effective_rank`
- 训练后用 `diagnose_single_image.py` 对比 pretrain 和 HR-only last 的奇异值谱。

## 2026-04-27 修正 Text-HR SVD loss 的归一化口径

## 背景
- 单图诊断发现旧版 `abs(sigma - tau)` 会鼓励模型把 image-to-text attention 的整体能量压低，从而让 raw singular values 远离 1 甚至接近 0。
- 师姐建议在 SVD 前先对 attention 矩阵做 Frobenius norm 归一化，固定矩阵总能量后再看奇异值分布。

## 修改
- `high_rank_image_text_attention_loss` 新增 `svd_mode="frobenius_uniform"`：
  - 对 `[heads * image_tokens, valid_text_tokens]` 矩阵先做 Frobenius norm 归一化。
  - 对归一化矩阵做 float32 SVD。
  - 用奇异值平方能量分布 `sigma^2 / sum(sigma^2)` 对齐均匀分布，避免通过压低 attention scale 降 loss。
- 保留旧 `legacy_tau` / `per_sample_all_heads_sqrt_norm` 模式用于复现实验。
- 训练日志新增：
  - `text_hr_normed_sigma_mean`
  - `text_hr_fro_norm_mean`
  - `text_hr_effective_rank_mean`
  - `text_hr_energy_top1_mean`
- `VQ_BL256_dino_disc_text_hr_v2.yaml` 和 `VQ_BL256_dino_disc_text_hr_only_v2.yaml` 默认切到 `svd_mode: "frobenius_uniform"`。
- 单图诊断脚本同时保存 raw singular values 和 Frobenius-normalized singular values，并输出 `frobenius_uniform_loss` / `frobenius_effective_rank` / `frobenius_energy_top1_ratio`。

## 建议验证
- 先重新跑 HR-only 单图 2 step smoke，确认 `Train Loss` 约等于 `weighted_text_hr_loss`。
- 再分别对 dense / medium / sparse 跑 HR-only 1000 step。
- 对训练前后 checkpoint 跑 `diagnose_single_image.py`，重点看：
  - `frobenius_uniform_loss` 是否下降。
  - `frobenius_effective_rank` 是否上升。
  - `frobenius_energy_top1_ratio` 是否下降。
  - `frobenius_norm` 是否仍明显塌缩；若塌缩，说明 attention-to-text mass 仍需要额外约束。

## 2026-04-27 新增师姐三种 Frobenius Text-HR loss

## 背景
- 旧 tau loss 直接让 raw singular values 接近 1，不适合 post-softmax attention。
- 单图诊断已经显示模型会压低 image-to-text attention mass，导致 raw singular values 远离 1 甚至接近 0。
- 这次按师姐建议，把新方法统一建立在 SVD 前 Frobenius norm 归一化上。

## 共同前置
- 对 selected decoder layer 的 post-softmax image-to-text attention slice 构造矩阵：
  - `A = text_attn[batch_idx, :, :, valid_mask]`
  - `A = A.reshape(num_heads * num_queries, valid_token_count).float()`
- 新方法统一做：
  - `A_norm = A / (torch.norm(A, p="fro") + eps)`
  - `sigma = torch.linalg.svdvals(A_norm)`
- `fro_norm` 日志记录归一化前 raw `A` 的 Frobenius norm，用来观察 text attention mass 是否继续塌缩。
- legacy tau 模式继续保留原有 `/ sqrt(num_heads)` 口径，方便复现旧实验。

## 三种方法
- `sigma_mean_mse`：
  - `loss = mean((sigma - mean(sigma))^2)`
- `gram_scaled_identity`：
  - `G = A_norm.T @ A_norm`
  - `target = I / T_valid`
  - `loss = mean((G - target)^2)`
  - 这是 Frobenius 归一化前提下对师姐 `A.T @ A -> I` 的尺度修正版；因为 `trace(G)=1`，所以不用未缩放的 `I`。
- `log_participation_ratio`：
  - `rank_score = (sum(sigma)^2) / (sum(sigma^2) + eps)`
  - `loss = -log(rank_score + eps)`
  - 严格按师姐图中公式，不额外加 `log(r)` 平移，因此 loss 可以为负。

## 修改
- 扩展 `high_rank_image_text_attention_loss` 的 `svd_mode`，新增：
  - `sigma_mean_mse`
  - `gram_scaled_identity`
  - `log_participation_ratio`
- 新增公共 helper，训练和单图诊断共用同一套公式。
- 训练日志新增：
  - `text_hr_raw_sigma_*`
  - `text_hr_normed_sigma_*`
  - `text_hr_participation_rank_mean`
  - `text_hr_participation_rank_ratio_mean`
  - `text_hr_gram_loss_mean`
- 单图诊断 `per_layer_metrics.csv` 新增：
  - `sigma_mean_mse_loss`
  - `gram_scaled_identity_loss`
  - `log_participation_ratio_loss`
  - `participation_rank`
  - `participation_rank_ratio`
  - `energy_top1_ratio`
- 新增 3 个 probe config：
  - `configs/vq/VQ_BL256_dino_disc_text_hr_probe_sigma_mean_v2.yaml`
  - `configs/vq/VQ_BL256_dino_disc_text_hr_probe_gram_identity_v2.yaml`
  - `configs/vq/VQ_BL256_dino_disc_text_hr_probe_log_rank_v2.yaml`

## 建议验证
- 先 smoke 师姐图中的 log-rank 方法：
  ```bash
  MODE=hr TAG=dense ITERS=2 \
  CONFIG=configs/vq/VQ_BL256_dino_disc_text_hr_probe_log_rank_v2.yaml \
  ASCEND_RT_VISIBLE_DEVICES=0 \
  bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
  ```
- smoke 通过后，对 dense / medium / sparse 分别跑三个 probe config。
- 对比：
  - `text_hr_loss`
  - `text_hr_participation_rank_ratio_mean`
  - `text_hr_energy_top1_mean`
  - `text_hr_fro_norm_mean`

## 2026-04-27 单图三种 Frobenius loss 结果补充

## dense / medium / sparse 结论
- 三种文字密度的单图 1000 step 都能把重建误差降下来。
- 在归一化 SVD 谱上，`gram_scaled_identity` 是三种方法里最稳定有效的：
  - dense：8-15 层平均 effective rank = 10.49，energy top1 = 0.229。
  - medium：8-15 层平均 effective rank = 9.49，energy top1 = 0.258。
  - sparse：8-15 层平均 effective rank = 8.97，energy top1 = 0.196。
- `sigma_mean_mse` 和 `log_participation_ratio` 在 medium / sparse 上基本仍保持低秩：
  - medium 的 effective rank 约 1.80 / 1.69。
  - sparse 的 effective rank 约 1.60 / 1.57。
- `gram_scaled_identity` 仍不是“模型一定依赖 text”的证明；text attention mass 仍偏低：
  - medium 8-15 层平均约 2.13%。
  - sparse 8-15 层平均约 2.07%。

## medium 结果
- `sigma_mean_mse`：训练日志中 HR loss 从 0.0285 到 0.0272，仅小幅下降；诊断 effective rank = 1.80，energy top1 = 0.865。
- `gram_scaled_identity`：HR loss 从 6.14e-04 到 7.49e-05；诊断 effective rank = 9.49，energy top1 = 0.258。
- `log_participation_ratio`：loss 最低到 -1.943，但最终 -1.426；诊断 effective rank = 1.69，energy top1 = 0.878。
- 注意 medium 的 layer 9 仍是异常层，`gram_scaled_identity` 后仍接近 rank-1：
  - layer 9 top singular values: 0.9998, 0.0194, 0.0043, 0.0025。

## sparse 结果
- `sigma_mean_mse`：HR loss 从 0.0333 到 0.0326，仅小幅下降；诊断 effective rank = 1.60，energy top1 = 0.889。
- `gram_scaled_identity`：HR loss 从 7.69e-04 到 1.01e-04；诊断 effective rank = 8.97，energy top1 = 0.196。
- `log_participation_ratio`：loss 从 -1.050 到 -1.290，但诊断 effective rank = 1.57，energy top1 = 0.886。
- sparse 的 layer 9 比 medium 好一些但仍是最弱层：
  - layer 9 top singular values: 0.6847, 0.6399, 0.3489, 0.0061。

## 当前判断
- 如果目标是三种师姐 loss 里先选一个继续试，结果支持优先用 `gram_scaled_identity`。
- 如果目标是证明模型真的使用文字，还需要继续做正确文本 / 空文本 / 错误文本的敏感性验证。

## 2026-04-28 Stage-1 Text Reconstruction Conditioning 第一阶段

## 背景
- 之前的 Text-HR 是弱注入版本：每个 step 只随机选 1 个 decoder layer，把 visual memory 和 text memory concat；同一层 attention 再用于 HR 约束。
- 这意味着 text 注入层和 HR 计算层绑在一起，decoder 8-15 里只有一层真正看到 text，CNN decoder 本身也没有直接 text 分支。
- 师姐的新建议是：重建头 / decoder 需要更明确地放入 text 信息，HR loss 只能作为额外约束，不能作为唯一主线。

## 本次实现
- 新增多层 text reconstruction conditioning：
  - decoder layer 8-15 全部注入 text；
  - HR 仍然每 step 只随机选 1 层返回 post-softmax attention；
  - 当前 HR 使用 `gram_scaled_identity`，即 Frobenius-normalized Gram high-rank orthogonality loss。
- 保留旧 selected-layer `project_text_memory()` 路径，旧 Text-HR probe config 仍可继续跑。
- 新增 `project_text_memory_by_layer()`，支持 `{decoder_layer: T5 hidden}` 多层输入。
- T5 hidden state 显式使用 `hidden_states[1:]`，避免 embedding output 导致 off-by-one。
- 新增 zero-init `visual_type_embedding`，用于区分 visual memory token。
- 新增 shared scalar `text_gate`，用 `sigmoid(text_gate_logit)`，默认初始化为 0.1，对应 `logit(0.1)`。
- 所有注入层都拼接正确的 padding mask；visual mask 长度按当前 visual memory token 数动态生成，不硬编码 256。
- validation 和 `scripts/stage1/evaluate_textatlas_reconstruction.py` 已同步支持多层 text injection，避免训练和重建图 forward 不一致。
- `visual_type_embedding` 加在 cross-attention 的 visual memory token 上；decoder query 是 `latent_tokens`，没有加 visual type embedding。
- matched native 和新方法 config 的 `freeze_post_quant_conv` 均设为 `False`，对齐原生 stage-1 baseline 默认训练边界。

## 新增日志
- numeric metrics：
  - `text_recon_enabled`
  - `text_injection_layer_count`
  - `text_gate`
  - `text_memory_norm_before_gate_mean`
  - `text_memory_norm_after_gate_mean`
  - `selected_text_memory_norm`
  - `visual_memory_norm_mean`
  - `text_visual_norm_ratio`
  - `empty_text_count`
  - `text_valid_tokens_mean`
- console log 保留：
  - `text_injection_layers`
  - `text_recon_layer_pairs`
- 继续保留已有 HR 指标，包括 `text_hr_loss`、`text_hr_gram_loss_mean`、`text_hr_effective_rank_mean`、`text_hr_energy_top1_mean`、`text_hr_participation_rank_ratio_mean`。
- 新增 selected layer attention mass：
  - `text_hr_text_attention_mass_mean`
  - `text_hr_visual_attention_mass_mean`
  - 用来判断 text 虽然被 concat 进去后，decoder 是否真的 attend 到 text keys。

## 新增配置
- 新方法：
  - `configs/vq/VQ_BL256_dino_disc_text_recon_concat_hr_v1.yaml`
  - 开启 `text_conditioning`、`text_recon_conditioning`、`text_hr`。
  - 注入层和 HR layer pairs 均为 8-15。
- matched native：
  - `configs/vq/VQ_BL256_dino_disc_matched_native_v1.yaml`
  - 从新方法 config 复制，只关闭：
    - `text_conditioning.enabled`
    - `text_recon_conditioning.enabled`
    - `text_hr.enabled`
  - 其它模型、loss、freeze strategy 保持一致，便于和新方法严格同参对比。

## 后续计划
- 第一阶段先跑 matched native vs 多层 text injection + Gram HR。
- 如果重建有收益，再补 text injection only vs text injection + HR，拆分 text 注入和 HR 的贡献。
- `scripts/stage1/evaluate_textatlas_reconstruction.py` 新增 `--text-input-mode correct|empty|shuffled` 和 `--wrong-text-seed`，用于 1000-step checkpoint 后固定 16/32 张图做 sensitivity：
  - correct 明显好于 empty/shuffled：说明 text 真正参与重建；
  - 三者接近：说明 text concat 进来了但模型基本没用。
- 后续方法暂不实现，只保留 config 结构；如果误开启会直接报 `NotImplementedError`：
  - visual memory mask；
  - rec_spatial_before_cnn residual；
  - AdaLN。

## 本地检查
- 已通过 `py_compile`：
  - `tokenizer/tokenizer_image/vq/vq_vit_model.py`
  - `tokenizer/tokenizer_image/vq/blocks.py`
  - `tokenizer/tokenizer_image/vq/vq_train.py`
  - `tokenizer/tokenizer_image/vq/vq_loss.py`
  - `scripts/stage1/evaluate_textatlas_reconstruction.py`
- 已用 Ruby YAML 解析检查两个新增 config 可读。

## 2026-04-28 Stage-1 Text Reconstruction Conditioning 第二阶段：concat_memory_visual_mask

## 目标
- 实现师姐第二种方法：`visual + text concat`，同时训练时随机 mask 一小部分 cross-attention visual memory。
- 只 mask visual memory token，不 mask decoder query。
- HR 逻辑保持不变，仍然是 selected decoder layer 的 `gram_scaled_identity`。

## 实现约束
- 新增 `text_recon_conditioning.mode=concat_memory_visual_mask`。
- 强校验：
  - `mode=concat_memory` 时 `visual_memory_mask.enabled` 必须为 `False`。
  - `mode=concat_memory_visual_mask` 时 `visual_memory_mask.enabled` 必须为 `True`。
- `visual_mask_token` 只在 mask mode 创建，`concat_memory` 不新增这个参数。
- 每个 forward 只采样一次 visual mask，decoder 8-15 所有注入层复用同一个 mask。
- mask 顺序：
  - 先用 `visual_mask_token` 替换被 mask 的 visual 内容；
  - 再加 `visual_type_embedding`，保证被 mask token 仍保留 visual 类型信息。
- mask 只在 training 生效；validation / eval / reconstruction grid 中关闭。
- padding mask 不变，被 mask 的 visual token 仍是 valid key。

## 新增日志
- numeric metrics：
  - `visual_memory_mask_ratio`
  - `visual_memory_mask_actual_ratio`
- eval 时：
  - `visual_memory_mask_ratio` 保留配置值；
  - `visual_memory_mask_actual_ratio=0`。

## 新增配置
- `configs/vq/VQ_BL256_dino_disc_text_recon_concat_mask_hr_v1.yaml`
- 基于 `VQ_BL256_dino_disc_text_recon_concat_hr_v1.yaml`，只改：
  - `mode: concat_memory_visual_mask`
  - `visual_memory_mask.enabled: True`
  - `visual_memory_mask.ratio: 0.05`
  - `visual_memory_mask.mode: learned_mask_token`

## 建议 smoke
```bash
MODE=hr TAG=dense ITERS=2 \
CONFIG=configs/vq/VQ_BL256_dino_disc_text_recon_concat_mask_hr_v1.yaml \
ASCEND_RT_VISIBLE_DEVICES=0 \
bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
```

需要确认日志包含：
- `text_recon_mode=concat_memory_visual_mask`
- `visual_memory_mask_ratio=0.05`
- `visual_memory_mask_actual_ratio` 接近 0.05
- `text_hr_svd_mode=gram_scaled_identity`
- `text_hr_loss`

## 2026-04-28 当前阶段收尾检查

## 已完成方法
- 方法 1：`concat_memory`
  - decoder 8-15 多层 text injection。
  - visual/text type embedding。
  - shared scalar text gate。
  - 支持 selected-layer `gram_scaled_identity` HR。
- 方法 2：`concat_memory_visual_mask`
  - 在 `concat_memory` 基础上，仅训练时随机 mask 一小部分 cross-attention visual memory。
  - 使用 learnable `visual_mask_token`。
  - 每个 forward 采样一次 mask，8-15 注入层复用。
  - validation / eval / reconstruction grid 关闭 mask。
  - padding mask 不变，被 mask visual token 仍是 valid key。
  - 支持 selected-layer `gram_scaled_identity` HR。

## 当前代码边界
- `text_recon_conditioning.mode` 当前只允许：
  - `concat_memory`
  - `concat_memory_visual_mask`
- `matched_native` 关闭：
  - `text_conditioning`
  - `text_recon_conditioning`
  - `text_hr`
- 旧 selected-layer Text-HR config 不走 `text_recon_conditioning`，仍保留兼容。
- `visual_memory_mask.enabled` 和 mode 有强校验：
  - `concat_memory` 必须关闭 mask。
  - `concat_memory_visual_mask` 必须开启 mask。
- checkpoint missing-key 白名单只包含当前实现需要的新参数：
  - `text_projection.*`
  - `text_type_embedding`
  - `visual_type_embedding`
  - `text_gate_logit`
  - `visual_mask_token`

## 当前 config
- matched native：
  - `configs/vq/VQ_BL256_dino_disc_matched_native_v1.yaml`
- concat memory + HR：
  - `configs/vq/VQ_BL256_dino_disc_text_recon_concat_hr_v1.yaml`
- concat memory visual mask + HR：
  - `configs/vq/VQ_BL256_dino_disc_text_recon_concat_mask_hr_v1.yaml`

## 实验记录状态
- 代码中已支持记录：
  - `text_hr_loss`
  - `text_hr_gram_loss_mean`
  - `text_hr_effective_rank_mean`
  - `text_hr_energy_top1_mean`
  - `text_hr_text_attention_mass_mean`
  - `text_hr_visual_attention_mass_mean`
  - `visual_memory_mask_ratio`
  - `visual_memory_mask_actual_ratio`
- 具体 1000-step 数值和 reconstruction grid 结论需要从服务器日志与输出图补录；本地仓库当前没有这些日志文件。

## 下一阶段计划
后续再实现剩余 text reconstruction conditioning 方法，本阶段不实现：
- `concat_head`
  - 在重建头前显式融合 visual feature 和 text feature。
- `residual_head`
  - 在 `rec_spatial_before_cnn` 位置加入 gated text residual。
- `residual_pooled_layer`
  - 用 pooled text feature 经 adapter 注入中间层或重建前特征。
- `residual_cross_attn`
  - 用 text feature 通过 cross-attention residual 注入 decoder hidden。
- `adaln`
  - pooled text -> MLP -> gamma/beta，调制 LayerNorm。

下一阶段开始前需要先固定对比表：
- matched native。
- `concat_memory + HR`。
- `concat_memory_visual_mask + HR`。
- reconstruction grid。
- correct / empty / shuffled text sensitivity。

## 2026-04-28 论文调研：Glyph-ByT5 与 LongTextAR baseline

## 调研对象
- Glyph-ByT5: A Customized Text Encoder for Accurate Visual Text Rendering, arXiv:2403.09622。
- Glyph-ByT5-v2: arXiv:2406.10208，当前有 `nlpcvcode/Glyph-SDXL-v2` HF 包和官方 `AIGText/Glyph-ByT5` 仓库。
- Beyond Words: Advancing Long-Text Image Generation via Multimodal Autoregressive Models, arXiv:2503.20198，即 LongTextAR / TextBinarizer。

## 和当前 Stage-1 的关系
- Glyph-ByT5 的主要价值在 text encoder：字符级 ByT5 + glyph 对齐预训练，适合替代或对照当前 frozen T5 text conditioning。
- 但当前主线是 GigaTok B-L tokenizer decoder-only fine-tune，目标是验证 TextAtlas 重建和 HR decoder attention；直接换 Glyph-ByT5 会引入新的 text encoder 变量，不适合作为第一组 matched native vs text-recon+HR 的严格主线。
- LongTextAR 的主要贡献在 text-focused binary tokenizer 和多模态 AR 生成；它的“decoder 好”更偏 LLaMA2/Chameleon AR decoder 与 image-token 预测，不是当前 GigaTok tokenizer 的 final CNN decoder 或 transformer reconstruction decoder。

## 可借鉴点
- LongTextAR 可作为老师要求的外部 baseline：论文报告、项目页和可视化对照优先记录；当前项目页显示 Github 仍为 Coming Soon，HF paper 页面也未列出官方模型/数据链接，因此直接复现实验成本高。
- LongTextAR 的 TextBinarizer 思路支持我们的研究动机：普通 VQ tokenizer 对长文本细节是瓶颈；但替换 GigaTok quantizer/codebook 属于架构改动，暂不进入 Stage-1 pilot。
- Glyph-ByT5 可作为后续 probe：保留同一 GigaTok checkpoint、同一 TextAtlas 数据、同一训练步数，只把 `text_conditioning.encoder_name` 从 T5 换成 Glyph-ByT5/ByT5 系列，并新增投影兼容；这应放在当前 matched native vs text-recon+HR 跑通之后。

## 当前建议
- Stage-1 先跑现有两组：`VQ_BL256_dino_disc_matched_native_v1.yaml` vs `VQ_BL256_dino_disc_text_recon_concat_hr_v1.yaml`。
- 若 text sensitivity 显示 correct 明显优于 empty/shuffled，再做 Glyph-ByT5 text encoder 对照；否则先修 text injection/attention 使用率，不急于换 encoder。
- 对外汇报时可把 LongTextAR 作为“强 text-focused tokenizer/AR baseline”，但代码实现层面只借鉴其 tokenizer bottleneck 论证和重建评估口径，不在当前第一轮改 GigaTok tokenizer 架构。

## 2026-04-29 Commit 1：residual_head 文本重建模式

## 实现范围
- 只实现剩余三种方法中的第一个 mode：`text_recon_conditioning.mode=residual_head`。
- `residual_head` 默认不接 HR，配置要求 `text_hr.enabled=False`。
- `residual_head` 要求 `visual_memory_mask.enabled=False`。
- 旧 selected-layer Text-HR、`concat_memory`、`concat_memory_visual_mask`、matched native 路径保持不变。

## 方法细节
- 新增公共 helper：`masked_mean_text(text_memory, text_key_padding_mask, eps=1e-6)`。
  - `text_key_padding_mask=True` 表示 padding。
  - 只平均 valid text token；全 padding 时 denominator clamp，避免除 0。
- `residual_head` 的 text 路径固定为：
  - `T5 hidden -> text_projection -> masked_mean_text -> residual_head_mlp`
  - 不加 `text_type_embedding`。
  - 不乘 concat 路径的 `text_gate`。
- 注入位置在 `VQVitModelPlus.decode()` 内：
  - `rec_spatial = self.s1to2decoder(...)`
  - `rec_spatial = rec_spatial + residual_head_gate * text_spatial`
  - `pixel_dec = self.decoder(rec_spatial)`
- channel 来源：
  - `C = self.s1to2decoder.token_size`
  - forward assert `rec_spatial.shape[1] == self.s1to2decoder.token_size`
  - forward assert `text_spatial.shape[1] == rec_spatial.shape[1]`
  - assert 失败时不 reshape 硬凑，应检查 `rec_spatial` channel 来源。

## 新增配置
- `configs/vq/VQ_BL256_dino_disc_text_recon_residual_head_v1.yaml`
  - `text_conditioning.enabled=True`
  - `text_conditioning.text_type_embedding=False`
  - `text_recon_conditioning.mode=residual_head`
  - `text_recon_conditioning.head_text_layer=15`
  - `text_recon_conditioning.visual_memory_mask.enabled=False`
  - `text_hr.enabled=False`

## 本地检查
- `python3 -m py_compile tokenizer/tokenizer_image/vq/vq_vit_model.py tokenizer/tokenizer_image/vq/blocks.py tokenizer/tokenizer_image/vq/vq_train.py tokenizer/tokenizer_image/vq/vq_loss.py scripts/stage1/evaluate_textatlas_reconstruction.py`：通过。
- YAML parse：本机没有 PyYAML，使用 Ruby `YAML.load_file` 读取 `VQ_BL256_dino_disc_text_recon_residual_head_v1.yaml`，通过。
- eval forward smoke：本机 Python 环境没有 `torch`，未能在本机执行；需要在 ModelArts/PyTorch 环境跑。
- checkpoint load smoke：本机 Python 环境没有 `torch`，未能在本机执行；需要在 ModelArts/PyTorch 环境跑。
- 全仓 `git diff --check` 当前会被既有无关文件 `docs/original_gigatok_line_review.md` 的 EOF 空行阻断；本 commit 相关文件需单独检查。

## 服务器 smoke 结果
- GitHub commit `4c108d6` 已推送并在服务器 fast-forward 拉取。
- checkpoint load smoke：通过。
  - missing keys 只有 `text_projection.*`、`residual_head_gate`、`residual_head_mlp.*`。
  - unexpected keys 数量为 0。
- 第一次 2-step smoke 启动失败：未设置 `PYTHONPATH=$PWD`，导致 `ModuleNotFoundError: No module named 'utils'`。
- 第二次 2-step smoke 启动失败：未设置 `TORCH_HOME` / `DINOV2_REPO_DIR`，DINOv2 走 GitHub 下载并被远端断开。
- 第三次 2-step smoke 使用缓存路径后通过：
  - `TORCH_HOME=/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch`
  - `DINOV2_REPO_DIR=/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main`
  - 2 step 训练完成，无 NaN/OOM。
  - step 1 Val MSE 0.022345，PSNR 16.5083。
  - step 2 Val MSE 0.025767，PSNR 15.8894。
  - checkpoint 保存到 `outputs/text_recon_residual_head_v1/residual_head_smoke_2step_cached/checkpoints/last.pt`。
- 发现 console log 只打印旧 text recon 固定字段，`residual_head` shape/norm stats 未出现在日志行；已追加通用 extra text recon stats console 打印，需重新跑一次 2-step 确认 shape 字段可见。

## 服务器待跑命令
```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss
git pull --ff-only origin codex/text-hr-decoder

SAVE_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_recon_residual_head_v1
TRAIN_MANIFEST=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/train_materialized_manifest_v2text.jsonl
VAL_MANIFEST=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/val_materialized_manifest_v2text.jsonl
CKPT=/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt
LOCAL_T5=/home/ma-user/work/GigaTok_hr/gigatok_persist/models/google_t5-v1_1-xl
mkdir -p "$SAVE_ROOT/logs"

python - <<'PY'
import yaml
src = "configs/vq/VQ_BL256_dino_disc_text_recon_residual_head_v1.yaml"
dst = "configs/vq/_local_text_recon_residual_head_v1.yaml"
local_t5 = "/home/ma-user/work/GigaTok_hr/gigatok_persist/models/google_t5-v1_1-xl"
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["text_conditioning"]["encoder_name"] = local_t5
cfg["text_conditioning"]["local_files_only"] = True
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("wrote", dst, "mode=", cfg["text_recon_conditioning"]["mode"])
PY

ASCEND_RT_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  tokenizer/tokenizer_image/vq/vq_train.py \
  --model-config configs/vq/_local_text_recon_residual_head_v1.yaml \
  --data-path "$TRAIN_MANIFEST" \
  --json-path "$TRAIN_MANIFEST" \
  --val-json-path "$VAL_MANIFEST" \
  --save-path "$SAVE_ROOT" \
  --vq-ckpt "$CKPT" \
  --dataset textatlas_image_text \
  --device-backend npu \
  --finetune \
  --global-batch-size 24 \
  --gradient-accumulation-steps 1 \
  --max-images 300000 \
  --iterations 2 \
  --num-workers 4 \
  --val-every 1 \
  --val-max-images 32 \
  --eval-batch-size 8 \
  --log-every 1 \
  --ckpt-every 999999 \
  --save-last \
  --sub-exp-dir residual_head_smoke_2step \
  --no-wandb 2>&1 | tee "$SAVE_ROOT/logs/residual_head_smoke_2step.log"
```
