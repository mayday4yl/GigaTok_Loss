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
