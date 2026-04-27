# 给师姐看的代码修改行号说明

这份文档按“相对于官方原版 GigaTok”的口径整理当前分支的主要修改，并标注当前分支中的行号范围，方便直接打开文件审阅。

## 对比口径

- 原版 GigaTok remote: `upstream=https://github.com/SilentView/GigaTok.git`
- 原版基线: `upstream/master`
- merge-base: `baf3de042034318aa762c28db4d14231efce8b0c`
- 当前分支: `codex/text-hr-decoder`

完整 diff 可用：

```bash
git diff --stat upstream/master...HEAD
git diff --name-status upstream/master...HEAD
```

行号说明：

- 下表行号是**当前分支文件中的行号**。
- 对新增文件，行号表示新增文件内相关代码范围。
- 对原版已有文件，行号表示相对原版 GigaTok 的主要修改区域。
- 核心方法相关位置也可以直接搜索 `Text-HR v2`。

## 一、最核心方法代码

### 1. Decoder cross-attention 返回 post-softmax attention

文件：`tokenizer/tokenizer_image/vq/blocks.py`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 638-685 | `TransformerDecoderLayer.forward_post(...)` 增加 `return_cross_attn_weights`，调用 `nn.MultiheadAttention` 时设置 `need_weights=return_cross_attn_weights`、`average_attn_weights=False`，返回 `cross_attn_weights` | 让选中 decoder 层能返回 post-softmax cross-attention 矩阵 `[B,H,Q,K]` |
| 688-745 | `TransformerDecoderLayer.forward_pre(...)` 做同样的 attention 返回逻辑 | 兼容 pre-norm decoder block |

给师姐看重点：第 `673-678` 行和第 `740-745` 行，那里是真正从 PyTorch attention 取权重的位置。

### 2. VQ tokenizer 接入 frozen T5 text feature

文件：`tokenizer/tokenizer_image/vq/vq_vit_model.py`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 357-389 | 新增 `configure_text_conditioning(...)` | 建立 T5 hidden state 到 GigaTok decoder width 的投影层，当前默认 `Linear + LayerNorm` |
| 391-404 | 新增 `project_text_memory(...)` | 将选中 T5 layer feature `[B,T,d_t5]` 投影为 decoder memory `[B,T,d_dec]` |
| 481-528 | 修改 `decode(...)`，支持 `selected_decoder_layer`、`decoder_text_features`、`decoder_text_key_padding_mask`，并返回该层 cross-attention | 只在选中的 decoder layer 注入 text memory，不影响原 image-only 路径 |
| 535-625 | 修改 `forward(...)` 参数和返回值，透传 text feature 与 attention weights | 训练阶段把 attention weights 交给 loss 计算 HR |

给师姐看重点：第 `357-404` 行是 text projection；第 `481-528` 行是 text 注入 decoder；第 `617-624` 行是返回 attention 给训练 loss。

### 3. Image-to-text attention 的 SVD / HR loss

文件：`tokenizer/tokenizer_image/vq/vq_loss.py`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 136-148 | `high_rank_attention_loss(...)` | 旧版 decoder attention HR loss，保留兼容 |
| 151-232 | 新增 `high_rank_image_text_attention_loss(...)` | 当前 Text-HR v2 核心：从 `[B,H,Q,K]` attention 中裁出 text slice，mask padding token，float32 SVD |
| 428-435 | `VQLoss.forward(...)` 增加 HR 相关入参 | 训练脚本把 attention、text mask、layer id、权重传进 loss |
| 558-586 | 计算 `hr_loss_term` 和 `text_hr_loss_term`，并加入 generator 总 loss | HR 真正进入优化目标的位置 |
| 640-685 | 日志和 wandb/cache 字段增加 `text_hr_loss`、`weighted_text_hr_loss`、`sigma_mean`、有效 token 数等 | 单独观察 HR 是否生效 |

给师姐看重点：第 `151-232` 行是 loss 定义；第 `581-586` 行是加入总 loss。

## 二、训练主线接入

文件：`tokenizer/tokenizer_image/vq/vq_train.py`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 436-456 | 读取 `text_conditioning` 和 `text_hr` config，检查开关合法性 | 让方法可通过 config 开关打开/关闭 |
| 504-557 | TextAtlas / text-conditioned 路径使用 resize+pad，validation 用 TextAtlas JSONL dataset | 保证 baseline 和 HR 对同一预处理口径比较 |
| 663-690 | 加载 frozen T5 tokenizer / T5EncoderModel | 提供文本 hidden states，T5 本身不训练 |
| 692-700 | 调用 `vq_model.configure_text_conditioning(...)` | 给 tokenizer 增加 text projection 参数 |
| 812-824 | 旧 checkpoint 兼容新增 text 参数 | 原版 GigaTok checkpoint 没有 `text_projection` / `text_type_embedding`，这里允许随机初始化 |
| 983-995 | 根据 config 构建 `[t5_layer, decoder_layer]` pairs | 明确 T5 层和 decoder 层怎么对应 |
| 1124-1169 | 每个 step 随机选一组 layer pair，tokenize 文本，跑 frozen T5，取选中 hidden state | 训练时得到要注入 decoder 的 text feature |
| 1219-1278 | 前向时把 text feature 传给 VQ model，并把返回的 attention weights / text mask 传给 `VQLoss` | Text-HR 进入训练主循环 |

给师姐看重点：第 `1124-1169` 行是每步选层和取 T5 feature；第 `1260-1278` 行是把 attention 交给 loss。

## 三、配置文件

### 1. 我们的方法配置

文件：`configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 42-63 | stage-1 finetune 基本训练配置，冻结 encoder / quantizer / codebook / post-quant conv | decoder-only finetune |
| 65-79 | `text_conditioning` | 开启 frozen T5，设置 `max_length=128`、projection、type embedding |
| 81-105 | `text_hr` | 开启 Text-HR，设置 layer pairs 8-15、post-softmax、image-to-text、mask padding、SVD 模式 |

给师姐看重点：第 `65-105` 行就是方法开关和超参。

### 2. 原生 GigaTok baseline 配置

文件：`configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 46-69 | 原生 tokenizer baseline 训练配置，无 `text_conditioning`、无 `text_hr` | 用来和 HR 做主对照 |
| 71-91 | 原生 VQ loss 配置 | 保持 GigaTok stage-1 原损失 |

给师姐看重点：第 `68-69` 行说明 baseline 不加 text / HR。

## 四、TextAtlas 数据接入

### 1. 新增 TextAtlas JSONL dataset

文件：`dataset/textatlas.py`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 9-22 | 文本合法性规则，例如过滤 raw prompt 形式 | 防止把 prompt 当作渲染文字训练 |
| 25-60 | `TextAtlasImageTextDataset.__init__` 读取 materialized JSONL，检查 `image_path`、`text`、状态字段 | 构造 `(image, text)` 样本 |
| 65-75 | `__getitem__` 返回 image tensor 对应的 text string | 训练时给 T5 encoder 使用 |
| 78-86 | `build_textatlas_image_text(...)` | 接入统一 dataset builder |

文件：`dataset/build.py`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 6 | import `build_textatlas_image_text` | 注册新数据集 |
| 27-28 | `args.dataset == "textatlas_image_text"` 时返回 TextAtlas dataset | 训练命令可用 `--dataset textatlas_image_text` |

## 五、单图过拟合与逐层诊断

### 1. 单图 HR / baseline 运行脚本

文件：`scripts/stage1/single_image_debug/run_single_image_overfit.sh`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 14-42 | 定义 `SAVE_ROOT`、`TAG`、`MODE`、HR config、baseline config、ckpt、日志频率等 | 通过 `MODE=hr` 或 `MODE=baseline` 控制实验 |
| 69-93 | 调用 `torchrun tokenizer/tokenizer_image/vq/vq_train.py` | 单图 overfit 入口 |

给师姐看重点：`MODE=baseline` 对应原生 GigaTok baseline；`MODE=hr` 对应 text + HR。

### 2. 单图逐层诊断脚本

文件：`scripts/stage1/single_image_debug/diagnose_single_image.py`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 39-57 | 命令行参数：manifest、config、ckpt、输出目录、layer mode、seed 等 | 控制诊断对象 |
| 81-108 | tensor / scalar 统计工具 | 统计特征均值、方差、RMS 等 |
| 124-196 | `attention_svd_metrics(...)` | 对 image-to-text attention 做 SVD，计算 `sigma_top1_ratio`、`effective_rank`、`rank90/rank95` 等 |
| 199-211 | 保存重建图和 raw diff 图 | 可视化目标图、重建图、差异图 |
| 214-231 | 构建诊断 layer pairs，支持 `all_decoder` 或 config pairs | 可看 24 层或只看训练层 |
| 234-246 | hook decoder layer feature | 看每层 decoder feature 统计 |
| 261-413 | 主流程：加载模型、T5、encode image、逐层 decode、保存重建图和 metrics csv | 输出师姐要求的每层特征、SVD、重建表现 |

给师姐看重点：第 `140-196` 行是 SVD 指标；第 `331-382` 行是逐层 decode 并收集 attention / feature / recon 指标。

### 3. 通用 TextAtlas 重建评估

文件：`scripts/stage1/evaluate_textatlas_reconstruction.py`

| 当前行号 | 修改内容 | 作用 |
|---|---|---|
| 212-243 | 加载 text-conditioned T5 tokenizer / encoder | 支持评估 text-conditioned checkpoint |
| 246-337 | 构建 text layer pairs，加载 tokenizer checkpoint，兼容新增 text 参数 | 评估时复现训练结构 |
| 340-392 | `reconstruct_batch(...)` 支持 image-only 和 text-conditioned 两条路径 | 导出重建图和指标 |
| 395-406 | MSE / MAE / PSNR / SSIM 计算 | 重建指标 |

## 六、NPU / validation / checkpoint 兼容等工程改动

这些不是方法核心，但相对原版 GigaTok 也是重要修改。

| 文件 | 当前行号 | 修改内容 | 作用 |
|---|---|---|---|
| `tokenizer/tokenizer_image/vq/vq_train.py` | 179-276 | online reconstruction validation 指标计算 | 训练中输出 Val MSE / MAE / PSNR / SSIM |
| `tokenizer/tokenizer_image/vq/vq_train.py` | 534-575 | validation dataset 构建逻辑 | 支持 TextAtlas JSONL validation |
| `tokenizer/tokenizer_image/vq/vq_train.py` | 1520-1586 | 新增 CLI 参数，如 `--device-backend`、`--val-json-path`、`--no-wandb` 等 | 服务器实验可控 |
| `utils/distributed.py` | 6-17, 34-72 | 增加 `cuda` / `npu` backend 分支，并在 NPU 下使用 `hccl` | Ascend 服务器运行 |
| `utils/model_init.py` | 53-91, 101-114 | 支持 `DINOV2_REPO_DIR` 本地加载 DINOv2，并固定 DINOv2-B embed dim 检查 | 避免在线下载失败，保持 frozen DINOv2-B |
| `utils/model_init.py` | 181-213 | `custom_load(...)` 增加 `ignore_missing_keys` | 兼容旧 GigaTok checkpoint 中没有新增 text projection 参数 |
| `utils/resume_log.py` | 25-39, 58, 113-128, 134-152 | wandb import 失败时延迟报错，配合 `--no-wandb` 继续训练 | 服务器 wandb / protobuf 环境不稳定时不阻塞训练 |

## 七、给师姐看的最小代码包

如果她只想看我们方法主体，发这些即可：

```text
docs/original_gigatok_line_review.md
docs/original_gigatok_diff_map.md
configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml
configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml
tokenizer/tokenizer_image/vq/blocks.py
tokenizer/tokenizer_image/vq/vq_vit_model.py
tokenizer/tokenizer_image/vq/vq_loss.py
tokenizer/tokenizer_image/vq/vq_train.py
dataset/textatlas.py
scripts/stage1/single_image_debug/diagnose_single_image.py
```

## 八、一句话说明

相对原版 GigaTok，当前主改动是：

1. 在 stage-1 tokenizer decoder 的指定 cross-attention 层接入 frozen T5 文本特征。
2. 返回该层 post-softmax image-to-text attention，并对其做 SVD / high-rank 约束。
3. 增加 TextAtlas 数据读取、NPU 训练、online validation、单图 overfit 和逐层诊断工具，用来验证该方法是否有效。
