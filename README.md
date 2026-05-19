# GigaTok Text-HR Stage-1 Pilot

> 2026-05-19 handover note:
> This README is historical and describes the earlier Text-HR pilot direction.
> It is **not** the source of truth for the current realworld40_5k OCR-box gate + proxy-confidence experiment.
> For current handover, start with:
>
> - `docs/handover/README_FOR_HANDOVER.md`
> - `docs/handover/CURRENT_STATE.md`
> - `docs/handover/HANDOVER.md`
>
> The current main experiment disables HR/local/OCR visual-alignment and uses OCR-box gate + DeepSeek-OCR token-logprob proxy confidence.

本仓库当前用于在 GigaTok stage-1 tokenizer 上验证 Text-HR 方法。开发主线是：

```text
codex/text-hr-decoder
```

当前工作只覆盖 stage-1 tokenizer fine-tuning：

- 目标 tokenizer：GigaTok B-L (dino disc)
- 不改 AR model
- 不改 tokenizer encoder / quantizer / codebook 的主结构
- 第一版采用 decoder-only fine-tune
- HR 约束只加在 transformer decoder 的 cross-attention 上
- 使用 post-softmax attention weights
- 每个训练 step 随机选择 1 个 `[T5 layer, decoder layer]` pair 做 SVD regularization

官方原版 GigaTok 仓库作为对比基线：

```text
https://github.com/SilentView/GigaTok.git
```

本分支相对原版的完整差异说明见：

- `docs/original_gigatok_diff_map.md`
- `docs/original_gigatok_line_review.md`
- `docs/text_hr_code_map.md`

## 当前方法

Text-HR v2 的核心做法：

1. 使用 frozen T5 encoder 提供文本 hidden states。
2. 将选中的 T5 layer feature 投影到 GigaTok decoder width。
3. 只在选中的 GigaTok transformer decoder layer 注入 text memory。
4. 返回该层 post-softmax image-to-text cross-attention。
5. 对 image query 到 valid text token 的 attention matrix 做 SVD / high-rank loss。

HR loss 当前形式：

```text
sigma = svdvals(A.float())
p = sigma / (sum(sigma) + eps)
r = len(sigma)
hr_loss = mean((p - 1/r)^2)
```

当前 v2 实现中，实际诊断还会记录 attention mass、有效 rank、rank90/rank95 等指标，用于判断 HR 分支是否真的影响 decoder text attention。

## 主要代码位置

核心方法代码：

```text
tokenizer/tokenizer_image/vq/blocks.py
tokenizer/tokenizer_image/vq/vq_vit_model.py
tokenizer/tokenizer_image/vq/vq_loss.py
tokenizer/tokenizer_image/vq/vq_train.py
```

数据接入：

```text
dataset/textatlas.py
dataset/build.py
```

评估和诊断：

```text
scripts/stage1/evaluate_textatlas_reconstruction.py
scripts/stage1/single_image_debug/make_single_image_manifests.py
scripts/stage1/single_image_debug/run_single_image_overfit.sh
scripts/stage1/single_image_debug/diagnose_single_image.py
```

代码中可搜索：

```text
Text-HR v2
```

## 配置文件

当前主要 config：

```text
configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml
configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml
configs/vq/VQ_BL256_dino_disc_text_baseline_v2.yaml
```

用途：

- `VQ_BL256_dino_disc_text_hr_v2.yaml`：当前方法，T5 text conditioning + Text-HR loss。
- `VQ_BL256_dino_disc_stage1_baseline.yaml`：原生 GigaTok tokenizer baseline，不加 T5、不加 HR。
- `VQ_BL256_dino_disc_text_baseline_v2.yaml`：text conditioning baseline，不加 HR；当前不是主对照。

主实验对比应优先使用：

```text
HR:       configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml
baseline: configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml
```

服务器如果使用本地 T5 目录，可复制上述 HR config 为本地 `_pilot_text_hr_local.yaml`，只改：

```yaml
text_conditioning:
  encoder_name: /home/ma-user/work/GigaTok_hr/gigatok_persist/models/google_t5-v1_1-xl
  local_files_only: True
```

## 数据口径

当前阶段只使用固定 TextAtlas subset：

```text
CleanTextSynth
StyledTextSynth
TextVisionBlend
TextScenesHQ
LongWordsSubset-A
```

默认 train count：

```text
TextScenesHQ: 40,000
CleanTextSynth: 50,000
StyledTextSynth: 50,000
TextVisionBlend: 50,000
LongWordsSubset-A: 50,000
```

validation：

```text
每个 subset 2,000 images
可选 source-only holdout 每个 subset 500 rows
```

预处理规则：

- 保持长宽比
- resize + pad 到 `256x256`
- 不做 center crop

服务器常用 manifest：

```text
/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/train_materialized_manifest_v2text.jsonl
/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/val_materialized_manifest_v2text.jsonl
```

## 环境要点

本地仓库可用于改代码和看 diff。长训练主要在 ModelArts / Ascend NPU 环境跑。

服务器常用环境变量：

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch
export DINOV2_REPO_DIR=/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main

export TRAIN_MANIFEST=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/train_materialized_manifest_v2text.jsonl
export VAL_MANIFEST=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/val_materialized_manifest_v2text.jsonl
export VQ_CKPT=/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt
```

## 4 卡 baseline / 4 卡 HR 训练示例

8 张卡机器上可以同时跑 baseline 和 HR。推荐用不同 `ASCEND_RT_VISIBLE_DEVICES` 和不同输出目录。

baseline：

```bash
export SAVE_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_pilot_gbs32

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 \
  tokenizer/tokenizer_image/vq/vq_train.py \
  --model-config configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml \
  --dataset textatlas_image_text \
  --data-path "$TRAIN_MANIFEST" \
  --json-path "$TRAIN_MANIFEST" \
  --val-json-path "$VAL_MANIFEST" \
  --save-path "$SAVE_ROOT" \
  --sub-exp-dir baseline_2k_gbs32_seed0 \
  --vq-ckpt "$VQ_CKPT" \
  --finetune \
  --iterations 2000 \
  --global-batch-size 32 \
  --max-images 0 \
  --num-workers 4 \
  --mixed-precision bf16 \
  --device-backend npu \
  --log-every 20 \
  --val-every 200 \
  --val-max-images 500 \
  --eval-batch-size 8 \
  --val-num-workers 4 \
  --ckpt-every 1000 \
  --save-last \
  --no-wandb
```

HR：

```bash
export SAVE_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_pilot_gbs32

ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc_per_node=4 \
  tokenizer/tokenizer_image/vq/vq_train.py \
  --model-config configs/vq/_pilot_text_hr_local.yaml \
  --dataset textatlas_image_text \
  --data-path "$TRAIN_MANIFEST" \
  --json-path "$TRAIN_MANIFEST" \
  --val-json-path "$VAL_MANIFEST" \
  --save-path "$SAVE_ROOT" \
  --sub-exp-dir hr_2k_gbs32_seed0 \
  --vq-ckpt "$VQ_CKPT" \
  --finetune \
  --iterations 2000 \
  --global-batch-size 32 \
  --max-images 0 \
  --num-workers 4 \
  --mixed-precision bf16 \
  --device-backend npu \
  --log-every 20 \
  --val-every 200 \
  --val-max-images 500 \
  --eval-batch-size 8 \
  --val-num-workers 4 \
  --ckpt-every 1000 \
  --save-last \
  --no-wandb
```

如果服务器可以直接从 Hugging Face 或本地 cache 解析 T5，也可以把 HR config 换成：

```text
configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml
```

## 单图 overfit 和逐层诊断

师姐当前要求先用单图确认结构没有 bug，并观察每层 feature 与 SVD 奇异值分布。

推荐三类图：

```text
dense:  CleanTextSynth:train:2713
medium: TextScenesHQ:train:2803
sparse: LongWordsSubset-A:train:3149
```

生成单图 manifest。这里显式传 `--target`，避免使用旧默认 dense 样本：

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

python scripts/stage1/single_image_debug/make_single_image_manifests.py \
  --source-manifest "$TRAIN_MANIFEST" \
  --output-dir /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/manifests \
  --target dense=CleanTextSynth:train:2713 \
  --target medium=TextScenesHQ:train:2803 \
  --target sparse=LongWordsSubset-A:train:3149
```

单图 HR：

```bash
CONFIG=configs/vq/_pilot_text_hr_local.yaml \
MODE=hr \
TAG=dense \
ITERS=1000 \
ASCEND_RT_VISIBLE_DEVICES=0 \
bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
```

单图原生 baseline：

```bash
MODE=baseline \
TAG=dense \
ITERS=1000 \
ASCEND_RT_VISIBLE_DEVICES=0 \
bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
```

诊断 HR checkpoint 的每层 attention / SVD / 重建：

```bash
python scripts/stage1/single_image_debug/diagnose_single_image.py \
  --manifest-jsonl /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/manifests/dense.jsonl \
  --config configs/vq/_pilot_text_hr_local.yaml \
  --ckpt /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/dense_hr_1img_1000step/checkpoints/last.pt \
  --output-dir /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/diagnostics/dense_hr_last \
  --device-backend npu \
  --mixed-precision bf16 \
  --layer-mode all_decoder \
  --text-layer-mode same \
  --seed 0
```

诊断输出：

```text
target_256.png
recon_layerXX_textXX.png
diff_layerXX_textXX.png
per_layer_metrics.csv
summary.json
singular_values_layerXX_textXX.npy
```

重点看：

- `Val MSE / Val MAE / Val PSNR`
- `rec_loss / direct_rec_loss / feature_rec_loss`
- `text_hr_loss`
- `weighted_text_hr_loss`
- `text_hr_sigma_mean`
- `text_hr_valid_text_tokens_mean`
- `text_attention_mass_mean`
- `effective_rank`
- `rank90 / rank95`

## 当前已知结论

1. 代码链路已经跑通：
   - frozen T5 加载
   - decoder text conditioning
   - post-softmax attention 返回
   - text HR loss 加入总 loss
   - NPU / HCCL 单卡与多卡训练
   - online validation
   - 单图 overfit 与逐层诊断

2. 500 step / 1000 step 级别 pilot 中，HR loss 确实进入总 loss：
   - HR run 的 train loss 比 baseline 大约多出 `weighted_text_hr_loss` 的量级。
   - 日志中可见 `text_hr_loss`、`selected_decoder_layer`、`selected_text_layer`。

3. 当前效果仍需要继续诊断：
   - baseline 与 HR 的重建图差异不明显。
   - dense 单图能 overfit 到较低 MSE / 较高 PSNR。
   - 但 text attention mass 偏低，说明 decoder 仍主要依赖 image/code 路径。
   - 当前 v2 HR 约束未明显把 image-to-text attention 推成更有用的高秩结构。

后续优先事项：

- 继续单图 dense / medium / sparse 诊断。
- 对比 pretrain、early checkpoint、last checkpoint 的每层 SVD 分布。
- 判断是否需要调 `text_hr.hr_loss_weight`、`tau`、layer pairs 或 text injection 方式。
- 在确认单图结构有效后，再做更长步数 pilot。

## 给师姐看的最小材料

如果只看核心方法，发：

```text
docs/original_gigatok_line_review.md
tokenizer/tokenizer_image/vq/blocks.py
tokenizer/tokenizer_image/vq/vq_vit_model.py
tokenizer/tokenizer_image/vq/vq_loss.py
tokenizer/tokenizer_image/vq/vq_train.py
configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml
configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml
```

如果要看单图诊断，再加：

```text
scripts/stage1/single_image_debug/README.md
scripts/stage1/single_image_debug/make_single_image_manifests.py
scripts/stage1/single_image_debug/run_single_image_overfit.sh
scripts/stage1/single_image_debug/diagnose_single_image.py
```

## 原版 GigaTok 信息

原项目论文和模型：

- Paper: `https://arxiv.org/abs/2504.08736`
- Project: `https://silentview.github.io/GigaTok/`
- Models: `https://huggingface.co/YuuTennYi/GigaTok`

本仓库仍保留原版代码结构和 MIT License。原版引用：

```bibtex
@inproceedings{gigatok,
    author    = {Xiong, Tianwei and Liew, Jun Hao and Huang, Zilong and Feng, Jiashi and Liu, Xihui},
    title     = {GigaTok: Scaling Visual Tokenizers to 3 Billion Parameters for Autoregressive Image Generation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {18770-18780}
}
```
