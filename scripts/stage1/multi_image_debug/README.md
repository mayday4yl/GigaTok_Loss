# CleanTextSynth 密集文字 100 图机制验证

本目录用于第一阶段多图机制验证：只从 `CleanTextSynth` 里选 100 张密集文字图，同一批图片同时作为 train / val / holdout。这个设置不是泛化评估，只回答一个问题：

> 在多图 overfit 场景下，Glyph-ByT5 + residual cross-attn + visual block mask 是否开始依赖具体文本内容？

当前第一阶段固定使用后层 text branch：

```text
Glyph-ByT5 text layer 0 -> decoder layers 20, 23
mode = residual_cross_attn_visual_mask
text_hr.enabled = false
```

训练脚本会基于已有 Glyph probe config 自动生成本轮专用 config，不直接修改仓库里的 YAML。

## 1. 构建 100 张同批 manifest

默认从 `CleanTextSynth` 中按文本长度选最长的 100 张，作为密集文字代理。`--same-eval-as-train` 会输出三份内容完全相同的 manifest。

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

python3 scripts/stage1/multi_image_debug/make_multi_image_manifests.py \
  --train-source-manifest /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/train_materialized_manifest_v2text.jsonl \
  --output-dir /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests \
  --name cleantextsynth_dense_100 \
  --subset CleanTextSynth \
  --per-subset 100 \
  --selection longest_text \
  --same-eval-as-train \
  --seed 0 \
  --require-existing-images
```

输出：

- `cleantextsynth_dense_100_train.jsonl`
- `cleantextsynth_dense_100_val.jsonl`
- `cleantextsynth_dense_100_holdout.jsonl`
- `cleantextsynth_dense_100_summary.json`

## 1.1 构建 OCR-readable 子集 manifest

如果要排除“256x256 下文字本身已经不可读”的混杂因素，先用
`scripts/stage1/ocr_debug/diagnose_gt_ocr_readability.py` 跑 GT OCR 可读性诊断，
再根据 `gt_ocr_predictions.jsonl` 筛出 OCR 能读准的样本。这个子集仍然是同批
train / val / holdout，只用于机制验证。

默认筛选规则：

```text
ocr_cer <= 0.05 或 ocr_ned_similarity >= 0.95
```

生成 50 张可读子集：

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

RUN_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug
OCR_DIAG=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_ocr_readability_debug/cleantextsynth_dense_100_gt256

python3 scripts/stage1/ocr_debug/make_ocr_readable_manifests.py \
  --source-manifest "$RUN_ROOT/manifests/cleantextsynth_dense_100_holdout.jsonl" \
  --ocr-predictions-jsonl "$OCR_DIAG/gt_ocr_predictions.jsonl" \
  --output-dir "$RUN_ROOT/manifests" \
  --name cleantextsynth_ocr_readable_50 \
  --max-rows 50 \
  --max-cer 0.05 \
  --min-ned 0.95 \
  --match-mode any \
  --require-existing-images
```

如果不足 50 张，先放宽到 `--max-cer 0.10 --min-ned 0.90`，或者加
`--allow-fewer` 先跑已有样本。输出：

- `cleantextsynth_ocr_readable_50_train.jsonl`
- `cleantextsynth_ocr_readable_50_val.jsonl`
- `cleantextsynth_ocr_readable_50_holdout.jsonl`
- `cleantextsynth_ocr_readable_50_summary.json`

判断口径：

- 如果 OCR-readable 子集上 correct / shuffled 仍然拉不开，说明问题更偏向 text branch 对齐/注入，而不是 256 分辨率本身不可读。
- 如果这个子集能拉开，而原 dense100 拉不开，说明密集长文本缩到 256 后确实是主要混杂因素，后续要重拿原始分辨率或改 crop 策略。

## 2. 训练 matched native baseline

这个 baseline 不加 text branch、不加 HR，用来判断 text 分支是否破坏原本 visual reconstruction。

```bash
MODE=matched_native \
MANIFEST_NAME=cleantextsynth_dense_100 \
ITERS=500 \
GLOBAL_BATCH_SIZE=20 \
VAL_MAX_IMAGES=100 \
ASCEND_RT_VISIBLE_DEVICES=0 \
bash scripts/stage1/multi_image_debug/run_multi_image_probe.sh
```

## 3. 训练 Glyph r010 / r020 / r030

`run_multi_image_probe.sh` 会自动生成：

- `text_recon_conditioning.layers=[20,23]`
- `text_recon_conditioning.layer_pairs=[[0,20],[0,23]]`
- `visual_memory_mask.strategy=block_random`
- `visual_memory_mask.fixed_pattern=true`
- `visual_memory_mask.apply_in_eval=true`

当前优先跑 r010 / r020 / r030，用来判断 0.3 mask 是否偏大。r070 仍可作为 oracle-like 强遮挡机制 probe，但不再作为第一优先级。

```bash
MODE=glyph_r010 \
MANIFEST_NAME=cleantextsynth_dense_100 \
ITERS=500 \
GLOBAL_BATCH_SIZE=20 \
VAL_MAX_IMAGES=100 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/stage1/multi_image_debug/run_multi_image_probe.sh

MODE=glyph_r020 \
MANIFEST_NAME=cleantextsynth_dense_100 \
ITERS=500 \
GLOBAL_BATCH_SIZE=20 \
VAL_MAX_IMAGES=100 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/stage1/multi_image_debug/run_multi_image_probe.sh

MODE=glyph_r030 \
MANIFEST_NAME=cleantextsynth_dense_100 \
ITERS=500 \
GLOBAL_BATCH_SIZE=20 \
VAL_MAX_IMAGES=100 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/stage1/multi_image_debug/run_multi_image_probe.sh
```

可选强遮挡：

```bash
MODE=glyph_r070 \
MANIFEST_NAME=cleantextsynth_dense_100 \
ITERS=500 \
GLOBAL_BATCH_SIZE=20 \
VAL_MAX_IMAGES=100 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/stage1/multi_image_debug/run_multi_image_probe.sh
```

训练日志重点看：

- `Val MSE / PSNR / SSIM`
- `glyph_truncated_ratio`
- `visual_memory_mask_actual_ratio`
- `masked_region_mse`
- `residual_cross_attn_proj_norm_mean`
- `residual_cross_attn_attn_entropy_norm_mean`
- `residual_cross_attn_attn_top1_mean`

## 4. 生成 nomask / evalmask config

训练用 fixed block mask；评估时需要同时看两种口径：

- `nomask`：关闭 eval mask，只看 text branch 本身是否影响重建。
- `evalmask`：保持 fixed mask，检查 masked visual 缺失时是否更依赖 text。

```bash
RUN_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug
R010_EVALMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r010_l20_23_evalmask.yaml
R020_EVALMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r020_l20_23_evalmask.yaml
R030_EVALMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r030_l20_23_evalmask.yaml
R070_EVALMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r070_l20_23_evalmask.yaml
R010_NOMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r010_l20_23_nomask.yaml
R020_NOMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r020_l20_23_nomask.yaml
R030_NOMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r030_l20_23_nomask.yaml
R070_NOMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r070_l20_23_nomask.yaml

python3 - \
  "$R010_EVALMASK_CONFIG" "$R010_NOMASK_CONFIG" \
  "$R020_EVALMASK_CONFIG" "$R020_NOMASK_CONFIG" \
  "$R030_EVALMASK_CONFIG" "$R030_NOMASK_CONFIG" \
  "$R070_EVALMASK_CONFIG" "$R070_NOMASK_CONFIG" <<'PY'
import sys
import yaml

pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
for src, dst in pairs:
    with open(src, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["text_recon_conditioning"]["visual_memory_mask"]["apply_in_eval"] = False
    with open(dst, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    print("wrote", dst)
PY
```

## 5. correct / empty / shuffled / wrong sensitivity

对同一批 holdout manifest 做 sensitivity。这里 holdout 和 train/val 是同一批图片，只用于机制验证。

```bash
RUN_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug
HOLDOUT_MANIFEST=$RUN_ROOT/manifests/cleantextsynth_dense_100_holdout.jsonl
MATCHED_CKPT=$RUN_ROOT/cleantextsynth_dense_100_matched_native_500step_seed0/checkpoints/last.pt
R010_CKPT=$RUN_ROOT/cleantextsynth_dense_100_glyph_r010_l20_23_500step_seed0/checkpoints/last.pt
R020_CKPT=$RUN_ROOT/cleantextsynth_dense_100_glyph_r020_l20_23_500step_seed0/checkpoints/last.pt
R030_CKPT=$RUN_ROOT/cleantextsynth_dense_100_glyph_r030_l20_23_500step_seed0/checkpoints/last.pt
R070_CKPT=$RUN_ROOT/cleantextsynth_dense_100_glyph_r070_l20_23_500step_seed0/checkpoints/last.pt
R010_EVALMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r010_l20_23_evalmask.yaml
R020_EVALMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r020_l20_23_evalmask.yaml
R030_EVALMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r030_l20_23_evalmask.yaml
R070_EVALMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r070_l20_23_evalmask.yaml
R010_NOMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r010_l20_23_nomask.yaml
R020_NOMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r020_l20_23_nomask.yaml
R030_NOMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r030_l20_23_nomask.yaml
R070_NOMASK_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r070_l20_23_nomask.yaml

for MASK_TAG in nomask evalmask; do
  for TEXT_MODE in correct empty shuffled wrong; do
    if [[ "$MASK_TAG" == "nomask" ]]; then
      R010_CONFIG=$R010_NOMASK_CONFIG
      R020_CONFIG=$R020_NOMASK_CONFIG
      R030_CONFIG=$R030_NOMASK_CONFIG
      R070_CONFIG=$R070_NOMASK_CONFIG
    else
      R010_CONFIG=$R010_EVALMASK_CONFIG
      R020_CONFIG=$R020_EVALMASK_CONFIG
      R030_CONFIG=$R030_EVALMASK_CONFIG
      R070_CONFIG=$R070_EVALMASK_CONFIG
    fi

    python3 scripts/stage1/evaluate_textatlas_reconstruction.py \
      --run matched_native:configs/vq/VQ_BL256_dino_disc_matched_native_v1.yaml:$MATCHED_CKPT \
      --run glyph_r010:$R010_CONFIG:$R010_CKPT \
      --run glyph_r020:$R020_CONFIG:$R020_CKPT \
      --run glyph_r030:$R030_CONFIG:$R030_CKPT \
      --run glyph_r070:$R070_CONFIG:$R070_CKPT \
      --manifest-jsonl "$HOLDOUT_MANIFEST" \
      --output-dir "$RUN_ROOT/recon_eval/${MASK_TAG}_${TEXT_MODE}" \
      --device-backend npu \
      --mixed-precision bf16 \
      --batch-size 8 \
      --max-images 0 \
      --grid-samples 40 \
      --text-input-mode "$TEXT_MODE" \
      --wrong-text-seed 0 \
      --fixed-wrong-text "THIS IS A FIXED WRONG TEXT 0123456789"
  done
done
```

判断口径：

- `correct < shuffled/wrong << empty`：100 图 overfit 下使用具体文本内容。
- `correct ≈ shuffled < wrong/empty`：对同分布错配不敏感，但能区分明显错误文本。
- `correct ≈ shuffled/wrong < empty`：主要使用“有文本”，具体内容依赖不足。
- `correct ≈ shuffled ≈ wrong ≈ empty`：text branch 仍没有形成内容依赖。
- `evalmask` 有 gap、`nomask` 没 gap：说明 mask 是触发 text 使用的关键。
- r010/r020 比 r030 更好：说明 0.3 mask 可能过强。
- r070 有 gap、低比例没 gap：说明需要更强遮挡才触发 text 使用。

## 5.1 DeepSeek-OCR 离线评估

如果要验证“重建图里的文字是否真的可读”，可以在上面的 reconstruction sensitivity 基础上额外打开 OCR backend。这个步骤只做评估，不参与训练，也不会反传梯度。

前提：服务器上已有 DeepSeek-OCR 模型目录，建议放在：

```text
/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR
```

如果模型目录不同，用 `DEEPSEEK_OCR_MODEL=/path/to/DeepSeek-OCR` 指定。DeepSeek-OCR 官方 Transformers 示例依赖 CUDA / torch2.6 / flash-attn；当前 NPU 环境不一定能直接加载，因此第一轮建议先用较小图片数做连通性测试。

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

RUN_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug
HOLDOUT_MANIFEST=$RUN_ROOT/manifests/cleantextsynth_dense_100_holdout.jsonl
MATCHED_CKPT=$RUN_ROOT/cleantextsynth_dense_100_matched_native_500step_seed0/checkpoints/last.pt
R020_CKPT=$RUN_ROOT/cleantextsynth_dense_100_glyph_r020_l20_23_500step_seed0/checkpoints/last.pt
R020_CONFIG=$RUN_ROOT/configs/cleantextsynth_dense_100_glyph_r020_l20_23_evalmask.yaml
DEEPSEEK_OCR_MODEL=/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR

python3 scripts/stage1/evaluate_textatlas_reconstruction.py \
  --run matched_native:configs/vq/VQ_BL256_dino_disc_matched_native_v1.yaml:$MATCHED_CKPT \
  --run glyph_r020:$R020_CONFIG:$R020_CKPT \
  --manifest-jsonl "$HOLDOUT_MANIFEST" \
  --output-dir "$RUN_ROOT/ocr_eval/deepseek_ocr_r020_correct_smoke" \
  --device-backend npu \
  --mixed-precision bf16 \
  --batch-size 1 \
  --max-images 4 \
  --grid-samples 4 \
  --text-input-mode correct \
  --ocr-backend deepseek_ocr \
  --deepseek-ocr-model "$DEEPSEEK_OCR_MODEL" \
  --deepseek-ocr-attn-implementation eager \
  --deepseek-ocr-dtype bf16 \
  --deepseek-ocr-base-size 1024 \
  --deepseek-ocr-image-size 640 \
  --deepseek-ocr-crop-mode \
  --deepseek-ocr-save-results \
  --ocr-jsonl "$RUN_ROOT/ocr_eval/deepseek_ocr_r020_correct_smoke/ocr_predictions.jsonl"
```

连通性通过后，再跑 `correct / empty / shuffled / wrong` 四组。OCR 指标看：

- `ocr_cer`：越低越好。
- `ocr_ned_similarity`：越高越好。
- `ocr_predictions.jsonl`：逐样本 GT text / OCR pred text / CER。

如果 DeepSeek-OCR 在当前 NPU 环境无法加载，先保留为离线评估方向；训练式 OCR loss 不要直接接入主训练。

## 6. 轻量 layer sweep

如果 ratio 消融后仍然是 `correct≈shuffled/wrong<empty`，可以固定当前较稳的 `ratio=0.2`，只扫 decoder 注入层。这个脚本会自动：

1. 为每组层生成 config；
2. 训练每组层的 Glyph text branch；
3. 跑 `correct / empty / shuffled / wrong` sensitivity；
4. 输出 `layer_sweep_summary_*.csv`。

默认是轻量设置：

```text
mask ratio = 0.2
layers:
  l20_23: 20,23
  l16_20_23: 16,20,23
  l12_16_20_23: 12,16,20,23
steps = 200
eval = evalmask only
```

运行：

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

SWEEP_ITERS=200 \
GLOBAL_BATCH_SIZE=20 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/stage1/multi_image_debug/run_layer_sweep.sh
```

如果想更完整地同时看 `nomask/evalmask`：

```bash
EVAL_MASK_TAGS="nomask evalmask" \
SWEEP_ITERS=200 \
GLOBAL_BATCH_SIZE=20 \
ASCEND_RT_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
bash scripts/stage1/multi_image_debug/run_layer_sweep.sh
```

如果想临时换层组合：

```bash
LAYER_SPECS="l20_23:20,23 l16_20_23:16,20,23 l16_23dense:16,17,18,19,20,21,22,23" \
bash scripts/stage1/multi_image_debug/run_layer_sweep.sh
```

只检查 config 生成、不训练：

```bash
DRY_RUN=1 bash scripts/stage1/multi_image_debug/run_layer_sweep.sh
```

判断口径仍然是：

- `correct` 比 `shuffled/wrong` 明显好：层数改善了具体文本内容利用。
- 只比 `empty` 好：仍然主要使用“有 text feature”信号。
- MSE 变好但 sensitivity 不变：层数只改善了整体重建，没有解决 text-content alignment。
