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
