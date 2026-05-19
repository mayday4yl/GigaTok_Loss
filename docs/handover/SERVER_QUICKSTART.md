# Server Quickstart

Last updated: 2026-05-19

This file is for the first 30 minutes on the servers. It shows how to log in, activate the right environment, check current results, and run the remaining evaluation code without guessing paths.

## 1. New 8-Card NPU Server

Login:

```bash
ssh root@139.9.157.110
docker exec -it node1new bash
source /data/conda/bin/activate train
```

Project setup:

```bash
ROOT=/data2/duoduo_25/yl_GigaTok_Loss
REPO=$ROOT/repo/GigaTok_Loss_two_ablation
PERSIST=$ROOT/gigatok_persist
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$ROOT/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$PERSIST/cache/torch
```

Check that the 250-epoch training runs are done:

```bash
LOGROOT=$ROOT/logs/continue_250ep_serial_20260518_133045
cat "$LOGROOT/global.state"
ls "$LOGROOT"/*.done
tail -n 20 "$LOGROOT/main_ocr_box_proxy.log"
tail -n 20 "$LOGROOT/gigatok_5k_baseline.log"
```

Expected:

- `status=done`
- `main_ocr_box_proxy.done`
- `gigatok_5k_baseline.done`
- `global.done`
- both logs end at `step=0026250` and `Done!`

Final checkpoint paths:

```bash
MAIN_CKPT=$PERSIST/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
GIGA_CKPT=$PERSIST/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

Configs and text cache:

```bash
MAIN_CONFIG=$PERSIST/outputs/ocr_box_gate/realworld40_5k/configs/ocr_box_gate_v2_conf_proxy.realworld40_5k.newserver.yaml
GIGA_CONFIG=$PERSIST/outputs/original_gigatok_baseline_realworld40_5k/configs/orig_gigatok_realworld40_5k.yaml
TEXT_CACHE=$PERSIST/outputs/text_feature_cache/realworld40_glyph_mapped/text_features.pt
```

Quick file check:

```bash
for f in "$MAIN_CKPT" "$GIGA_CKPT" "$MAIN_CONFIG" "$GIGA_CONFIG" "$TEXT_CACHE"; do
  test -f "$f" && echo "OK $f" || echo "MISSING $f"
done
```

## 2. Run Main Holdout Eval On NPU

Primary holdout is `balanced_200`. Secondary holdout is `medium_200`.

Use the path-aligned `.npu_eval.jsonl` files. Do not use the original 3090-prefix proxy JSONL for NPU eval.

Balanced variables:

```bash
SET=holdout_v2_readable_balanced_200
MANIFEST=$PERSIST/outputs/overnight_allin/selection/$SET/$SET.jsonl
BBOX=$PERSIST/outputs/overnight_allin/prepared_inputs/$SET/$SET.with_proxy_conf.npu_eval.jsonl
OUT=$PERSIST/outputs/overnight_allin/eval_fixed_path/$SET/full_200_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
```

Medium variables:

```bash
SET=holdout_v2_readable_medium_200
MANIFEST=$PERSIST/outputs/overnight_allin/selection/$SET/$SET.jsonl
BBOX=$PERSIST/outputs/overnight_allin/prepared_inputs/$SET/$SET.with_proxy_conf.npu_eval.jsonl
OUT=$PERSIST/outputs/overnight_allin/eval_fixed_path/$SET/full_200_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
```

Smoke first:

```bash
/data/conda/envs/train/bin/python scripts/stage1/eval_unified_realworld40_5k.py \
  --manifest "$MANIFEST" \
  --bbox-jsonl "$BBOX" \
  --output-dir "$OUT/smoke_10" \
  --run main_last:"$MAIN_CONFIG":"$MAIN_CKPT":"$TEXT_CACHE" \
  --run baseline_last:"$GIGA_CONFIG":"$GIGA_CKPT" \
  --mode smoke \
  --max-images 10 \
  --batch-size 4 \
  --device-backend npu \
  --device-id 0 \
  --mixed-precision bf16
```

Full 200 run:

```bash
nohup /data/conda/envs/train/bin/python scripts/stage1/eval_unified_realworld40_5k.py \
  --manifest "$MANIFEST" \
  --bbox-jsonl "$BBOX" \
  --output-dir "$OUT/full_200" \
  --run main_last:"$MAIN_CONFIG":"$MAIN_CKPT":"$TEXT_CACHE" \
  --run baseline_last:"$GIGA_CONFIG":"$GIGA_CKPT" \
  --mode holdout \
  --max-images 0 \
  --batch-size 4 \
  --device-backend npu \
  --device-id 0 \
  --mixed-precision bf16 \
  > "$OUT/full_200.log" 2>&1 < /dev/null &
echo $! > "$OUT/full_200.pid"
```

Monitor:

```bash
tail -f "$OUT/full_200.log"
ls "$OUT/full_200"
```

Expected output files include:

```bash
eval_summary.shard00.json
per_sample_metrics.shard00.jsonl
holdout_qual20_grid.png
reconstructions/
gt_256/
```

## 3. 8x3090 CUDA Server

Login:

```bash
ssh ss420f.cs.cityu.edu.hk
```

Project setup:

```bash
PERSIST=/public/sdc/yl_GigaTok_Loss/gigatok_persist
REPO=/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation
PY=/data/duoduo_25/anaconda3/envs/RiemDiff_img/bin/python
cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export HF_HOME=/public/sdc/yl_GigaTok_Loss/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export TORCH_HOME=$PERSIST/cache/torch
export TOKENIZERS_PARALLELISM=false
```

Find readable50 basic eval outputs:

```bash
READABLE_ROOT=$PERSIST/outputs/overnight_allin/readable50_eval
find "$READABLE_ROOT" -maxdepth 2 -type f \
  \( -name 'eval_summary.shard00.json' -o -name 'per_sample_metrics.shard00.jsonl' -o -name 'holdout_qual20_grid.png' \) \
  | sort
```

## 4. Run OCR CER/NED On Saved Reconstructions

Use this only after reconstruction eval outputs already exist. This does not retrain.

Example for one readable50 set:

```bash
SET=readable50_textscenes40_balanced
EVAL_DIR=$PERSIST/outputs/overnight_allin/readable50_eval/$SET
MANIFEST=$PERSIST/outputs/overnight_allin/selection/$SET/$SET.jsonl
OUT=$EVAL_DIR/ocr_readability_$(date +%Y%m%d_%H%M%S)
DEEPSEEK=$PERSIST/models/DeepSeek-OCR
mkdir -p "$OUT"

nohup "$PY" scripts/stage1/ocr_debug/evaluate_saved_recon_ocr_readability.py \
  --eval-dir "$EVAL_DIR" \
  --manifest-jsonl "$MANIFEST" \
  --per-sample-jsonl "$EVAL_DIR/per_sample_metrics.shard00.jsonl" \
  --output-dir "$OUT" \
  --runs w001 w001_ocrvis_only w001_hr_only w001_local_only w001_hr_local ocrvis_hr_local ocr_box_gate_v1 ocr_box_gate_v2_conf_proxy \
  --include-gt \
  --device-backend cuda \
  --device-id 0 \
  --ocr-backend deepseek_ocr \
  --deepseek-ocr-model "$DEEPSEEK" \
  --deepseek-ocr-attn-implementation eager \
  --deepseek-ocr-dtype bf16 \
  --deepseek-ocr-max-new-tokens 1024 \
  > "$OUT/ocr_readability.log" 2>&1 < /dev/null &
echo $! > "$OUT/ocr_readability.pid"
```

Monitor:

```bash
tail -f "$OUT/ocr_readability.log"
find "$OUT" -maxdepth 2 -type f | sort
```

If the manifest path does not exist for a readable50 set, stop and inspect the selection directory instead of guessing.

## 5. Safety Rules

- Do not run continuation training with `--finetune`.
- Do not use stale 100-epoch checkpoint paths from `global.state`; use the `*_continue_250ep/.../last.pt` paths.
- Do not write outside project roots.
- Do not delete shared-server files.
- Do not use `rsync --delete`.
- For NPU eval, use `.npu_eval.jsonl` bbox/proxy files.
- For readable50, use the 3090 outputs as authoritative.
