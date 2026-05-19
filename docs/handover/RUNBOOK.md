# Runbook

Last updated: 2026-05-19

This file gives operational instructions for continuing the project. Paths are absolute server paths because most artifacts are not stored in Git.

## New 8-Card NPU Server

Access pattern:

```bash
ssh root@139.9.157.110
docker exec -it node1new bash
source /data/conda/bin/activate train
```

Project root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss
```

Do not write outside:

```bash
/data2/duoduo_25/yl_GigaTok_Loss
```

## Check 250 Epoch Training Status

```bash
ROOT=/data2/duoduo_25/yl_GigaTok_Loss
LOGROOT=$ROOT/logs/continue_250ep_serial_20260518_133045
cat $LOGROOT/global.state
ls $LOGROOT/*.done
tail -n 40 $LOGROOT/main_ocr_box_proxy.log
tail -n 40 $LOGROOT/gigatok_5k_baseline.log
```

Expected:

- `status=done`
- `iterations=26250`
- `main_ocr_box_proxy.done`
- `gigatok_5k_baseline.done`
- `global.done`

## Final Checkpoints

Main OCR-box proxy:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

GigaTok-only baseline:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

## Resume Training Safely

Only resume if there is a clear reason. For continuation:

- Pass `--vq-ckpt <explicit last.pt>`.
- Do not pass `--finetune`.
- Do not allow fallback to the initial VQ checkpoint.
- Keep the same module switches and data manifests.
- Increase `iterations` as a total target step count, not additional steps.

For example, 250 epochs was represented as:

```bash
iterations=26250
global_batch_size=48
gradient_accumulation_steps=1
nproc_per_node=8
```

If extending to a new total target, set `iterations` to the new total target step.

## Main Holdout Eval

Primary holdout:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/selection/holdout_v2_readable_balanced_200/holdout_v2_readable_balanced_200.jsonl
```

Secondary holdout:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/selection/holdout_v2_readable_medium_200/holdout_v2_readable_medium_200.jsonl
```

Use the path-aligned proxy files for NPU eval:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_balanced_200/holdout_v2_readable_balanced_200.with_proxy_conf.npu_eval.jsonl
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_medium_200/holdout_v2_readable_medium_200.with_proxy_conf.npu_eval.jsonl
```

The original failure was caused by proxy `image_path` values using the 3090 prefix:

```bash
/public/sdc/yl_GigaTok_Loss/...
```

while the NPU manifest uses:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/...
```

The `.npu_eval.jsonl` copies only rewrite the top-level `image_path` used for joining. They preserve bbox/proxy/confidence values.

## readable50 Ablation Eval

readable50 ablation outputs are on the 8x3090 server, not the NPU server:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

Sets:

```bash
readable50_textscenes40_balanced
readable50_textscenes40_medium
readable50_textscenes40_dense_lite
```

Each set contains:

```bash
eval_summary.shard00.json
per_sample_metrics.shard00.jsonl
holdout_qual20_grid.png
reconstructions/
gt_256/
```

The NPU readable50 eval branch is not the authoritative result. Use the 3090 result as the readable50 ablation result.

## OCR Readability Second Pass

Script:

```bash
/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation/scripts/stage1/ocr_debug/evaluate_saved_recon_ocr_readability.py
```

Use the 3090 Python environment:

```bash
/data/duoduo_25/anaconda3/envs/RiemDiff_img/bin/python
```

DeepSeek-OCR model:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/models/DeepSeek-OCR
```

This pass should operate on saved `gt_256/` and `reconstructions/` images. It should not retrain or rerun reconstruction.

## Useful Safety Checks

Check for accidental live training:

```bash
ps -eo pid,ppid,stat,etime,cmd | grep -E "torchrun|vq_train.py" | grep -v grep
```

Check `--finetune` did not appear in continuation logs:

```bash
grep -R -- "--finetune" /data2/duoduo_25/yl_GigaTok_Loss/logs/continue_250ep_serial_20260518_133045 || true
```

Check proxy normalization:

```bash
grep -m 5 "robust_p10_p90_from_mean_logprob" <proxy_jsonl>
grep -m 5 "global_all_valid_boxes_after_merge" <proxy_jsonl>
```
