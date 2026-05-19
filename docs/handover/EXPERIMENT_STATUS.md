# Experiment Status

Last updated: 2026-05-19

## 1. Main OCR-Box Proxy Experiment

Status: trained to 250 epoch / `step=26250`.

Run output:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep
```

Checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

Training log:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/logs/continue_250ep_serial_20260518_133045/main_ocr_box_proxy.log
```

Latest confirmed state:

- `step=0026250`
- completed with `Done!`
- no intended formal training still running

## 2. GigaTok-Only Baseline

Status: trained to 250 epoch / `step=26250`.

Run output:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep
```

Checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

Training log:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/logs/continue_250ep_serial_20260518_133045/gigatok_5k_baseline.log
```

Latest confirmed state:

- `step=0026250`
- completed with `Done!`

## 3. Data

5k train / 500 val / 200 holdout derived from previously selected realworld40 10k/1k/500 data.

New-server manifests:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_rich_realworld_upweighted_manifests/train_5k_realworld40_text_rich.newserver.jsonl
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_rich_realworld_upweighted_manifests/val_500_realworld40_text_rich.newserver.jsonl
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_rich_realworld_upweighted_manifests/holdout_200_realworld40_text_rich.newserver.jsonl
```

Main final holdout candidates:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/selection/holdout_v2_readable_balanced_200/holdout_v2_readable_balanced_200.jsonl
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/selection/holdout_v2_readable_medium_200/holdout_v2_readable_medium_200.jsonl
```

Primary rule:

- `holdout_v2_readable_balanced_200`: primary main table.
- `holdout_v2_readable_medium_200`: secondary/supplementary.

## 4. Main Holdout Eval

Initial NPU eval failed because proxy JSONL `image_path` used the 3090 prefix while the NPU manifest used `/data2/...`.

Path-aligned proxy copies:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_balanced_200/holdout_v2_readable_balanced_200.with_proxy_conf.npu_eval.jsonl
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_medium_200/holdout_v2_readable_medium_200.with_proxy_conf.npu_eval.jsonl
```

Alignment summary:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_balanced_200/path_alignment_summary.md
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_medium_200/path_alignment_summary.md
```

Confirmed during alignment:

- manifest rows: 200
- proxy rows: 200
- aligned rows: 200
- image exists: 200/200
- duplicate image path: 0
- missing after alignment: 0
- match method: `sha256_after_prefix_map`
- only top-level `image_path` was changed

10-sample smoke eval after the path fix completed for both sets:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/eval_fixed_path/holdout_v2_readable_balanced_200/smoke_10
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/eval_fixed_path/holdout_v2_readable_medium_200/smoke_10
```

Full 200-image holdout eval still needs final completion/verification. Do not use old pre-fix failure files as the final status, but also do not report final holdout tables until the full 200-image outputs exist.

## 5. readable50 Ablation

Authoritative readable50 eval location is the 3090 server:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

Sets:

```bash
readable50_textscenes40_balanced
readable50_textscenes40_medium
readable50_textscenes40_dense_lite
```

Each set currently has:

- `eval_summary.shard00.json`
- `per_sample_metrics.shard00.jsonl`
- `holdout_qual20_grid.png`
- `reconstructions/`
- `gt_256/`

The basic reconstruction metrics are available:

- PSNR
- SSIM
- Text-PSNR
- Text-SSIM

OCR CER/NED second-pass readability is still missing or needs verification.

No readable50 OCR CER/NED table was confirmed in the checked 3090 readable50 output directories at handover time.

## 6. Current readable50 Basic Metric Trend

Balanced set:

- `ocr_box_gate_v2_conf_proxy` has the best PSNR among the available 8 methods.
- `w001_ocrvis_only` is close and has slightly better SSIM/Text-PSNR in some views.
- `ocr_box_gate_v1` is not clearly better.
- HR/local combinations do not show stable improvement.

Medium and dense_lite sets:

- `w001_ocrvis_only` is slightly ahead.
- `ocr_box_gate_v2_conf_proxy` is close behind.
- `ocrvis_hr_local` is generally weakest.

The interpretation should be revisited after OCR CER/NED is added.

## 7. Missing / Not Ready Items

- `gigatok_only` same-protocol readable50 checkpoint was not found. Do not substitute pretrained GigaTok or the 5k baseline into the readable50 ablation main table.
- OCR CER/NED for readable50 should be computed from saved reconstructions using DeepSeek-OCR readability script.
- NPU-side readable50 eval failures are not required for the main ablation; use 3090 readable50 results.
- Main holdout_v2 full eval needs completion after the path alignment fix and smoke success.
