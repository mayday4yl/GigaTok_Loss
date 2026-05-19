# Current State Snapshot

Last updated: 2026-05-19

Start here before reading older notes, figures, or server logs. Some files in this repository are historical planning records, and several server-side `FAILED` / `NOT_READY` markers came from earlier attempts before later fixes.

## Authoritative Current State

### Training

Both current `realworld40_5k` 250-epoch training runs completed on the new 8-card NPU server.

Main OCR-box proxy checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

GigaTok 5k baseline checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

The training wrapper state showed `status=done`, `iterations=26250`, `global_batch_size=48`, `gradient_accumulation_steps=1`, and `nproc_per_node=8`.

Important caveat: the wrapper `global.state` still records some old 100-epoch checkpoint paths. Use the two paths above as the current 250-epoch checkpoints.

### Main Holdout Eval

The overnight full NPU holdout eval initially failed because bbox/proxy JSONL rows used the 3090 path prefix while NPU manifests used `/data2/...`. This was a path-join problem, not a method change.

Path-aligned NPU eval proxy files now exist:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_balanced_200/holdout_v2_readable_balanced_200.with_proxy_conf.npu_eval.jsonl
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_medium_200/holdout_v2_readable_medium_200.with_proxy_conf.npu_eval.jsonl
```

Alignment summaries confirmed, for both sets:

- manifest rows: 200
- source proxy rows: 200
- aligned rows: 200
- manifest image exists: 200
- aligned image exists: 200
- duplicate aligned image path: 0
- missing manifest after alignment: 0
- match method: `sha256_after_prefix_map`
- only the top-level `image_path` was changed; bbox/proxy/confidence values were preserved.

10-sample smoke eval passed for both holdout sets after the path fix:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/eval_fixed_path/holdout_v2_readable_balanced_200/smoke_10
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/eval_fixed_path/holdout_v2_readable_medium_200/smoke_10
```

Current status: full 200-image holdout_v2 eval is still pending or needs explicit verification. Do not treat old failed markers as final after the path fix, but do not claim final holdout tables until the full 200-image run is present.

### readable50 Ablation

Authoritative readable50 basic eval outputs are on the 3090 server:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

Sets:

```bash
readable50_textscenes40_balanced
readable50_textscenes40_medium
readable50_textscenes40_dense_lite
```

Each set has basic reconstruction outputs:

- `eval_summary.shard00.json`
- `per_sample_metrics.shard00.jsonl`
- `holdout_qual20_grid.png`
- `reconstructions/`
- `gt_256/`

Current status: PSNR / SSIM / Text-PSNR / Text-SSIM exist. OCR CER/NED second-pass readability was not found or not verified in these output directories.

The same-protocol `gigatok_only` checkpoint for the old readable50 ablation was not found. Do not substitute pretrained GigaTok or the realworld40 5k baseline into the readable50 ablation main table. If used, label them as separate references.

## Historical Records That Can Mislead

The Text-HR diagrams and early HR/local/OCR-visual-alignment notes are historical context. They are not the final current method.

Current main method:

- OCR-box gate + proxy confidence
- DeepSeek-OCR bbox in 256x256 resize-pad coordinates
- proxy confidence from DeepSeek-OCR token logprob
- global robust p10/p90 normalization after merge
- image-side gate controlling residual text injection
- OCR CE enabled
- HR/local/OCR visual-alignment disabled

Historical or non-current material:

- `docs/stage1_text_hr_flow.svg`
- `docs/text_hr_v2_framework.svg`
- `docs/text_hr_v2_detailed_flow.svg`
- older worklog sections about HR/local/OCR visual-alignment as active method candidates
- NPU readable50 failed markers from the initial overnight supervisor branch

## What To Do Next

1. Run or verify full 200-image NPU eval for:
   - `holdout_v2_readable_balanced_200` as primary
   - `holdout_v2_readable_medium_200` as secondary
2. Add OCR CER/NED for readable50 from saved reconstructions, or clearly mark it missing.
3. Build final tables only from verified full eval outputs.
4. Keep historical figures clearly labeled if they are included in slides or handover material.
