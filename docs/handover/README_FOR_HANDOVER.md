# README For Handover

Last updated: 2026-05-19

This is the first file to read if you are taking over from zero. The repository still contains older Text-HR and ablation notes, so this page explains what is current, what is historical, and which files answer which question.

## 5-Minute Summary

Current project state:

- Task: stage-1 GigaTok tokenizer fine-tuning for text-rich image reconstruction.
- Current main method: OCR-box gate + proxy confidence.
- Baseline: GigaTok-only tokenizer baseline on the same `realworld40_5k` data.
- Current training data: `train_5k / val_500`, with holdout candidates `holdout_v2_readable_balanced_200` and `holdout_v2_readable_medium_200`.
- Current training status: both main method and baseline have finished 250 epochs / `step=26250` on the new 8-card NPU server.
- Current eval status: path-aligned holdout smoke passed, but full 200-image holdout eval still needs completion or explicit verification.
- readable50 ablation status: three 50-image curated sets have basic reconstruction metrics on the 3090 server; OCR CER/NED is still missing or unverified.

Do not use the root `README.md` or Text-HR SVGs as the current method description. They document earlier project stages.

## Read In This Order

1. `docs/handover/CURRENT_STATE.md`
   - The current source of truth.
   - Contains live-checked training/eval status and which outputs are still pending.

2. `docs/handover/HANDOVER.md`
   - One-page project scope, current experiment, and immediate next steps.

3. `docs/handover/EXPERIMENT_STATUS.md`
   - More detailed status for training, data, holdout eval, readable50, and missing items.

4. `docs/handover/SERVER_PATHS.md`
   - Where everything lives on the new 8-card NPU and 8x3090 CUDA server.

5. `docs/handover/RUNBOOK.md`
   - Operational commands for checking training, continuing safely, holdout eval, readable50 eval, and OCR readability.

6. `docs/handover/SERVER_QUICKSTART.md`
   - First 30 minutes on the servers: login, activate env, check training, run holdout eval, and run OCR readability.

7. `docs/handover/CODE_CHANGES.md`
   - Code map: which files implement text conditioning, OCR CE, OCR-box gate, proxy confidence, gradient accumulation, selection, and eval.

8. `docs/handover/KNOWN_ISSUES.md`
   - Failure modes to avoid: `--finetune` on resume, checkpoint fallback, proxy normalization drift, bbox path mismatch, and readable50 pitfalls.

9. `docs/handover/GIT_AND_ARTIFACTS.md`
   - What is in Git versus what must stay as server artifacts.

10. `docs/handover/DOCS_INDEX.md`
   - Index of older docs and diagrams. Use it only after understanding which parts are historical.

## Current Method In Plain Terms

The current main method is:

1. Use Glyph/ByT5 text features for text conditioning.
2. Use DeepSeek-OCR teacher-forcing CE.
3. Use DeepSeek-OCR text boxes to build a 16x16 image-side OCR gate.
4. Use DeepSeek-OCR token-logprob proxy confidence to modulate the OCR gate.
5. Keep the gate as image-side control over residual text injection.
6. Keep HR/local/OCR visual-alignment disabled for the current main experiment.

Important method constraints:

- OCR bbox coordinates are for the actual 256x256 resize-pad training input.
- Proxy confidence must come from DeepSeek-OCR token logprob.
- Final confidence normalization must be global robust p10/p90 after merge.
- Do not use fake/default confidence.
- Do not use shard-local or min/max normalization.

## Current Artifact Locations

New 8-card NPU project root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss
```

Main 250-epoch checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

GigaTok 5k baseline 250-epoch checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

3090 readable50 basic eval output:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

Current branch:

```bash
codex/handover-gigatok-20260519
```

## What Is Done Versus Not Done

| Area | Status | Notes |
|---|---|---|
| Main OCR-box proxy training | Done | 250 epoch / `step=26250` |
| GigaTok 5k baseline training | Done | 250 epoch / `step=26250` |
| Training checkpoint paths | Done | Use `*_continue_250ep/.../last.pt`, not stale 100-epoch paths in `global.state` |
| holdout_v2 path alignment | Done | Matched by `sha256_after_prefix_map`; only top-level `image_path` rewritten |
| holdout_v2 smoke eval | Done | `smoke_10` for balanced and medium passed after path fix |
| holdout_v2 full 200 eval | Pending / verify | Do not report final holdout tables until full 200 outputs exist |
| readable50 basic eval | Done | Three 50-image sets on 3090 |
| readable50 OCR CER/NED | Missing / verify | Needs second pass on saved reconstructions |
| readable50 `gigatok_only` old ablation checkpoint | Missing | Do not substitute pretrained or 5k baseline into main ablation table |

## Common Questions

### Can we train more from the 250-epoch checkpoints?

Yes, only if using explicit `--vq-ckpt <last.pt>`, no `--finetune`, same module switches, and `iterations` as a total target step count. See `RUNBOOK.md` and `KNOWN_ISSUES.md`.

### Which eval should be trusted for readable50?

Use the 3090 readable50 outputs. The NPU readable50 branch was an overnight supervisor attempt and is not authoritative.

### Is the final paper table ready?

No. The training is done, but final reporting still needs verified full 200-image holdout eval and OCR CER/NED if those metrics are required.

### Are the Text-HR diagrams current?

No. They are historical. The current main method is OCR-box gate + proxy confidence with HR/local/OCR visual-alignment disabled.

## Safe Next Actions

1. Verify or complete full 200-image NPU eval for balanced and medium holdout sets.
2. Add OCR CER/NED readability metrics from saved readable50 reconstructions.
3. Generate final tables only from verified full outputs.
4. Keep all large artifacts on the servers; do not commit checkpoints, caches, proxy JSONL, or reconstructions to Git.
