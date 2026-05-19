# GigaTok Loss Handover

This branch is a handover branch. It is not a polished public project README. Its purpose is to help a teammate start from zero and recover the current code, server artifacts, experiment status, and safe next commands without reading the whole chat history.

Current branch:

```bash
codex/handover-gigatok-20260519
```

## Read First

Read these files in order:

```bash
docs/handover/README_FOR_HANDOVER.md
docs/handover/CURRENT_STATE.md
docs/handover/SERVER_QUICKSTART.md
docs/handover/HANDOVER.md
```

Then use these as needed:

```bash
docs/handover/RUNBOOK.md
docs/handover/EXPERIMENT_STATUS.md
docs/handover/SERVER_PATHS.md
docs/handover/CODE_CHANGES.md
docs/handover/KNOWN_ISSUES.md
docs/handover/GIT_AND_ARTIFACTS.md
docs/handover/DOCS_INDEX.md
```

`CURRENT_STATE.md` is the most important status snapshot. Some older notes and SVGs remain in the repo for history, but they are not the latest source of truth.

## Current Experiment In One Page

Active task:

- Text-rich GigaTok discrete tokenizer fine-tuning.
- Dataset: `realworld40_5k`.
- Main method: OCR-box gate + DeepSeek-OCR token-logprob proxy confidence.
- Baseline: GigaTok-only tokenizer baseline on the same 5k data.
- Current main comparison: main OCR-box proxy vs GigaTok 5k baseline.

Current data:

- train: `train_5k_realworld40_text_rich`
- val: `val_500_realworld40_text_rich`
- holdout candidates:
  - `holdout_v2_readable_balanced_200` as primary candidate
  - `holdout_v2_readable_medium_200` as secondary/supplementary candidate

Current training status:

- Main OCR-box proxy run has reached 250 epochs / `step=26250`.
- GigaTok 5k baseline has reached 250 epochs / `step=26250`.
- Final checkpoints are on the new 8-card NPU server.

Current eval status:

- 3090 readable50 basic reconstruction eval exists for three curated 50-image sets.
- OCR CER/NED and final holdout tables need verification/completion before being used in a report.
- Check `docs/handover/CURRENT_STATE.md` before quoting any number.

## Active Servers

New 8-card NPU server project root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss
```

8x3090 CUDA server project roots:

```bash
/public/sdc/yl_GigaTok_Loss
/data/duoduo_25/yl_GigaTok_Loss
```

Large artifacts are not in Git. Checkpoints, text feature cache, OCR bbox/proxy JSONL, reconstruction images, and logs live on the servers above. Exact paths are in:

```bash
docs/handover/SERVER_PATHS.md
docs/handover/CURRENT_STATE.md
docs/handover/EXPERIMENT_STATUS.md
```

## Key Code Areas

Tokenizer training/model code:

```bash
tokenizer/tokenizer_image/vq/
```

OCR-box gate:

```bash
dataset/ocr_box_gate.py
```

OCR bbox/proxy confidence/data selection/eval helpers:

```bash
scripts/stage1/ocr_debug/
scripts/stage1/eval_unified_realworld40_5k.py
```

The `scripts/stage1/...` directory name is historical. Do not use that directory name as the paper-facing method name.

Overnight/downstream orchestration helper:

```bash
tools/run_all_in_companion_downstream_worker.py
```

## Do Not Break These Rules

- Do not resume continuation training with `--finetune`.
- Do not silently fall back to the initial VQ checkpoint.
- Do not use fake/default proxy confidence.
- Do not use min/max or shard-local normalization for final proxy confidence.
- Do not enable HR/local/OCR visual-alignment for the current main OCR-box proxy experiment.
- Do not overwrite reported manifests, checkpoints, or eval outputs.
- Do not report old logs or stale failed states without checking `CURRENT_STATE.md`.

## What Git Contains

Git contains:

- code changes;
- configs and helper scripts;
- handover docs;
- runbooks and status notes.

Git does not contain:

- model checkpoints;
- datasets;
- text feature cache;
- OCR bbox/proxy JSONL artifacts;
- reconstruction images;
- large logs.

Those artifacts must stay on the active servers and be referenced by path.

## If You Only Have 10 Minutes

1. Open `docs/handover/README_FOR_HANDOVER.md`.
2. Open `docs/handover/CURRENT_STATE.md`.
3. SSH into the new 8-card NPU server and verify the two `last.pt` checkpoint paths from `CURRENT_STATE.md`.
4. SSH into the 3090 server and verify readable50 outputs from `CURRENT_STATE.md`.
5. Use `docs/handover/RUNBOOK.md` for any next command; do not invent paths from memory.
