# GigaTok Loss Handover

This branch is a handover branch. It is not a polished public project README. Its purpose is to help a teammate start from zero and recover the current code, server artifacts, experiment status, and safe next commands without reading the whole chat history.

Current branch:

```bash
codex/handover-gigatok-20260519
```

## Handover TL;DR

If you are taking over the experiments, start from this README and treat the two 8-card servers as the active workspace:

- **New 8-card NPU server**: training checkpoints, 5k data preparation artifacts, OCR-box gate inputs, and current main/baseline continuation outputs.
- **8x3090 CUDA server**: readable50 / legacy ablation archive, CUDA-side OCR/eval utilities, readable50 reconstruction outputs, and related logs.
- For future work, use only these two active 8-card servers unless a new server is explicitly documented.

The current main experiment is **OCR-box gate + DeepSeek-OCR token-logprob proxy confidence** on `realworld40_5k`. The matched baseline is **GigaTok-only** on the same train/val/holdout setup. Both have completed 250 epochs / `step=26250` on the new 8-card NPU server.

Large artifacts are intentionally not stored in Git. This repository tells you which code to use and where the server artifacts live.

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
docs/handover/CONVERSATION_SUMMARY.md
docs/handover/AGENT_MEMORY.md
docs/handover/SERVER_PATHS.md
docs/handover/SERVER_ARTIFACT_INDEX.md
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

Naming glossary:

| Name | Meaning |
|---|---|
| `realworld40` | Text-rich dataset variant where real-world / mixed image sources are upweighted to about 40%. In this project, the real-world part mainly means TextScenesHQ + TextVisionBlend. |
| `realworld40_5k` | The current 5k training setup: `train_5k`, `val_500`, and related holdout/eval artifacts derived from the realworld40 selection. |
| `train_5k_realworld40_text_rich` | 5,000-image training manifest for the current experiments. |
| `val_500_realworld40_text_rich` | 500-image validation manifest used for training-time validation and sanity checks. |
| `holdout_200_realworld40_text_rich` | Earlier 200-image holdout split derived from the 5k setup. Later `holdout_v2` candidates were prepared for final reporting. |
| `holdout_v2_readable_balanced_200` | Primary 200-image holdout candidate for main final tables. It is intended to be readable and source-balanced. |
| `holdout_v2_readable_medium_200` | Secondary / supplementary 200-image holdout candidate, biased toward medium text coverage. |
| `readable50_textscenes40_*` | Curated 50-image probe sets for readable50 ablation/effect checks. These are not the main unbiased holdout. |
| `realworld40_glyph_mapped` | Text feature cache directory for the current realworld40 setup. |
| `realworld40_5k` under `outputs/ocr_box_gate/` | OCR bbox and proxy-confidence artifacts used by the current main OCR-box gate training/eval. |

Current training status:

- Main OCR-box proxy run has reached 250 epochs / `step=26250`.
- GigaTok 5k baseline has reached 250 epochs / `step=26250`.
- Final checkpoints are on the new 8-card NPU server.

Current eval status:

- 3090 readable50 basic reconstruction eval exists for three curated 50-image sets.
- OCR CER/NED and final holdout tables need verification/completion before being used in a report.
- Check `docs/handover/CURRENT_STATE.md` before quoting any number.

What is still not final:

- Full 200-image `holdout_v2` tables should be verified or regenerated from the fixed-path NPU eval inputs before reporting.
- readable50 OCR CER/NED was not verified in the latest readable50 output directories.
- Some server directories include failed/intermediate attempts. Use the paths below and `docs/handover/CURRENT_STATE.md` instead of browsing by directory name alone.

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

Large artifacts are not in Git. Checkpoints, text feature cache, OCR bbox/proxy JSONL, reconstruction images, and logs live on the servers above. Exact paths and plain-language explanations are in:

```bash
docs/handover/SERVER_PATHS.md
docs/handover/SERVER_ARTIFACT_INDEX.md
docs/handover/CURRENT_STATE.md
docs/handover/EXPERIMENT_STATUS.md
```

## Critical Server Artifacts

These are the minimum artifacts a teammate needs to find before continuing the experiments.

### New 8-card NPU server

Project root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss
```

Main OCR-box proxy 250-epoch checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

GigaTok 5k baseline 250-epoch checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

5k selected manifests:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_rich_realworld_upweighted_manifests
```

Text feature cache:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_feature_cache/realworld40_glyph_mapped
```

OCR bbox / proxy confidence JSONL for the 5k training setup:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/realworld40_5k
```

holdout_v2 prepared inputs and fixed-path smoke eval attempts:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin
```

Training logs for the 100-epoch to 250-epoch continuation:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/logs/continue_250ep_serial_20260518_133045
```

### 8x3090 CUDA server

Main CUDA artifact root:

```bash
/public/sdc/yl_GigaTok_Loss
```

CUDA-side repo copy:

```bash
/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation
```

CUDA-side persist root:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist
```

readable50 eval outputs:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

legacy readable50 method checkpoint/config archive:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/legacy_readable50_methods_20260519/gigatok_persist_subset
```

If the legacy archive is needed, verify the transfer status before relying on it:

```bash
cat /public/sdc/yl_GigaTok_Loss/logs/legacy_readable50_methods_transfer_20260519/rsync.log
```

## Where Things Are

Use this as the first artifact map. If a path looks stale, verify it against `docs/handover/CURRENT_STATE.md` before using it.

| Item | Where to look |
|---|---|
| Current Git code | This branch, `codex/handover-gigatok-20260519` |
| New NPU repo copy | `/data2/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation` |
| New NPU persist root | `/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist` |
| 3090 repo copy | `/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation` |
| 3090 persist root | `/public/sdc/yl_GigaTok_Loss/gigatok_persist` |
| Main OCR-box proxy 250-epoch checkpoint | See exact path in `docs/handover/CURRENT_STATE.md` |
| GigaTok 5k baseline 250-epoch checkpoint | See exact path in `docs/handover/CURRENT_STATE.md` |
| 5k train/val manifests | New NPU persist root, `outputs/text_rich_realworld_upweighted_manifests/` |
| 5k text feature cache | New NPU persist root, `outputs/text_feature_cache/realworld40_glyph_mapped/` |
| 5k OCR bbox/proxy JSONL | New NPU persist root, `outputs/ocr_box_gate/realworld40_5k/` |
| holdout_v2 prepared inputs/eval | New NPU persist root, `outputs/overnight_allin/` |
| readable50 eval outputs | 3090 persist root, `outputs/overnight_allin/readable50_eval/` |
| readable50 legacy method checkpoint archive | 3090 persist root, `outputs/legacy_readable50_methods_20260519/` after transfer completes |
| Training/eval logs | New NPU `/data2/duoduo_25/yl_GigaTok_Loss/logs`, 3090 `/public/sdc/yl_GigaTok_Loss/logs` |

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

Minimal verification commands:

```bash
# New 8-card NPU server
ssh root@139.9.157.110
docker exec -it node1new bash
ls -lh /data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
ls -lh /data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt

# 8x3090 CUDA server
ssh ss420f.cs.cityu.edu.hk
ls /public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
ls /public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/legacy_readable50_methods_20260519/gigatok_persist_subset
```

## How To Read This Repository

For handover, read by purpose rather than by folder order:

1. **Current status:** `docs/handover/CURRENT_STATE.md`
2. **Server paths:** `docs/handover/SERVER_PATHS.md`
3. **Server artifact meanings:** `docs/handover/SERVER_ARTIFACT_INDEX.md`
4. **How to run/check things:** `docs/handover/RUNBOOK.md` and `docs/handover/SERVER_QUICKSTART.md`
5. **What code changed:** `docs/handover/CODE_CHANGES.md`
6. **Known traps:** `docs/handover/KNOWN_ISSUES.md`
7. **Historical notes:** `docs/handover/DOCS_INDEX.md`

Do not start by reading old experiment notes or SVGs. They are useful only after the current state is understood.
