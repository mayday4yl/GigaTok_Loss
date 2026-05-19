# Server Artifact Index

Last updated: 2026-05-19

This file explains the important server-side artifact directories in plain language. It exists because the server filenames were created across many experiment/debug runs and are not self-explanatory.

Do not rename server directories just to make them prettier. Many scripts and docs refer to the current paths. If an artifact is moved, update this file, `SERVER_PATHS.md`, and `CURRENT_STATE.md` in the same commit.

## How To Use This File

If you cannot connect to the servers, this file tells you what should exist and why it matters. If you can connect to the servers, use the verification commands below before reporting results or deleting anything.

## Active Server Roots

| Server | Role | Root |
|---|---|---|
| New 8-card NPU | main training, final checkpoints, NPU eval attempts | `/data2/duoduo_25/yl_GigaTok_Loss` |
| 8x3090 CUDA | readable50 eval, DeepSeek-OCR / CUDA-side utilities, legacy method archive | `/public/sdc/yl_GigaTok_Loss` and `/data/duoduo_25/yl_GigaTok_Loss` |

## New 8-Card NPU Artifacts

| Artifact | Path | What it is | Keep? | Verification |
|---|---|---|---|---|
| Main OCR-box proxy 250ep checkpoint | `/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt` | Final continuation checkpoint for the current main method. | Yes, critical | `test -s <path>` and `torch.load` should show model/optimizer/steps. |
| GigaTok 5k baseline 250ep checkpoint | `/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt` | Matched GigaTok-only baseline checkpoint. | Yes, critical | `test -s <path>` and `torch.load` should show model/optimizer/steps. |
| 5k selected manifests | `/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_rich_realworld_upweighted_manifests` | Train/val/holdout manifest files and selection summaries. | Yes | Check `train_5k*.jsonl`, `val_500*.jsonl`, and summaries. |
| Main text feature cache | `/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_feature_cache/realworld40_glyph_mapped` | Text feature cache used by OCR-box proxy method. | Yes | `text_features.pt` should exist; coverage is documented in preparation summaries. |
| Main OCR bbox/proxy JSONL | `/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/realworld40_5k` | Final train/val/holdout OCR bbox and proxy confidence files. | Yes | Expect plain bbox and `_with_proxy_conf.jsonl`. |
| Training logs | `/data2/duoduo_25/yl_GigaTok_Loss/logs/continue_250ep_serial_20260518_133045` | Logs/states for the 100ep to 250ep continuation. | Yes until final report | Logs should end around `step=0026250`. |
| Overnight summary | `/data2/duoduo_25/yl_GigaTok_Loss/logs/overnight_summary_20260518_141032.md` | Summary of overnight training/eval orchestration. Some entries may be stale; cross-check `CURRENT_STATE.md`. | Keep for audit | Read with `CURRENT_STATE.md`. |
| holdout_v2 prepared inputs/eval | `/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin` | holdout_v2/readable50 manifests, bbox/proxy, smoke evals, failed attempts, path-aligned eval copies. | Yes until final tables are regenerated | Smoke outputs exist; full 200 eval still needs verification/completion. |

## 8x3090 CUDA Artifacts

| Artifact | Path | What it is | Keep? | Verification |
|---|---|---|---|---|
| 3090 repo copy | `/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation` | CUDA-side repo copy used for eval/OCR utilities. | Yes | `git status` and branch should be checked before running. |
| 3090 persist root | `/public/sdc/yl_GigaTok_Loss/gigatok_persist` | Main CUDA-side artifact root. | Yes | Root for models, readable50 eval, and legacy archive. |
| DeepSeek-OCR model | `/public/sdc/yl_GigaTok_Loss/gigatok_persist/models/DeepSeek-OCR` | Local DeepSeek-OCR model for OCR/readability/proxy work. | Yes | Must load without network. |
| readable50 eval outputs | `/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval` | Three curated readable50 sets with basic reconstruction metrics, reconstructions, and grids. | Yes | Each set should contain `eval_summary.shard00.json`, `per_sample_metrics.shard00.jsonl`, and grids. |
| readable50 legacy method archive | `/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/legacy_readable50_methods_20260519/gigatok_persist_subset` | Legacy 8-method readable50 checkpoint/config archive copied for future reruns. | Yes after transfer completes | Verify expected `0002500.pt` count before relying on it. |
| legacy archive transfer log | `/public/sdc/yl_GigaTok_Loss/logs/legacy_readable50_methods_transfer_20260519/rsync.log` | Log for the legacy readable50 checkpoint/config transfer. | Keep until transfer verified | Check rsync exit status and checkpoint count. |
| companion logs | `/public/sdc/yl_GigaTok_Loss/logs/companion_downstream_20260518_225156` and `/public/sdc/yl_GigaTok_Loss/logs/repair_text_cache_readable_eval_20260519_003115` | Logs from CUDA-side downstream/eval attempts. | Keep for audit | Useful for reconstructing failed/succeeded eval attempts. |

## Quick Verification Commands

New 8-card NPU:

```bash
ssh root@139.9.157.110
docker exec -it node1new bash
ls -lh /data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
ls -lh /data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
ls /data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_rich_realworld_upweighted_manifests
```

8x3090 CUDA:

```bash
ssh ss420f.cs.cityu.edu.hk
ls /public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
ls /public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/legacy_readable50_methods_20260519/gigatok_persist_subset
```

Check whether the legacy readable50 transfer is still running:

```bash
ssh ss420f.cs.cityu.edu.hk 'cat /public/sdc/yl_GigaTok_Loss/logs/legacy_readable50_methods_transfer_20260519/rsync.pid; ps -p $(cat /public/sdc/yl_GigaTok_Loss/logs/legacy_readable50_methods_transfer_20260519/rsync.pid) -o pid,etime,cmd'
```

## Naming Warnings

- `overnight_allin` contains both useful outputs and failed/intermediate attempts. Trust `CURRENT_STATE.md` before reporting anything from it.
- `scripts/stage1/...` is a historical code directory name, not the paper-facing method name.
- Do not substitute the 5k GigaTok baseline or pretrained GigaTok into the old readable50 ablation table unless clearly labeled as a separate reference.
- Do not delete directories with `last.pt`, `0002500.pt`, `0026250.pt`, bbox/proxy JSONL, text feature cache, or readable50 reconstructions.
