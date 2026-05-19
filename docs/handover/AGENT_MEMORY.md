# AGENT_MEMORY.md

Last updated: 2026-05-19

This file is an operational memory for the next coding/research agent. It is not a paper introduction and not a full chat transcript. It is meant to let a new agent understand the current experiment state, server layout, critical files, safe constraints, and next actions without reading the original conversation.

## Agent Startup Protocol

If you are a new coding/research agent receiving this file, treat it as the handover prompt. Do not ask the user to restate the old chat history. Do not require an additional prompt to understand the project.

Your first turn should be a read-only takeover pass:

1. Read this file completely.
2. Read the repository README and the handover docs listed in Section 0 if they are available.
3. Do not start training.
4. Do not start evaluation.
5. Do not modify files.
6. Do not delete files.
7. Do not move or rename server artifacts.
8. Do not trust stale `FAILED`, `NOT_READY`, or intermediate markers until cross-checking against `CURRENT_STATE.md` and the actual server paths.

After reading, produce a concise takeover summary with:

- current main method;
- current baseline;
- current dataset and naming;
- active server roles;
- completed checkpoints;
- trusted outputs;
- unverified outputs;
- top next actions;
- hard constraints / forbidden actions.

If server access is available, perform only read-only verification first:

- check that the two 250ep checkpoints exist;
- check `train_5k` and `val_500` manifest row counts;
- check readable50 output directories on the 3090 server;
- check whether the legacy readable50 archive transfer has completed.

If server access is not available, explicitly state:

```text
server artifacts not verified in this session
```

Then proceed using this file and the repository handover docs as the operational memory.

## 0. Source Of Truth

Use these files as the primary source of truth:

```text
README.md
docs/handover/CURRENT_STATE.md
docs/handover/SERVER_ARTIFACT_INDEX.md
docs/handover/SERVER_PATHS.md
docs/handover/RUNBOOK.md
```

The chat history should not be treated as the only source of truth. It contains temporary plans, failed attempts, corrected assumptions, and outdated paths. If chat history conflicts with the files above, trust the repository handover docs and verify on the active servers.

The repository contains code and handover documentation. It does not contain large artifacts such as checkpoints, datasets, text feature cache, OCR bbox/proxy JSONL, reconstruction images, or large logs. Those live on the active servers.

## 1. Current Goal

The project goal is text-rich GigaTok discrete tokenizer fine-tuning.

Current main method:

```text
OCR-box gate + DeepSeek-OCR token-logprob proxy confidence
```

Current matched baseline:

```text
GigaTok-only tokenizer baseline on the same realworld40_5k data
```

The active comparison is:

```text
main OCR-box proxy method vs GigaTok 5k baseline
```

Current main method components:

- text conditioning enabled;
- OCR teacher-forcing / OCR CE enabled;
- OCR-box gate enabled;
- proxy confidence enabled;
- bbox from DeepSeek-OCR;
- proxy confidence from DeepSeek-OCR generated token logprob;
- HR / local similarity / OCR visual-alignment disabled.

Current baseline components:

- no text conditioning;
- no OCR CE;
- no OCR-box gate;
- no proxy confidence;
- no HR/local/OCR visual-alignment.

## 2. Current Dataset Naming

### `realworld40`

`realworld40` means a text-rich data variant where real-world / mixed image sources are upweighted to about 40%.

In this project, the real-world / mixed component mainly refers to:

```text
TextScenesHQ
TextVisionBlend
```

The synthetic / cleaner text image sources include:

```text
StyledTextSynth
CleanTextSynth
LongWordsSubset-A
```

### `realworld40_5k`

`realworld40_5k` is the current main training setup:

```text
train = 5000
val = 500
holdout = 200
```

It was derived from a previously selected/scored realworld40 10k/1k/500 pool. It was not rescreened from scratch from the full 300k source at this stage.

### `train_5k_realworld40_text_rich`

5,000-image training manifest for the current main and baseline experiments.

### `val_500_realworld40_text_rich`

500-image validation manifest used for training-time validation and sanity checks.

### `holdout_200_realworld40_text_rich`

Earlier 200-image holdout split derived from the 5k setup. Later `holdout_v2` candidates were prepared for final reporting.

### `holdout_v2_readable_balanced_200`

Primary 200-image holdout candidate for main final tables.

Intended properties:

- readable text;
- source-balanced;
- fixed as primary before final reporting;
- should not be swapped after seeing results.

### `holdout_v2_readable_medium_200`

Secondary / supplementary 200-image holdout candidate.

Intended properties:

- readable text;
- biased more toward medium text coverage;
- should be treated as supplementary, not swapped into primary based on results.

### `readable50_textscenes40_*`

Three curated 50-image probe sets used for readable50 ablation/effect checks:

```text
readable50_textscenes40_balanced
readable50_textscenes40_medium
readable50_textscenes40_dense_lite
```

These are not the main unbiased holdout. They should be described as curated/probe sets, not as the final random test set.

## 3. Active Servers

Only two active servers should be used for future work unless a new server is explicitly documented.

### New 8-card NPU server

Role:

- main training;
- final 250-epoch checkpoints;
- 5k data artifacts;
- OCR bbox/proxy artifacts for the main training setup;
- NPU-side eval attempts;
- current main/baseline continuation outputs.

Project root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss
```

Expected repo copy:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation
```

Persist root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist
```

### 8x3090 CUDA server

Role:

- readable50 eval outputs;
- CUDA-side OCR/eval utilities;
- DeepSeek-OCR / CUDA-side helper work;
- legacy readable50 method checkpoint/config archive.

Main artifact root:

```bash
/public/sdc/yl_GigaTok_Loss
```

Repo copy:

```bash
/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation
```

Persist root:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist
```

Large files are not in git. Checkpoints, caches, bbox/proxy JSONL, reconstruction images, and logs live on these servers.

## 4. Critical Artifacts

### Main OCR-box proxy 250ep checkpoint

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

Last verified size:

```text
about 6.2G
```

### GigaTok 5k baseline 250ep checkpoint

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

Last verified size:

```text
about 7.1G
```

### 5k manifests

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_rich_realworld_upweighted_manifests
```

Important files include:

```bash
train_5k_realworld40_text_rich.newserver.jsonl
val_500_realworld40_text_rich.newserver.jsonl
holdout_200_realworld40_text_rich.newserver.jsonl
```

Last verified row counts:

```text
train = 5000
val = 500
```

### Text feature cache

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/text_feature_cache/realworld40_glyph_mapped
```

This cache is used by the OCR-box proxy method.

### OCR bbox/proxy

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/realworld40_5k
```

This directory should contain final bbox and `_with_proxy_conf` JSONL artifacts for the 5k training setup.

### holdout_v2 prepared/eval

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin
```

This contains holdout_v2 selections, prepared inputs, path-aligned proxy JSONL copies, smoke evals, and some failed/intermediate attempts. Do not trust every marker in this directory blindly. Cross-check with `docs/handover/CURRENT_STATE.md`.

### readable50 eval

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

Expected subdirectories:

```bash
readable50_textscenes40_balanced
readable50_textscenes40_medium
readable50_textscenes40_dense_lite
```

These contain basic reconstruction eval outputs and grids.

### legacy readable50 archive

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/legacy_readable50_methods_20260519/gigatok_persist_subset
```

This is the archive of legacy readable50 method checkpoint/config files.

Important caveat:

```text
At handover time, this archive transfer was still in progress. Verify rsync log and checkpoint count before relying on it.
```

Transfer log:

```bash
/public/sdc/yl_GigaTok_Loss/logs/legacy_readable50_methods_transfer_20260519/rsync.log
```

## 5. Training State

### Main OCR-box proxy

Status:

```text
completed
250 epochs
step = 26250
```

Checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

### GigaTok 5k baseline

Status:

```text
completed
250 epochs
step = 26250
```

Checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

### Resume / continuation rules

If continuing training:

- do not use `--finetune`;
- explicitly pass `--vq-ckpt <last.pt>`;
- do not silently fall back to the initial VQ checkpoint;
- do not silently fall back to a random/latest numeric checkpoint unless explicitly reviewed;
- verify checkpoint contains model state, optimizer state, discriminator optimizer state, and step count;
- `iterations=26250` means total target optimizer steps, not additional steps after resume.

Step/epoch mapping used in this project:

```text
100 epochs = 10500 optimizer steps
250 epochs = 26250 optimizer steps
```

## 6. Method Details To Preserve

### OCR bbox

The bbox should come from DeepSeek-OCR.

Coordinate system:

```text
256x256 resize-pad input coordinates
```

Do not use original image coordinates for gate/mask/eval unless a script explicitly converts them to the 256x256 input space.

### Proxy confidence

Proxy confidence should come from DeepSeek-OCR generated token log probabilities.

Required final normalization:

```text
normalization = robust_p10_p90_from_mean_logprob
normalization_scope = global_all_valid_boxes_after_merge
```

Formula:

```text
confidence = clamp((mean_logprob - p10_logprob) / (p90_logprob - p10_logprob + 1e-8), 0, 1)
```

Valid boxes are boxes where:

```text
mean_logprob is not None
token_span is not None
proxy_conf_valid = true
```

Invalid boxes should be:

```text
proxy_conf_valid = false
confidence = 0.0
```

Invalid boxes must not participate in p10/p90 calculation.

Forbidden final behavior:

- no min/max normalization;
- no shard-local final normalization;
- no fake/default confidence;
- no GT text replacement for OCR text;
- no silent fallback if token-logprob scoring fails.

### Gate semantics

The gate is an image-side gate controlling residual text injection.

Preserve this meaning:

```text
OCR/text feature injection is spatially controlled by bbox-derived gate values.
```

Do not change it into a CE loss gate or a different model mechanism without explicitly documenting a new experiment.

### Main method module switches

Current main method should keep:

```text
text_conditioning: enabled
ocr_teacher_forcing_loss / OCR CE: enabled
ocr_box_gate: enabled
proxy confidence: enabled
text_hr: disabled
local_similarity_loss: disabled
ocr_visual_alignment_loss: disabled
```

## 7. Eval State

### readable50

readable50 basic reconstruction eval exists on the 3090 server:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

Expected sets:

```bash
readable50_textscenes40_balanced
readable50_textscenes40_medium
readable50_textscenes40_dense_lite
```

Available / trusted at handover:

- basic reconstruction outputs;
- PSNR / SSIM;
- Text-PSNR / Text-SSIM;
- per-sample JSONL;
- reconstruction images;
- qualitative grids.

Not confirmed:

```text
readable50 OCR CER/NED
```

Do not report readable50 OCR CER/NED unless it is computed or verified from saved reconstructions.

### holdout_v2

Two holdout_v2 sets exist:

```text
holdout_v2_readable_balanced_200
holdout_v2_readable_medium_200
```

Path-aligned NPU eval proxy files exist for both. They were created because the first full eval attempt failed due to image path prefix mismatch between manifest and bbox/proxy JSONL.

Important:

- alignment matched by `sha256_after_prefix_map`;
- only top-level `image_path` was rewritten;
- bbox/proxy/confidence values were preserved;
- 10-sample smoke eval passed for both balanced and medium after the fix.

Current caveat:

```text
Full 200-image holdout_v2 eval still needs verification or rerun before final reporting.
```

Do not treat failed/intermediate markers in `overnight_allin` as final state without checking `CURRENT_STATE.md`.

## 8. Known Issues / Do Not Do

Do not use any server outside the two active 8-card servers as a future dependency unless explicitly documented.

Do not put large files into git:

- checkpoints;
- datasets;
- text feature cache;
- OCR bbox/proxy JSONL;
- reconstruction images;
- large logs.

Do not substitute unrelated models into tables:

- do not put pretrained GigaTok into readable50 old ablation main table;
- do not put the realworld40 5k baseline into old readable50 ablation main table unless clearly labeled as a separate reference.

Do not report unverified results:

- do not report full holdout_v2 final table unless full 200-image eval is verified;
- do not report OCR CER/NED unless verified;
- do not treat smoke_10 as full eval.

Do not change evaluation selection after seeing results:

- `holdout_v2_readable_balanced_200` remains primary;
- `holdout_v2_readable_medium_200` remains secondary/supplementary.

Do not change proxy semantics:

- no min/max normalization;
- no shard-local normalization;
- no fake/default confidence;
- no GT text replacement.

Do not resume training with:

```bash
--finetune
```

Do not delete server-side critical artifacts:

- `last.pt`;
- `0002500.pt`;
- `0026250.pt`;
- bbox/proxy JSONL;
- text feature cache;
- readable50 reconstructions;
- logs needed to audit training/eval.

## 9. Next Recommended Actions

Priority order:

1. Verify or regenerate full 200-image holdout_v2 eval:
   - `holdout_v2_readable_balanced_200` as primary;
   - `holdout_v2_readable_medium_200` as secondary.
2. Compute or verify readable50 OCR CER/NED from saved reconstructions.
3. Verify legacy readable50 archive transfer completion and checkpoint count.
4. Build final tables only from verified full outputs.
5. Write method/results draft using verified values only.
6. If continuing training further, use explicit resume from `last.pt`, no `--finetune`, and do not fallback to init checkpoint.

## 10. Quick Verification Commands

### New 8-card NPU server

```bash
ssh root@139.9.157.110
docker exec -it node1new bash
```

Inside the container:

```bash
ROOT=/data2/duoduo_25/yl_GigaTok_Loss

ls -lh $ROOT/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt

ls -lh $ROOT/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt

wc -l \
  $ROOT/gigatok_persist/outputs/text_rich_realworld_upweighted_manifests/train_5k_realworld40_text_rich.newserver.jsonl \
  $ROOT/gigatok_persist/outputs/text_rich_realworld_upweighted_manifests/val_500_realworld40_text_rich.newserver.jsonl

ls $ROOT/gigatok_persist/outputs/ocr_box_gate/realworld40_5k

ls $ROOT/gigatok_persist/outputs/overnight_allin
```

Optional checkpoint key check:

```bash
/data/conda/envs/train/bin/python - <<'PY'
import torch
paths = [
"/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt",
"/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt",
]
for p in paths:
    ckpt = torch.load(p, map_location="cpu")
    print("PATH", p)
    print("KEYS", sorted(ckpt.keys())[:50])
    print("steps", ckpt.get("steps", ckpt.get("step", None)))
PY
```

### 8x3090 CUDA server

```bash
ssh ss420f.cs.cityu.edu.hk
```

Check readable50 eval outputs:

```bash
ROOT=/public/sdc/yl_GigaTok_Loss

ls $ROOT/gigatok_persist/outputs/overnight_allin/readable50_eval

find $ROOT/gigatok_persist/outputs/overnight_allin/readable50_eval -maxdepth 2 -type f | head -n 50
```

Check legacy readable50 archive transfer:

```bash
ROOT=/public/sdc/yl_GigaTok_Loss

cat $ROOT/logs/legacy_readable50_methods_transfer_20260519/rsync.log | tail -n 40

if [ -f $ROOT/logs/legacy_readable50_methods_transfer_20260519/rsync.pid ]; then
  ps -p $(cat $ROOT/logs/legacy_readable50_methods_transfer_20260519/rsync.pid) -o pid,etime,stat,cmd
fi

du -sh $ROOT/gigatok_persist/outputs/legacy_readable50_methods_20260519/gigatok_persist_subset

find $ROOT/gigatok_persist/outputs/legacy_readable50_methods_20260519/gigatok_persist_subset -name "*.pt" | wc -l
```

## 11. Handoff Message

You can send this to the teammate:

```text
我已经把代码和交接文档整理好了。代码/文档里只放小文件和说明，大文件都在两台 8 卡服务器上。README 和 docs/handover 里写了当前方法、实验状态、checkpoint/cache/bbox/proxy/eval/log 的路径，以及哪些结果已经完成、哪些还不能直接引用。

当前两个 250ep checkpoint 已经完成：
1. OCR-box gate + proxy confidence 主方法；
2. 同一 realworld40_5k 数据上的 GigaTok-only baseline。

需要继续确认的是：
1. full 200-image holdout_v2 eval；
2. readable50 OCR CER/NED；
3. legacy readable50 checkpoint/config archive transfer 是否完全结束。

交接入口：
请使用随附的代码压缩包、私有仓库权限，或服务器项目目录中的 README.md 和 docs/handover/。如果使用私有 GitHub 仓库，需要先确认接手人已有访问权限。
```
