# Conversation Summary For Handover

Last updated: 2026-05-19

This file is not a verbatim chat export. The raw chat contains many temporary plans, failed attempts, corrected assumptions, and outdated paths. This summary preserves the project-relevant decisions, current experiment state, server layout, and known unfinished items needed for handover.

Use the repository README and `docs/handover/CURRENT_STATE.md` as the source of truth. Use this file as background context for how the current state was reached.

## 1. Project Goal

The project is about improving GigaTok discrete image tokenizer reconstruction for text-rich images.

The current main method is:

- text-conditioned tokenizer fine-tuning;
- OCR-box gate;
- DeepSeek-OCR token-logprob proxy confidence;
- OCR teacher-forcing / OCR CE loss;
- image-side gating of residual text injection;
- HR / local similarity / OCR visual-alignment disabled for the current main run.

The current baseline is:

- GigaTok-only tokenizer fine-tuning on the same `realworld40_5k` data;
- no text conditioning;
- no OCR CE;
- no OCR-box gate;
- no proxy confidence;
- no HR/local/OCR visual-alignment.

The current comparison is therefore:

```text
OCR-box proxy main method vs GigaTok-only baseline
```

## 2. Naming And Dataset Decisions

The dataset name `realworld40_5k` was introduced during the conversation.

Meaning:

- `realworld40`: text-rich data variant with about 40% real-world / mixed image sources.
- Real-world / mixed sources mainly mean `TextScenesHQ` and `TextVisionBlend`.
- `5k`: current training scale is 5,000 train images.

Final current split:

```text
train_5k_realworld40_text_rich
val_500_realworld40_text_rich
holdout_200_realworld40_text_rich
```

Later holdout candidates:

```text
holdout_v2_readable_balanced_200
holdout_v2_readable_medium_200
```

Intended use:

- `holdout_v2_readable_balanced_200`: primary candidate for main final table.
- `holdout_v2_readable_medium_200`: secondary / supplementary candidate.
- These should not be swapped after looking at results.

readable50 sets:

```text
readable50_textscenes40_balanced
readable50_textscenes40_medium
readable50_textscenes40_dense_lite
```

Intended use:

- curated/probe sets for readable50 ablation and qualitative checking;
- not the main unbiased holdout;
- do not describe them as the final random test set.

## 3. Data Selection History

The first idea was to use a 10k train / 1k val / 500 holdout dataset. This was later reduced to:

```text
train = 5000
val = 500
holdout = 200
```

Reason:

- reduce preprocessing cost;
- reduce OCR bbox / proxy-confidence time;
- match available time and compute;
- still keep a text-rich mixed-source dataset.

The selected 5k data was derived from the already selected/scored 10k data, not from a full rescreen of 300k data.

Important constraints during selection:

- keep real-world / mixed image share around 40%;
- prefer readable text-rich samples;
- avoid too many easy sparse-text images;
- avoid a dataset dominated by white-paper dense document images;
- keep train / val / holdout split leakage at zero;
- allow train duplicates only if explicitly marked, but val/holdout should not contain duplicates.

## 4. Server Layout

Two active servers remain relevant for future work.

### New 8-card NPU server

Role:

- main training;
- final 250-epoch checkpoints;
- 5k data artifacts;
- NPU-side eval attempts;
- current main/baseline continuation outputs.

Project root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss
```

Important checkpoint paths:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt

/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

### 8x3090 CUDA server

Role:

- readable50 evaluation outputs;
- CUDA-side OCR/eval utilities;
- legacy readable50 method checkpoint/config archive;
- DeepSeek-OCR / CUDA-side helper work.

Main artifact root:

```bash
/public/sdc/yl_GigaTok_Loss
```

Repo copy:

```bash
/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation
```

readable50 output root:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

legacy readable50 method archive:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/legacy_readable50_methods_20260519/gigatok_persist_subset
```

At the time of handover, this archive transfer was still in progress and should be verified before use.

## 5. Training History

Two formal 250-epoch runs were completed on the new 8-card NPU server.

### Main OCR-box proxy run

Configuration:

- OCR-box gate enabled;
- proxy confidence enabled;
- OCR CE enabled;
- text conditioning enabled;
- HR/local/OCR visual-alignment disabled;
- global batch size 48;
- 8 NPU workers;
- continued from 100 epochs to 250 epochs.

Current final checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/ocr_box_gate/formal_runs/ocr_box_gate_conf_proxy_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

Status:

```text
completed at step=26250
```

### GigaTok-only baseline

Configuration:

- same `realworld40_5k` data;
- no text/OCR/gate/proxy/HR/local/ocrvis modules;
- global batch size 48 on the new 8-card NPU server for the continuation phase.

Current final checkpoint:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/original_gigatok_baseline_realworld40_5k/orig_gigatok_realworld40_5k_gbs48_continue_250ep/train/checkpoints/last.pt
```

Status:

```text
completed at step=26250
```

## 6. Resume Rules Established During The Conversation

Important resume rules:

- never resume continuation training with `--finetune`;
- `--finetune` resets or changes training semantics and can skip optimizer state;
- resume should explicitly pass `--vq-ckpt <last.pt>`;
- do not silently fall back to initial VQ checkpoint;
- do not silently fall back to an unrelated numeric checkpoint;
- checkpoint should include model state, optimizer state, discriminator optimizer state, and step count;
- continuation target `iterations=26250` means total target optimizer steps, not extra steps after resume.

Epoch / step relationship used:

```text
100 epochs = 10500 optimizer steps
250 epochs = 26250 optimizer steps
```

## 7. Proxy Confidence Decisions

The method uses DeepSeek-OCR token log probabilities as proxy confidence.

Required final normalization:

```text
normalization = robust_p10_p90_from_mean_logprob
normalization_scope = global_all_valid_boxes_after_merge
```

Formula:

```text
confidence = clamp((mean_logprob - p10_logprob) / (p90_logprob - p10_logprob + 1e-8), 0, 1)
```

Valid boxes:

- `mean_logprob is not None`;
- `token_span is not None`;
- `proxy_conf_valid = true`.

Invalid boxes:

- `proxy_conf_valid = false`;
- `confidence = 0.0`;
- excluded from p10/p90.

Forbidden:

- fake/default confidence;
- min/max normalization;
- shard-local final normalization;
- GT text scoring;
- changing bbox/gate semantics to make the pipeline pass.

## 8. Evaluation State

### Main holdout_v2

The initial full NPU eval failed because bbox/proxy JSONL image paths used a 3090-side prefix while NPU manifests used `/data2/...`.

This was a path-join mismatch, not a model or method change.

Path-aligned NPU eval proxy files now exist for:

```bash
holdout_v2_readable_balanced_200
holdout_v2_readable_medium_200
```

The path-alignment used:

```text
match method: sha256_after_prefix_map
```

Only the top-level `image_path` was rewritten. bbox/proxy/confidence values were preserved.

10-sample smoke eval passed after the path fix for both holdout_v2 sets.

Current caveat:

```text
Full 200-image holdout_v2 eval still needs verification or regeneration before reporting final numbers.
```

### readable50

Authoritative readable50 basic reconstruction outputs are on the 3090 server:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

Sets:

```text
readable50_textscenes40_balanced
readable50_textscenes40_medium
readable50_textscenes40_dense_lite
```

Basic metrics and reconstructions exist:

- PSNR;
- SSIM;
- Text-PSNR;
- Text-SSIM;
- per-sample JSONL;
- reconstruction images;
- grids.

Current caveat:

```text
OCR CER/NED second-pass readability was not confirmed complete at handover time.
```

The same-protocol `gigatok_only` checkpoint for old readable50 ablation was not found. Do not substitute pretrained GigaTok or the realworld40 5k baseline into the readable50 ablation main table unless clearly labeled as a separate reference.

## 9. Code Areas Discussed

Important code areas:

```bash
tokenizer/tokenizer_image/vq/
dataset/ocr_box_gate.py
scripts/stage1/ocr_debug/
scripts/stage1/eval_unified_realworld40_5k.py
tools/run_all_in_companion_downstream_worker.py
```

Note:

```text
scripts/stage1/...
```

is a historical folder name. It should not be used as the paper-facing method name.

## 10. Git / Repository Handover Work

A dedicated handover branch was created and made the GitHub default branch:

```bash
codex/handover-gigatok-20260519
```

Repository URL:

```text
https://github.com/mayday4yl/GigaTok_Loss
```

The README was converted into a handover entry point. It now explains:

- current method;
- current dataset naming;
- server roles;
- critical server paths;
- what is finished;
- what is not final;
- how to read the handover docs.

Handover docs live in:

```bash
docs/handover/
```

Key files:

```bash
README_FOR_HANDOVER.md
CURRENT_STATE.md
SERVER_ARTIFACT_INDEX.md
SERVER_PATHS.md
RUNBOOK.md
EXPERIMENT_STATUS.md
CODE_CHANGES.md
KNOWN_ISSUES.md
GIT_AND_ARTIFACTS.md
```

## 11. Known Caveats To Tell The Next Owner

Do not overclaim these items:

1. Full 200-image holdout_v2 final tables need verification/regeneration.
2. readable50 OCR CER/NED needs verification/completion.
3. The legacy readable50 method archive on the 3090 server may still be transferring; verify the rsync log and checkpoint count before use.
4. Some server directories contain failed or intermediate attempts; do not report numbers just because a file exists.
5. Old HR/local/OCR visual-alignment figures and notes are historical, not the current main method.
6. The old readable50 `gigatok_only` checkpoint was not found under the same protocol.

## 12. Suggested Handover Message

You can send this message with the GitHub link:

```text
I整理好了项目交接分支，GitHub 默认打开就是交接版 README。代码、方法口径、当前进度、两台 8 卡服务器上的 checkpoint/cache/eval/log 路径都写在 README 和 docs/handover 里。大文件没有进 git，需要按文档去服务器上看。当前两个 250ep checkpoint 已完成；full holdout_v2 表格和 readable50 OCR CER/NED 还需要按文档补验证。

Repo: https://github.com/mayday4yl/GigaTok_Loss
```

