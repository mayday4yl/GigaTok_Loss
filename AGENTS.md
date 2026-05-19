# AGENTS.md

## Current Project

Text-rich image tokenization for GigaTok B-L discrete tokenizer fine-tuning.

The current paper-facing direction is a text-conditioned discrete tokenizer that improves reconstruction of text-rich images by combining:

- Glyph/ByT5 text feature conditioning;
- DeepSeek-OCR teacher-forcing loss;
- OCR-box spatial gating for localized residual text injection;
- DeepSeek-OCR token-logprob proxy confidence for low-confidence text suppression;
- a GigaTok-only baseline trained on the same `realworld40_5k` data.

Avoid describing the active work with the legacy internal phase label in user-facing docs or paper-facing summaries. Some source directories still have historical names; treat those names as implementation paths, not as the method name.

## Current Scope

- Target tokenizer: GigaTok B-L with the DINO discriminator/VQ checkpoint setting.
- Active task: discrete tokenizer fine-tuning for text-rich image reconstruction.
- Do not change the AR model.
- Do not redesign the full training framework unless explicitly asked.
- Keep encoder/quantizer/codebook freeze semantics aligned with the current run configs.
- Keep current DINOv2 loading strategy unless explicitly asked.

## Current Main Method

The current main experiment is OCR-box gate + proxy confidence:

- text conditioning enabled;
- OCR teacher-forcing CE enabled;
- OCR-box gate enabled;
- proxy confidence derived from DeepSeek-OCR token logprob;
- final proxy normalization uses global robust p10/p90 over all valid boxes after merge;
- HR/local/OCR visual-alignment disabled for the main method.

The GigaTok-only baseline is the matched baseline on the same `realworld40_5k` train/val data.

## Historical Code Paths

High-rank, local-sim, and OCR visual-alignment code exists for historical experiments and ablations. Do not present those modules as part of the active main method unless the user explicitly asks about those ablations.

## Data

Current main data:

- `train_5k_realworld40_text_rich`
- `val_500_realworld40_text_rich`
- holdout candidates `holdout_v2_readable_balanced_200` and `holdout_v2_readable_medium_200`

Text-rich selection uses TextScenesHQ, TextVisionBlend, StyledTextSynth, CleanTextSynth, and LongWordsSubset-A. Keep aspect ratio and resize-pad to 256x256; do not replace with crop unless explicitly requested.

## Safety Rules

- Do not resume continuation training with `--finetune`.
- Do not silently fall back to the initial VQ checkpoint for continuation runs.
- Do not use fake/default proxy confidence.
- Do not use min/max or shard-local proxy normalization for final proxy JSONL.
- Do not enable HR/local/OCR visual-alignment for the current main OCR-box proxy experiment.
- Do not modify/delete files outside project-owned server roots.
- Do not delete checkpoints, manifests, or large artifacts unless explicitly requested.

## Workflow

- 默认使用中文与用户交流。
- 对复杂任务先检查代码和路径，再给出计划；用户确认后再做高风险修改。
- 修改代码后先做 smoke test，再建议长时间训练或 eval。
- 每个重要阶段结束后，把结论写入 `docs/codex_worklog.md`。
- 交接相关说明优先更新 `docs/handover/README_FOR_HANDOVER.md`、`docs/handover/CURRENT_STATE.md`、`docs/handover/RUNBOOK.md`。
