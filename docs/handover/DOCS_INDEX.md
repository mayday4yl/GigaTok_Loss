# Documentation Index

Last updated: 2026-05-19

This index explains the older project notes that are useful for handover but are not required to run the current 5k main experiment.

## Method / Code Understanding

### `docs/text_feature_injection_audit.md`

Explains how the current `glyph_byt5` text encoder path works, why config layer `0` means the single exposed Glyph-mapped ByT5 output rather than the shallow ByT5 layer, and how text features are injected.

Use this when changing or explaining text conditioning.

### `docs/text_feature_cache_plan.md`

Design note for text feature caching. It explains what should be cached, cache keys, and why the trainable `text_projection` should usually stay outside the frozen text feature cache.

Use this before modifying text feature cache generation.

### `docs/original_gigatok_line_review.md`

Older line-level review of changes relative to original GigaTok. Some line numbers may be stale after later changes, but it is still useful as a conceptual map of where the main model/training changes entered.

Use this for code review orientation, not as an exact line-number authority.

## Experiment Planning Notes

### `docs/longer_run_plan.md`

Older plan for longer OCR/no-OCR sweeps. It explains why early short sweeps were not enough and why longer baseline training was considered before increasing OCR loss weight.

Use this to understand historical experiment decisions.

## Figures / Diagrams

### `docs/stage1_text_hr_flow.svg`

Early stage-1 Text-HR pipeline diagram.

### `docs/text_hr_v2_framework.svg`

Framework diagram for the Text-HR v2 direction.

### `docs/text_hr_v2_detailed_flow.svg`

More detailed Text-HR v2 flow diagram.

These SVGs are for communication and paper/slides reference. They are not source of truth for the latest OCR-box proxy method.

## Historical Sweep Script

### `scripts/stage1/ocr_debug/run_ocr_recon_effect_sweep.sh`

Script for earlier OCR reconstruction-effect weight sweeps. It is not the current realworld40_5k main training script, but it can help reproduce older OCR CE weight experiments.

Recent small edit adds intermediate OCR weights:

- `w0005 = 0.005`
- `w0015 = 0.015`
- `w0025 = 0.025`

## Deliberately Not Included

The local folder `tokenizer/tokenizer_image/改动/` is a temporary backup/copy folder and should not be used as source code. It is intentionally not included in the handover commits.
