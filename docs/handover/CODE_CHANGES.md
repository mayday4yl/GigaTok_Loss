# Code Changes Summary

Last updated: 2026-05-19

This is a high-level map of changes made during the project. Use `git diff` for exact line-level details.

## High-Rank / Text Conditioning Work

Relevant files:

```bash
tokenizer/tokenizer_image/vq/blocks.py
tokenizer/tokenizer_image/vq/vq_vit_model.py
tokenizer/tokenizer_image/vq/vq_loss.py
tokenizer/tokenizer_image/vq/vq_train.py
configs/vq/
```

Purpose:

- Add text-conditioned tokenizer fine-tuning paths.
- Add high-rank loss experiments on decoder attention.
- Support logging and config switches for optional losses.

Original tokenizer-only rule remains important: do not modify the AR model for this work.

Current-status note: HR/local/OCR visual-alignment code exists for historical and ablation work, but the current main `realworld40_5k` OCR-box proxy experiment disables HR/local/OCR visual alignment.

## OCR Teacher Forcing / OCR-Box Gate

Relevant files:

```bash
dataset/ocr_box_gate.py
scripts/stage1/ocr_debug/generate_deepseek_ocr_bbox_jsonl.py
scripts/stage1/ocr_debug/probe_deepseek_ocr_token_confidence.py
scripts/stage1/eval_unified_realworld40_5k.py
```

Purpose:

- Generate OCR bbox rows from DeepSeek-OCR.
- Generate proxy confidence from DeepSeek-OCR token logprobs.
- Convert bbox to a 16x16 image-side gate for residual text injection.

Method constraints:

- Bbox coordinate space must be 256x256 resize-pad input coordinates.
- Proxy confidence must use DeepSeek-OCR token logprob.
- Final confidence normalization must be global robust p10/p90 over valid boxes after merge.
- The gate controls image-side residual text injection, not the OCR CE loss itself.

## Gradient Accumulation

Relevant file:

```bash
tokenizer/tokenizer_image/vq/vq_train.py
```

Purpose:

- Support `gradient_accumulation_steps`.
- Treat `global_batch_size` as micro global batch.
- Compute effective global batch as `global_batch_size * gradient_accumulation_steps`.
- Step optimizer/scheduler and increment train step only on accumulation boundary.

Important:

- `iterations`, `val_every`, and `ckpt_every` are optimizer-step based.
- `gradient_accumulation_steps=1` should behave like the original training loop.

## Realworld40 / hardmix Dataset Selection

Relevant files:

```bash
scripts/stage1/ocr_debug/select_text_rich_realworld40_fast.py
scripts/stage1/ocr_debug/select_text_rich_hardmix_manifests.py
```

Purpose:

- Select text-rich but not all white-page-document samples.
- Upweight TextScenesHQ and TextVisionBlend real-world / mixed samples.
- Produce train/val/holdout/qual manifests and scored manifests.

Important:

- The final current training data was downsampled to `train_5k / val_500 / holdout_200`.
- Do not re-randomize these splits for a reported experiment.

## Unified Eval and OCR Readability

Relevant files:

```bash
scripts/stage1/eval_unified_realworld40_5k.py
scripts/stage1/ocr_debug/evaluate_saved_recon_ocr_readability.py
scripts/stage1/ocr_debug/evaluate_two_ablation_readable50.py
```

Purpose:

- Generate reconstruction metrics from the same saved reconstructions.
- Use a shared input-side 256x256 bbox/mask for text-region metrics.
- Run OCR readability as a second pass over saved reconstructions.

Important:

- SSIM should be computed on images converted to `[0,1]` with `data_range=1.0`.
- Text-region metrics should use shared input-side 256x256 bbox/mask.
- Single-sample failures should be recorded, not silently skipped.

## CUDA / 3090 Compatibility Patch

File patched on the 3090 server:

```bash
/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation/tokenizer/tokenizer_image/vq/glyph_byt5.py
```

Backup:

```bash
/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation/tokenizer/tokenizer_image/vq/glyph_byt5.py.before_cache_position_patch_20260519_003057
```

Purpose:

- Add compatibility for newer Transformers/T5 attention calling convention involving `cache_position`.
- This was for evaluation/cache compatibility, not a method change.

## Path Alignment Fix

Problem:

- NPU eval manifest used `/data2/duoduo_25/yl_GigaTok_Loss/...`.
- Proxy JSONL generated on 3090 used `/public/sdc/yl_GigaTok_Loss/...`.
- The evaluator joins bbox rows by exact `image_path`, so it failed with `bbox missing`.

Fix:

- Generate `.npu_eval.jsonl` copies where only top-level `image_path` is rewritten to the NPU manifest path.
- Match rows by sha256 after prefix mapping.
- Preserve bbox/proxy/confidence/logprob/token_span values.

Outputs:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_balanced_200/path_alignment_summary.md
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/prepared_inputs/holdout_v2_readable_medium_200/path_alignment_summary.md
```
