# GigaTok Loss Handover

Last updated: 2026-05-19

This handover summarizes the project state, where the important files live, and how a new owner can continue without reverse-engineering the last several weeks of experiments.

**Read first:** `README_FOR_HANDOVER.md`, then `CURRENT_STATE.md`. `CURRENT_STATE.md` is the authoritative snapshot for the current server status. Older notes, SVG diagrams, and failed-marker files are useful history but may not reflect the latest progress after path fixes and reruns.

## Project Scope

The work extends the GigaTok B-L discrete tokenizer fine-tuning pipeline for text-heavy image reconstruction. The project has several experimental branches of ideas:

- Original GigaTok / VQ tokenizer baseline.
- Text-conditioned tokenizer fine-tuning using Glyph-SDXL / ByT5 text features.
- OCR teacher-forcing loss.
- High-rank / local-sim / OCR visual-alignment ablations.
- OCR-box gate, including a proxy-confidence variant that uses DeepSeek-OCR token logprobs.

The current main line is **stage-1 tokenizer fine-tuning only**. The AR model is not part of the active experiment.

Historical HR/local/OCR-visual-alignment material remains in this repo for context. The current main OCR-box proxy experiment keeps HR/local/OCR visual alignment disabled.

## Current Main Experiment

The current primary experiment is:

- Dataset: `realworld40_5k`
- Train split: `train_5k_realworld40_text_rich`
- Val split: `val_500_realworld40_text_rich`
- Holdout candidates: `holdout_v2_readable_balanced_200` and `holdout_v2_readable_medium_200`
- Main method: OCR-box gate + proxy confidence
- Baseline: GigaTok-only baseline trained on the same 5k data
- Target length: 250 epochs, represented as `iterations=26250`

As of this handover:

- The main OCR-box proxy run has continued from 100 epoch to 250 epoch on the new 8-card NPU server and completed at `step=26250`.
- The GigaTok-only baseline has also continued to 250 epoch on the new 8-card NPU server and completed at `step=26250`.
- 3090-side readable50 basic reconstruction evals exist for three curated 50-image sets.
- OCR CER/NED second-pass readability for readable50 still needs to be completed or verified.
- NPU-side holdout_v2 eval initially failed due to `image_path` mismatch between manifest and bbox/proxy JSONL. Path-aligned proxy JSONL copies were generated for the two holdout_v2 sets, and 10-sample smoke eval passed after the fix. Full 200-image eval still needs completion/verification.

## Primary Rules To Preserve

- Do not resume with `--finetune`.
- Resume from `last.pt` or an explicit checkpoint only.
- Do not silently fall back to the initial VQ checkpoint for continuation runs.
- Keep train/val/test manifests fixed once an experiment is reported.
- For OCR-box gate v2, proxy confidence must come from DeepSeek-OCR token logprob and use global robust p10/p90 normalization.
- Do not use fake/default confidence values.
- Do not use shard-local confidence normalization in final proxy JSONL.
- Keep HR/local/OCR visual-alignment disabled for the current main OCR-box proxy experiment.
- Treat `holdout_v2_readable_balanced_200` as the primary main table and `holdout_v2_readable_medium_200` as secondary/supplementary.

## Documentation Map

- `README_FOR_HANDOVER.md`: first-read guide for someone taking over from zero.
- `CURRENT_STATE.md`: authoritative current snapshot and warnings about stale historical records.
- `RUNBOOK.md`: commands and procedures to continue training/eval.
- `SERVER_QUICKSTART.md`: exact login/env/check/eval command templates for first-time server use.
- `SERVER_PATHS.md`: important paths on the new 8-card NPU, 8x3090 CUDA server, and old 2-card NPU.
- `EXPERIMENT_STATUS.md`: current status of training, eval, readable50, and unresolved items.
- `CODE_CHANGES.md`: summary of key code changes and what they were for.
- `KNOWN_ISSUES.md`: known failure modes and how to avoid repeating them.
- `GIT_AND_ARTIFACTS.md`: what belongs in Git and what should remain as external artifacts.

## What Should Be Preserved

Keep these outside Git, but do not delete unless they have been archived elsewhere:

- Final 250 epoch checkpoints for both main and baseline runs.
- `realworld40_5k` manifests and summaries.
- Final OCR bbox/proxy JSONL files for training and holdout evaluation.
- Text feature cache used by the main method.
- readable50 reconstruction/eval outputs on 3090.
- Overnight logs and summaries, at least until the final report is prepared.

## Immediate Next Steps

1. Finish or verify NPU holdout_v2 eval using the path-aligned `.npu_eval.jsonl` proxy files.
2. Finish OCR CER/NED second-pass readability for readable50 outputs on 3090.
3. Generate final tables with:
   - PSNR
   - SSIM
   - DeepSeek OCR CER
   - DeepSeek OCR NED
   - Text-PSNR
   - Text-SSIM
4. Decide which readable50 curated set is used as the probe display set. Do not call it an unbiased final test.
5. Commit only code/config/docs to Git; leave checkpoints and large outputs in server artifact directories.
