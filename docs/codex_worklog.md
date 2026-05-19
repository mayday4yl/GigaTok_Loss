# Codex Worklog

This handover branch keeps only the current handover-oriented worklog. Older exploratory notes were intentionally removed from this branch because they contained stale server paths and historical method directions that are no longer dependencies for the current project handover.

## 2026-05-19 Handover Branch

- Created branch `codex/handover-gigatok-20260519`.
- Added handover documentation under `docs/handover/`.
- Added `.gitignore` rules to keep checkpoints, caches, reconstruction images, OCR/proxy JSONL files, datasets, and large logs out of Git.
- Added `docs/handover/README_FOR_HANDOVER.md` as the first file for a new maintainer.
- Added `docs/handover/CURRENT_STATE.md` as the current live-checked status snapshot.
- Added `docs/handover/SERVER_QUICKSTART.md` with login, environment, status-check, holdout eval, and OCR readability command templates.

## Current Confirmed State

- New 8-card NPU training wrapper state: `status=done`.
- Main OCR-box proxy training completed at `step=0026250` and saved the final `*_continue_250ep/.../last.pt`.
- GigaTok 5k baseline training completed at `step=0026250` and saved the final `*_continue_250ep/.../last.pt`.
- NPU fixed-path holdout_v2 currently has confirmed `smoke_10` outputs for balanced and medium.
- Full 200-image holdout_v2 eval still needs completion or explicit verification before final reporting.
- 3090 readable50 has basic eval outputs for the three curated sets.
- OCR CER/NED readability for readable50 still needs completion or explicit verification.

## Current Method Reminder

The current main experiment is OCR-box gate + DeepSeek-OCR token-logprob proxy confidence. HR/local/OCR visual-alignment code exists in the repository for historical and ablation work, but those modules are disabled for the current main method.

## Artifact Policy

Git contains code, scripts, configs, and handover docs. Large artifacts remain on the active project servers and are referenced by absolute paths in `docs/handover/`.
