# Git and Artifact Policy

Last updated: 2026-05-19

## Git Remote

Current remote:

```bash
origin  https://github.com/mayday4yl/GigaTok_Loss.git
upstream https://github.com/SilentView/GigaTok.git
```

Handover branch:

```bash
codex/handover-gigatok-20260519
```

## Put These In Git

- Source code changes.
- Config YAML files that define experiments.
- Small scripts for training/eval/preflight.
- Handover docs.
- Small CSV/JSON summaries when useful for review.
- `.gitignore`.

## Do Not Put These In Git

- `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`
- text feature caches
- OCR bbox/proxy JSONL files
- reconstruction image folders
- full datasets
- large logs
- temporary review/export folders
- `gigatok_persist/`
- `outputs/`

## Current Local Worktree Warning

At handover time, the local worktree contains many modified and untracked files. Do not commit everything blindly.

Review with:

```bash
git status --short
git diff --stat
git diff -- <path>
```

Suggested commit grouping:

1. Handover docs and `.gitignore`.
2. Core training/eval code changes.
3. Configs and scripts for realworld40_5k.
4. Optional analysis docs.

Avoid committing temporary folders:

```bash
tmp_readable50/
tmp_sequential_5000_review/
tmp_setting_tune_review/
tmp_text_feature_cache_review/
eval_artifacts/
gpt_pro_method_review_excluded/
gpt_pro_method_review_package/
```

## Artifact Index

Use `SERVER_ARTIFACT_INDEX.md`, `SERVER_PATHS.md`, and `EXPERIMENT_STATUS.md` as the artifact index. If a future owner copies artifacts, update those docs with the new path and date.
