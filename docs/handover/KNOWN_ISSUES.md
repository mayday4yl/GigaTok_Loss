# Known Issues

Last updated: 2026-05-19

## 1. Resume Semantics

`--finetune` must not be used for continuation runs. In the training code, finetune mode can reset step accounting and may skip optimizer state semantics.

Safe continuation requires:

- explicit `--vq-ckpt <last.pt>`
- no `--finetune`
- checkpoint contains model and optimizer state
- same config/module switches
- `iterations` set to the total target step count

## 2. Checkpoint Fallback

Do not let a formal continuation silently fall back to:

- initial `VQ_BL256_dino_disc.pt`
- a mismatched checkpoint
- auto-discovered latest checkpoint if the intended checkpoint is missing

Formal scripts should fail fast when the intended checkpoint is absent.

## 3. Proxy Confidence Normalization

Final proxy confidence should use:

```text
normalization = robust_p10_p90_from_mean_logprob
normalization_scope = global_all_valid_boxes_after_merge
```

Formula:

```text
confidence = clamp((mean_logprob - p10_logprob) / (p90_logprob - p10_logprob + 1e-8), 0, 1)
```

Valid boxes only:

- `mean_logprob is not None`
- `token_span is not None`
- `proxy_conf_valid = true`

Invalid boxes:

- `confidence = 0.0`
- `proxy_conf_valid = false`
- excluded from p10/p90

Do not use shard-local normalization, min/max normalization, or fake/default confidence.

## 4. Bbox / Manifest Path Mismatch

The evaluator joins bbox rows by exact `image_path`.

If proxy/bbox JSONL was generated on 3090 and eval runs on NPU, the paths may differ:

```bash
/public/sdc/yl_GigaTok_Loss/...
```

versus:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/...
```

Use path-aligned `.npu_eval.jsonl` copies or regenerate bbox/proxy with the correct final paths.

## 5. readable50 NPU Eval Is Not Required

readable50 ablation should use the 3090 outputs as the authoritative result. NPU readable50 eval was attempted by the overnight supervisor but failed due to path mismatch; that branch is not required for the main ablation story.

## 6. Missing `gigatok_only` readable50 Checkpoint

The same-protocol `gigatok_only` checkpoint for readable50 ablation was not found. Do not substitute:

- pretrained-only GigaTok
- the realworld40 5k baseline

into the readable50 ablation main table unless clearly labeled as a separate reference.

## 7. OCR CER/NED Still Needs Verification

readable50 basic metrics exist, but OCR CER/NED second-pass readability was not confirmed complete at handover time.

Use:

```bash
scripts/stage1/ocr_debug/evaluate_saved_recon_ocr_readability.py
```

on saved `gt_256/` and `reconstructions/` directories.

## 8. SSIM in Training Logs

Training logs may show `Val SSIM: nan`. Do not use those training-log SSIM values as final results. Use the unified eval script, which computes SSIM using `skimage.metrics.structural_similarity` on `[0,1]` images with `data_range=1.0`.

## 9. Large Artifact Handling

Do not put checkpoints, caches, reconstruction images, large JSONL bbox/proxy files, or logs in Git. Record their server paths in handover docs instead.

## 10. Shared Server Safety

The servers are shared. Only modify files under the project-owned roots:

- `/data2/duoduo_25/yl_GigaTok_Loss`
- `/public/sdc/yl_GigaTok_Loss`
- `/data/duoduo_25/yl_GigaTok_Loss`
- `/home/ma-user/work/GigaTok_hr` for old NPU work

Do not delete other users' files. Avoid `rm -rf`; avoid `rsync --delete`.
