# GigaTok Loss Handover

This branch is for project handover.

Current source of truth:

- `docs/handover/README_FOR_HANDOVER.md`
- `docs/handover/CURRENT_STATE.md`
- `docs/handover/SERVER_QUICKSTART.md`
- `docs/handover/HANDOVER.md`

Current main experiment:

- text-rich GigaTok discrete tokenizer fine-tuning
- `realworld40_5k`
- main method: OCR-box gate + DeepSeek-OCR token-logprob proxy confidence
- baseline: GigaTok-only tokenizer baseline on the same 5k data
- current primary servers:
  - new 8-card NPU: `/data2/duoduo_25/yl_GigaTok_Loss`
  - 8x3090 CUDA: `/public/sdc/yl_GigaTok_Loss` and `/data/duoduo_25/yl_GigaTok_Loss`

Do not use older Text-HR notes or SVGs as the current method description. They are historical context only.
