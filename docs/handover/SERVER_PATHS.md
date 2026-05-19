# Server Paths

Last updated: 2026-05-19

## New 8-Card NPU Server

SSH:

```bash
ssh root@139.9.157.110
docker exec -it node1new bash
```

Project root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss
```

Repo:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation
```

Persist root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/gigatok_persist
```

Logs:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/logs
```

Current 250 epoch training log root:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/logs/continue_250ep_serial_20260518_133045
```

Overnight summary:

```bash
/data2/duoduo_25/yl_GigaTok_Loss/logs/overnight_summary_20260518_141032.md
```

## 8x3090 CUDA Server

SSH:

```bash
ssh ss420f.cs.cityu.edu.hk
```

Project roots used during this work:

```bash
/public/sdc/yl_GigaTok_Loss
/data/duoduo_25/yl_GigaTok_Loss
```

Repo:

```bash
/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation
```

Persist root:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist
```

Python environment used for CUDA eval/OCR:

```bash
/data/duoduo_25/anaconda3/envs/RiemDiff_img/bin/python
```

readable50 eval outputs:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/outputs/overnight_allin/readable50_eval
```

Companion logs:

```bash
/public/sdc/yl_GigaTok_Loss/logs/companion_downstream_20260518_225156
/public/sdc/yl_GigaTok_Loss/logs/repair_text_cache_readable_eval_20260519_003115
```

DeepSeek-OCR:

```bash
/public/sdc/yl_GigaTok_Loss/gigatok_persist/models/DeepSeek-OCR
```

## Old 2-Card NPU Server

SSH alias:

```bash
ssh ModelArts-GPU
```

Typical project root:

```bash
/home/ma-user/work/GigaTok_hr
```

Typical repo:

```bash
/home/ma-user/work/GigaTok_hr/GigaTok_Loss_two_ablation
```

Typical persist root:

```bash
/home/ma-user/work/GigaTok_hr/gigatok_persist
```

This server was used for earlier readable50 and baseline experiments. It may need to be freed for other users. Before deleting anything, verify that required checkpoints/configs have been copied to the new NPU server or 3090 server.

## Do Not Delete Without Confirmation

- Any `last.pt` or `0002500.pt` / `0026250.pt` checkpoint used in reported experiments.
- `text_rich_realworld_upweighted_manifests`.
- `realworld40_5k` text feature cache.
- final OCR bbox/proxy JSONL files.
- readable50 eval outputs and reconstructions.
- overnight logs and summaries.
