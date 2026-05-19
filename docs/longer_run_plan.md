# Longer Run Plan

## Motivation

当前 500 step readable50 sweep 说明：

- OCR weight 增大后，weighted OCR loss 量级变大；
- raw OCR teacher-forcing CE 没有稳定转化为真实 OCR 可读性提升；
- reconstruction 指标没有改善，`w005` 更差；
- grid 肉眼仍然比较糊。

因此下一步不要先加更大的 OCR weight，也不要同时混多个 ablation。先检查基础 reconstruction 在更长训练下能否达到文字可读门槛。

## Plan

1. **先只长训 OCR-off baseline，不加 OCR 对照。**

   跑 `noocr` 1500 或 2000 step。

   判断目标：

   - readable50 上文字是否开始肉眼可见；
   - `val_mse / val_psnr / val_ssim` 是否继续改善；
   - grid 是否明显比 step500 清晰。

2. **如果 noocr 长训后文字仍然糊，先不要加 OCR。**

   这说明基础 tokenizer reconstruction 还没过文字可读门槛。

   在这种情况下，OCR loss 即使被优化，也很可能只是增加训练干扰。

3. **如果 noocr 长训后文字开始可见，再跑 OCR 0.02 对照。**

   只比较：

   ```text
   noocr long
   w002 long
   ```

   不先跑 `w005`，因为当前 500 step 结果已经显示 `0.05` 有伤重建趋势。

4. **不要先跑 OCR 0.05 长训。**

   0.05 当前表现是：

   - raw OCR CE 更低；
   - reconstruction 更差；
   - post-hoc OCR 没改善。

   所以它不是下一轮优先项。

## Prepared Commands

### 1. noocr 2000-step baseline

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

ASCEND_RT_VISIBLE_DEVICES=0,1 \
SAVE_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_2000 \
RUN_GROUPS="noocr" \
ITERS=2000 \
GLOBAL_BATCH_SIZE=8 \
LOG_EVERY=10 \
VAL_EVERY=100 \
CKPT_EVERY=250 \
RUN_POSTPROCESS=1 \
bash scripts/stage1/ocr_debug/run_ocr_recon_effect_sweep.sh
```

### 2. noocr 1500-step shorter option

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

ASCEND_RT_VISIBLE_DEVICES=0,1 \
SAVE_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_1500 \
RUN_GROUPS="noocr" \
ITERS=1500 \
GLOBAL_BATCH_SIZE=8 \
LOG_EVERY=10 \
VAL_EVERY=100 \
CKPT_EVERY=250 \
RUN_POSTPROCESS=1 \
bash scripts/stage1/ocr_debug/run_ocr_recon_effect_sweep.sh
```

### 3. OCR 0.02 long comparison, only after noocr long improves

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

ASCEND_RT_VISIBLE_DEVICES=0,1 \
SAVE_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w002_2000 \
RUN_GROUPS="w002" \
ITERS=2000 \
GLOBAL_BATCH_SIZE=8 \
LOG_EVERY=10 \
VAL_EVERY=100 \
CKPT_EVERY=250 \
RUN_POSTPROCESS=1 \
bash scripts/stage1/ocr_debug/run_ocr_recon_effect_sweep.sh
```

## Acceptance Check

Long run 后只看：

- `recon.png`
- `grid_500.png / grid_1000.png / grid_1500.png / grid_2000.png`，如果 postprocess 支持对应 step；
- `val_mse / val_psnr / val_ssim`
- post-hoc OCR 只在文字肉眼开始可见后再做。

今晚不自动启动这些命令。
