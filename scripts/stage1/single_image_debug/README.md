# 单图测试工具

这个目录只放 stage-1 单图过拟合和诊断代码，不改主训练框架。

目标：
- 用一张图确认 decoder-only text-HR fine-tune 能不能把 loss 降下来。
- 对指定 checkpoint 做逐层 decoder cross-attention 诊断。
- 保存每层 SVD 奇异值、注意力质量、特征统计和重建图，便于排查 HR 分支是否真的生效。

## 1. 生成三张单图 manifest

默认目标来自当前 train manifest：
- `dense`: `CleanTextSynth:train:2731`
- `medium`: `TextScenesHQ:train:2803`
- `sparse`: `LongWordsSubset-A:train:3149`

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

python scripts/stage1/single_image_debug/make_single_image_manifests.py \
  --source-manifest /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/train_materialized_manifest_v2text.jsonl \
  --output-dir /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/manifests
```

输出：
- `dense.jsonl`
- `medium.jsonl`
- `sparse.jsonl`
- `summary.json`

## 2. 单图 overfit

先跑 dense 的 HR：

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

MODE=hr \
TAG=dense \
ITERS=1000 \
ASCEND_RT_VISIBLE_DEVICES=0 \
bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
```

再跑同一张图的 text baseline，用来判断 HR 有没有额外引入问题。注意：这个 baseline 仍然有 text conditioning，只是不开 HR loss：

```bash
MODE=baseline \
TAG=dense \
ITERS=1000 \
ASCEND_RT_VISIBLE_DEVICES=0 \
bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
```

再跑同一张图的 original baseline。这个模式不加 T5、不加 text projection、不加 decoder text cross-attention、不加 HR，用来代表原生 GigaTok tokenizer：

```bash
MODE=original \
TAG=dense \
ITERS=1000 \
ASCEND_RT_VISIBLE_DEVICES=0 \
bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
```

然后可以换图：

```bash
MODE=hr TAG=medium ITERS=1000 ASCEND_RT_VISIBLE_DEVICES=0 bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
MODE=hr TAG=sparse ITERS=1000 ASCEND_RT_VISIBLE_DEVICES=0 bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
MODE=original TAG=medium ITERS=1000 ASCEND_RT_VISIBLE_DEVICES=0 bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
MODE=original TAG=sparse ITERS=1000 ASCEND_RT_VISIBLE_DEVICES=0 bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
```

默认 config 使用仓库内文件：
- HR: `configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml`
- text baseline: `configs/vq/VQ_BL256_dino_disc_text_baseline_v2.yaml`
- original baseline: `configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml`

如果服务器只能使用本地 T5 路径，显式传入服务器上的 local config：

```bash
CONFIG=configs/vq/_pilot_text_hr_local.yaml MODE=hr TAG=dense bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
CONFIG=configs/vq/_pilot_text_baseline_local.yaml MODE=baseline TAG=dense bash scripts/stage1/single_image_debug/run_single_image_overfit.sh
```

`MODE=original` 不依赖 T5，所以通常不需要服务器 local T5 config。

重点看训练日志里的：
- `rec_loss`
- `direct_rec_loss`
- `feature_rec_loss`
- `Val MSE`
- `Val MAE`
- `Val PSNR`
- HR 模式额外看：
  - `text_hr_loss`
  - `text_hr_sigma_mean`
  - `selected_decoder_layer`
  - `selected_text_layer`

## 3. 逐层 SVD 和特征诊断

这个诊断脚本只适用于 `MODE=hr` 和 `MODE=baseline`，因为它需要 decoder text cross-attention。`MODE=original` 没有 text cross-attention，不能做 text attention SVD；original 只比较重建日志和保存图像即可。

对训练前 checkpoint 诊断。脚本默认扫 24 个 decoder layer；这里显式写出 `--layer-mode all_decoder`，避免和只看配置层的快速检查混淆：

```bash
cd /home/ma-user/work/GigaTok_hr/GigaTok_Loss

python scripts/stage1/single_image_debug/diagnose_single_image.py \
  --manifest-jsonl /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/manifests/dense.jsonl \
  --config configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml \
  --ckpt /home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt \
  --output-dir /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/diagnostics/dense_pretrain \
  --device-backend npu \
  --mixed-precision bf16 \
  --layer-mode all_decoder \
  --text-layer-mode same \
  --seed 0
```

对单图训练后的 checkpoint 诊断：

```bash
python scripts/stage1/single_image_debug/diagnose_single_image.py \
  --manifest-jsonl /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/manifests/dense.jsonl \
  --config configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml \
  --ckpt /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/dense_hr_1img_1000step/checkpoints/last.pt \
  --output-dir /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug/diagnostics/dense_after_1000 \
  --device-backend npu \
  --mixed-precision bf16 \
  --layer-mode all_decoder \
  --text-layer-mode same \
  --seed 0
```

如果诊断官方原始 checkpoint，`text_projection` 和 `text_type_embedding` 不在原始 checkpoint 里，会按当前 config 随机初始化。`--seed 0` 用来固定这部分初始化。更严格的做法是诊断单图训练刚开始保存的早期 checkpoint。

输出：
- `per_layer_metrics.csv`
- `summary.json`
- `target_256.png`
- `recon_layerXX.png`
- `diff_layerXX.png`
- `singular_values_layerXX.npy`

`layer-mode=config_pairs` 只诊断训练配置里的 layer pairs，例如 8-15。若只想快速检查训练真正使用的层，可以这样跑：

```bash
python scripts/stage1/single_image_debug/diagnose_single_image.py \
  ... \
  --layer-mode config_pairs
```
