# HR Project Brief

## Baseline
Use GigaTok B-L (dino disc) as the starting point.

## Why this target
This is a strong discrete tokenizer baseline, and the current goal is to improve decoder-side information utilization without changing the tokenizer encoder or AR model.

## Architecture notes
- GigaTok uses a hybrid CNN + Transformer tokenizer
- 1D tokenizer uses Q-Former
- decoder is intentionally larger than encoder
- first stage should stay at 256x256

## Main implementation target
Add a high-rank regularization term to transformer decoder attention.

## Not the target
- Not the final CNN decoder attention
- Not AR model training
- Not continuous tokenizer
- Not no-tokenizer reconstruction models

## First version training constraints
- decoder-only fine-tuning
- freeze encoder
- freeze quantizer
- freeze codebook
- keep original losses
- add hr_loss as an extra term

## Total loss
L_total = L_original + lambda_hr * L_hr

## Logging
Need to log:
- hr_loss
- selected_layer
- a spectrum uniformity metric
