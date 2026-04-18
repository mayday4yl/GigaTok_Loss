# AGENTS.md

## Current project
Implement high-rank regularization for GigaTok discrete tokenizer fine-tuning.

## Current task scope
Only do stage-1 mainline:
- target tokenizer: GigaTok B-L (dino disc)
- task: add high-rank loss on transformer decoder attention
- do NOT change AR model
- do NOT redesign the whole training framework
- do NOT do ablations yet
- do NOT do full dataset integration in the first coding pass unless explicitly asked

## Key constraints
- Start from the existing GigaTok repo and existing checkpoint setting
- Prefer minimal code diff
- First do decoder-only fine-tuning
- Freeze encoder, quantizer, and codebook in the first version
- Apply high-rank loss only on transformer decoder attention, not on the final CNN decoder
- Use post-softmax attention weights
- Randomly select 1 decoder layer per training step for SVD regularization
- Compute SVD in float32 for stability
- Log hr_loss separately
- Add config switches so the feature can be turned off without editing code

## High-rank loss definition
For attention matrix A:
1. sigma = svdvals(A.float())
2. p = sigma / (sum(sigma) + eps)
3. r = len(sigma)
4. hr_loss = mean((p - 1/r)^2)

Average over batch, heads, and the selected layer.

## Expected workflow
1. First inspect the codebase and write a plan
2. Do not modify code before the plan is shown
3. After implementation, run a small smoke test
4. Show modified files, key diffs, and run commands
5. Keep notes in docs/codex_worklog.md

## Language
- 默认使用中文与我交流。
- 代码注释、提交说明、worklog 默认中文；必要术语可保留英文。

## Workflow
- 对复杂任务先做计划，再实现。
- 未经我明确要求，不要同时做模型改动、数据集接入和全量训练。
- 每个阶段结束后，把结论写入 docs/codex_worklog.md。
- 修改代码后，先做 smoke test，再建议长时间训练。