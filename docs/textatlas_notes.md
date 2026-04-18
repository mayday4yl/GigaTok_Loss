# TextAtlas5M Notes for Stage-1

## Goal
Use TextAtlas5M only after the HR-loss model code path is stable.

## First-stage subset choice
Use only:
- CleanTextSynth
- StyledTextSynth
- LongWordsSubset-M
- TextScenesHQ

Do not use in the first round:
- PPT2Structured
- Paper2Text
- TextVisionBlend

## Preprocess
- keep aspect ratio
- resize to fit within 256x256
- pad to 256x256
- do not center crop text-heavy images

## Suggested sample scale
- smoke test: 5k to 10k images
- first formal run: about 200k images total
