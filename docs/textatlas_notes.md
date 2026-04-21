# TextAtlas5M Notes for Stage-1

## Goal
Use a fixed local TextAtlas subset to compare baseline vs HR loss for tokenizer text reconstruction.

## Current stage-1 subset choice
The current fixed-count text reconstruction validation run uses:
- CleanTextSynth
- StyledTextSynth
- TextVisionBlend
- TextScenesHQ
- LongWordsSubset-A

TextScenesHQ uses `40,000` train images. The other four subsets use `50,000`
train images each. This stage no longer follows the original TextAtlas row
distribution.

Do not use in this round:
- PPT2Structured
- Paper2Text
- CoverBook
- PPT2Details

## Preprocess
- keep aspect ratio
- resize to fit within 256x256
- pad to 256x256
- do not center crop text-heavy images

## Suggested sample scale
- formal run: fixed local materialized set, `40k TextScenesHQ + 4 * 50k other train`
- validation: balanced 2k images per subset
- optional hold-out: source-only 500 images per subset
