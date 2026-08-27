# What Gives You Away?

[![tests](https://github.com/catmcgee/what-gives-you-away/actions/workflows/test.yml/badge.svg)](https://github.com/catmcgee/what-gives-you-away/actions/workflows/test.yml)
[![paper](https://img.shields.io/badge/paper-PDF-b31b1b.svg)](paper/main.pdf)
[![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB.svg)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Which parts of a conversation make a language model read its user as older,
more educated, wealthier, or a different gender?** This repository contains a
controlled, context-matched study of how linguistic edits move user-attribute
readouts in 3B, 8B, and 32B instruction models. The main 3B experiment uses
1,025 pairs over 100 base conversations. A fixed 505-pair panel is evaluated in
all three models for like-for-like comparison.

[Read the paper](paper/main.pdf) ·
[Explore the results](results/README.md) ·
[Reproduce the analysis](reproducibility/README.md) ·
[Inspect the dataset](data/DATA_CARD.md)

![Conversation cue map and persistence results](results/fig_conversation_sensitivity.png)

*The 1,025-pair 3B cue map and temporal persistence. Bold heatmap cells survive
correction across 36 comparisons.*

![Cross-model cue-map agreement](results/fig_cross_model.png)

*Cue ranks computed separately within each readout for the 36 primary cells.
This avoids treating separately fitted probe logits as though they shared a
calibrated unit.*

## Main finding

The readouts respond to a repeatable ordering of cues, with some differences
across checkpoints:

- In the 1,025-pair 3B experiment, 30 of 36 cue-by-readout effects survive
  correction. Twenty-eight exceed the matched lexical-edit floor.
- Within the age readout, personal context and slang rank highly. Grammar and
  personal context lead education; price and personal context lead
  socioeconomic status.
- On the shared panel, pooled within-readout cue-rank correlations are `.938`
  for 8B against 3B and `.850` for 32B against 3B. All 30 corrected 3B effects
  keep their sign; 29 remain corrected-significant in each larger model.
- The gender ordering varies more: emoji ranks first at 3B and third in both
  larger models, behind personal context and slang.
- Independently worded conversation shells produce `rho=.967-.992`; three
  readout-phrase paraphrases produce `rho=.875-.933` within models.
- Personal-context age movement persists after later turns. Several style
  effects, especially grammar and orthography, attenuate more quickly.

Selected 3B effects:

| Cue | Readout | Extra movement | 95% CI |
|---|---|---:|---:|
| Personal context | Age | 1.756 | [1.517, 2.003] |
| Emoji | Age | 0.565 | [0.479, 0.655] |
| Personal context | Gender | 0.625 | [0.492, 0.767] |
| Emoji | Gender | 0.466 | [0.348, 0.600] |
| Grammatical complexity | Education | 1.458 | [1.233, 1.680] |
| Personal context | Education | 1.448 | [1.185, 1.726] |
| Price language | Socioeconomic status | 0.915 | [0.774, 1.053] |
| Personal context | Socioeconomic status | 0.860 | [0.721, 1.005] |

Each pair differs at one user turn and holds the surrounding dialogue fixed.
The values compare cues within the same readout; separate probes do not share a
logit scale. The outcome measures absolute movement, not whether a class moved
toward older, younger, or another direction. It is not evidence that a
demographic label is true or appropriate to infer about a person.

## Design

| Component | Specification |
|---|---|
| Models | Llama 3.2 3B, Llama 3.1 8B, and OLMo 2 32B at pinned revisions |
| Controlled data | 1,025 minimal pairs over 100 bases at 3B; deterministic 505-pair cross-model panel |
| Conversation contexts | Two current-turn shells, one turn later, two turns later |
| Readouts | Age, gender, education, and socioeconomic status |
| Probe ensembles | Five separately fitted full-conversation seeds per model |
| Primary estimate | Cue movement minus the matched random-synonym floor |
| Inference | Base-cluster bootstrap and sign flips; BH correction over 36 cells |
| Cross-model endpoints | Within-readout cue ranks, signs, corrected significance, retention |
| Within-model robustness | Shell wording, three suffix paraphrases, probe seeds, temporal delay |

The probes were trained on complete conversations, so the experiment also uses
complete conversations. Assistant messages are fixed across each pair;
generated replies cannot become an uncontrolled mediator. Probe seeds and
conversation shells are averaged as repeated measurements rather than counted
as independent observations. Every one of the 60 held-out probe tests passes
the `.75` balanced-accuracy gate.

## Repository

```text
paper/                  manuscript source and compiled paper
data/                   controlled pairs and three five-seed probe ensembles
results/                per-model deltas, summaries, and publication figures
reproducibility/        study plans, environments, and artifact hashes
scripts/                build, train, run, analyze, and plot entry points
tests/                   dataset, design, split, metric, and inference checks
wgya/                    probe, metric, I/O, and statistical utilities
```

Every reported number can be recomputed from the released compressed deltas
without downloading model weights or using a GPU.

The arXiv source is flat and self-contained in [`paper/`](paper/). Upload
`main.tex`, `refs.bib`, `fig_conversation_sensitivity.pdf`, and
`fig_cross_model.pdf`; arXiv should treat `main.tex` as the top-level file.

## Reproduce

Install the locked environment and run the checks:

```bash
uv sync --group dev
uv run pytest
```

Rebuild the conversation views, aggregate every released run, and regenerate
the figures:

```bash
uv run scripts/build_conversation_pairs.py

uv run scripts/analyze_conversation_sensitivity.py \
  results/llama3b/deltas_*.json.gz
uv run scripts/analyze_conversation_sensitivity.py \
  results/cross_model/llama3b/deltas_*.json.gz
uv run scripts/analyze_conversation_sensitivity.py \
  results/cross_model/llama8b/deltas_*.json.gz
uv run scripts/analyze_conversation_sensitivity.py \
  results/cross_model/olmo32b/deltas_*.json.gz

uv run scripts/fig_conversation_sensitivity.py \
  results/llama3b/summary_*.json
uv run scripts/analyze_cross_model.py \
  results/cross_model/llama3b/summary_*.json \
  results/cross_model/llama8b/summary_*.json \
  results/cross_model/olmo32b/summary_*.json \
  --training-reports data/probes/training_report.json \
    data/probes/llama8b/training_report.json \
    data/probes/olmo32b/training_report.json \
  --output results/cross_model/summary.json
uv run scripts/fig_cross_model.py \
  results/cross_model/llama3b/summary_*.json \
  results/cross_model/llama8b/summary_*.json \
  results/cross_model/olmo32b/summary_*.json
```

Repeat the full 3B model run on a CUDA GPU:

```bash
uv run scripts/run_conversation_sensitivity.py \
  --pairs data/conversation_minimal_pairs.jsonl \
  --static-root data/probes \
  --out-dir results/llama3b \
  --batch-size 32 \
  --dtype float16 \
  --plan reproducibility/study_plan.json
```

Probe retraining requires an authorized local copy of the TalkTuner source
conversations. Full commands, artifact hashes, and environment details are in
[`reproducibility/README.md`](reproducibility/README.md).

## Citation

Citation metadata are available in [`CITATION.cff`](CITATION.cff). Until the
paper has an archival identifier:

```bibtex
@misc{mcgee2026whatgivesyouaway,
  author = {Cat McGee},
  title = {What Gives You Away? How Linguistic Cues Move User Readouts Across Three Language Models},
  year = {2026},
  url = {https://github.com/catmcgee/what-gives-you-away}
}
```

Repository code and controlled-pair data are MIT licensed. Model weights and
source probe-training conversations remain under their upstream terms; see
[`data/THIRD_PARTY_NOTICES.md`](data/THIRD_PARTY_NOTICES.md).
