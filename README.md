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

*Within-model ranks for the 36 primary cells. Ranks avoid treating separately
fitted probe logits as though they shared a calibrated unit.*

## Main finding

The model readouts respond to a stable core of cues, while weaker assignments
vary across checkpoints:

- In the 1,025-pair 3B experiment, 30 of 36 cue-by-readout effects survive
  correction. Twenty-eight exceed the matched lexical-edit floor.
- Disclosure most strongly moves age, grammar and orthography move education,
  price language moves socioeconomic status, and slang moves age.
- On the shared panel, the 8B cue map correlates `.927` with 3B and the 32B map
  correlates `.849`. All 30 corrected 3B effects keep their direction; 29 remain
  corrected-significant in each larger model.
- Emoji are the clearest exception: gender is strongest in the 3B panel, while
  age is strongest in the full 3B experiment, 8B, and 32B.
- Independently worded conversation shells produce `rho=.987-.995`; three
  readout-suffix paraphrases produce `rho=.842-.962` within models.
- Direct disclosure remains readable after later turns. Several surface-style
  effects, especially grammar and orthography, attenuate more quickly.

Selected 3B effects:

| Cue | Readout | Extra movement | 95% CI |
|---|---|---:|---:|
| Explicit disclosure | Age | 1.756 | [1.515, 1.996] |
| Grammatical complexity | Education | 1.458 | [1.238, 1.685] |
| Explicit disclosure | Education | 1.448 | [1.183, 1.734] |
| Price language | Socioeconomic status | 0.915 | [0.777, 1.056] |
| Emoji | Age | 0.565 | [0.479, 0.659] |
| Slang | Age | 0.547 | [0.434, 0.672] |

These are causal changes in fixed TalkTuner-style readouts: each pair differs
at exactly one user turn, and all surrounding dialogue is held constant. They
are not evidence that a demographic label is true, context-invariant, or
appropriate to infer about a real person.

## Design

| Component | Specification |
|---|---|
| Models | Llama 3.2 3B, Llama 3.1 8B, and OLMo 2 32B at pinned revisions |
| Controlled data | 1,025 minimal pairs over 100 bases at 3B; fixed 505-pair cross-model panel |
| Conversation contexts | Two current-turn shells, one turn later, two turns later |
| Readouts | Age, gender, education, and socioeconomic status |
| Probe ensembles | Five separately fitted full-conversation seeds per model |
| Primary estimate | Cue movement minus the matched random-synonym floor |
| Inference | Base-cluster bootstrap and sign flips; BH correction over 36 cells |
| Cross-model endpoints | Cell ranks, signs, corrected significance, dominant readout, retention |
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
