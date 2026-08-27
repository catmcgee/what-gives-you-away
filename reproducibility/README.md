# Reproducibility

The repository releases the study plans, immutable model revisions, processed
inputs, three probe ensembles, row-level deltas, and aggregate results.

## Environments

Local analysis uses the Python environment locked in `uv.lock`. The full 3B
run used an RTX 4090 with float16 inference; details are in
[`llama3b_environment.json`](llama3b_environment.json). The shared-panel 3B
run used an RTX 5090 with float16 inference; details are in
[`cross_model_llama3b_environment.json`](cross_model_llama3b_environment.json).
The 8B and 32B runs used one H100 NVL with bfloat16 inference; details are in
[`cross_model_environment.json`](cross_model_environment.json).

## Reproduce the released analysis

```bash
uv sync --group dev
uv run pytest
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

These commands rebuild the 4,100 conversation rows, recompute every
statistical summary from the 17,780 released pair-by-condition rows, assess the
probe gates, and regenerate both paper figures without downloading a model.

## Repeat the model runs

The runners infer the model from each probe ensemble and resolve its pinned
revision from `wgya/config.py`.

```bash
# Full Llama 3.2 3B experiment
uv run scripts/run_conversation_sensitivity.py \
  --pairs data/conversation_minimal_pairs.jsonl \
  --static-root data/probes \
  --out-dir results/llama3b \
  --batch-size 32 \
  --dtype float16 \
  --plan reproducibility/study_plan.json

# Shared-panel Llama 3.2 3B
uv run scripts/run_conversation_sensitivity.py \
  --pairs data/cross_model_conversation_minimal_pairs.jsonl \
  --static-root data/probes \
  --out-dir results/cross_model/llama3b \
  --batch-size 32 \
  --dtype float16 \
  --plan reproducibility/cross_model_plan.json

# Shared-panel Llama 3.1 8B
uv run scripts/run_conversation_sensitivity.py \
  --pairs data/cross_model_conversation_minimal_pairs.jsonl \
  --static-root data/probes/llama8b \
  --out-dir results/cross_model/llama8b \
  --batch-size 32 \
  --dtype bfloat16 \
  --plan reproducibility/cross_model_plan.json

# Shared-panel OLMo 2 32B
uv run scripts/run_conversation_sensitivity.py \
  --pairs data/cross_model_conversation_minimal_pairs.jsonl \
  --static-root data/probes/olmo32b \
  --out-dir results/cross_model/olmo32b \
  --batch-size 16 \
  --dtype bfloat16 \
  --plan reproducibility/cross_model_plan.json
```

The complete 32B run used an accelerator with about 94 GB of memory under this
loading strategy. Smaller batches reduce activation memory but not the model's
weight footprint.

## Retrain the probes

Probe retraining requires an authorized local checkout of the TalkTuner source
conversation dataset. Run the following once per model, changing the cache
path to a scratch location with adequate space:

```bash
# Llama 3.2 3B
uv run scripts/train_probes.py \
  --dataset-root /path/to/TalkTuner/data/dataset \
  --model unsloth/Llama-3.2-3B-Instruct \
  --out-root data/probes \
  --cache-dir /scratch/wgya/3b \
  --dtype float16

# Llama 3.1 8B
uv run scripts/train_probes.py \
  --dataset-root /path/to/TalkTuner/data/dataset \
  --model unsloth/Meta-Llama-3.1-8B-Instruct \
  --out-root data/probes/llama8b \
  --cache-dir /scratch/wgya/8b \
  --dtype bfloat16

# OLMo 2 32B
uv run scripts/train_probes.py \
  --dataset-root /path/to/TalkTuner/data/dataset \
  --model allenai/OLMo-2-0325-32B-Instruct \
  --out-root data/probes/olmo32b \
  --cache-dir /scratch/wgya/32b \
  --dtype bfloat16 \
  --batch-size 2
```

Rows sharing a TalkTuner generator folder and numeric source index remain
together in 70/15/15 train/layer-selection/test partitions. The final split is
never used for layer selection. Every released seed for every readout exceeds
the `.75` held-out balanced-accuracy threshold specified in
[`cross_model_plan.json`](cross_model_plan.json).

## Artifact integrity

The hashes below cover the principal released inputs and result artifacts.
Recompute them with `shasum -a 256 <path>`.

<!-- ARTIFACT_HASHES_START -->

| Artifact | SHA-256 |
|---|---|
| `data/minimal_pairs.jsonl` | `ce97b0751cf5b10957a27dfdf813a398387641c3bbcafad8f5daebcb35c86ba4` |
| `data/conversation_minimal_pairs.jsonl` | `1ad854377c6b5c241fc465a89f32ba71c92d488b67acd491742949fa2003ee8b` |
| `data/cross_model_minimal_pairs.jsonl` | `b0fabd3f9734d2a04d3cc0169fd9ac7ce3801e9bfad17989510ec691e808a035` |
| `data/cross_model_conversation_minimal_pairs.jsonl` | `4dc275e0dc1cb2143e75b4a9376c7bb70e6fdda57563c69394bbad618a219804` |
| `data/probes/training_report.json` | `e2f7b80ca6208856fab2c5172bbabbc12904c4975c2be53c0c97c3b9c41c32a2` |
| `data/probes/llama8b/training_report.json` | `dfb4659403ea046352418c520b7a6513a000e6dc8248db1a7f9ee67d2cfb306d` |
| `data/probes/olmo32b/training_report.json` | `dfce9884bd3e27137c679cc7993cc61ba916d949e4311ecac30eee1d1a18c6ed` |
| `results/llama3b/deltas_d61d45c5c48f.json.gz` | `06f8046e33b499df8e40da2231962502d90b056d10f4698512dc0164254657f2` |
| `results/llama3b/summary_d61d45c5c48f.json` | `ba8d3c68a7c5eae51b73dbe9ed449e36802a4d466af79f36922a3e8dfa0ac685` |
| `results/cross_model/llama3b/deltas_302951caa648.json.gz` | `e91733feca1586c0ccd69459cfc232bf8818257526d3ffd3df91953d14e442eb` |
| `results/cross_model/llama3b/summary_302951caa648.json` | `1090a112417092170e6e86471b040bd1e46ff9a73870280f704a15baf3bbdbef` |
| `results/cross_model/llama8b/deltas_e7901d719f28.json.gz` | `1caddd0e69dc0467ce1e982072b9ba7ce009f26e0b01dea3dfb75d2b9e6c89d8` |
| `results/cross_model/llama8b/summary_e7901d719f28.json` | `acbf580aca43caca396506448e94054976b43a11f96e9e5ae20b9c548d7449cf` |
| `results/cross_model/olmo32b/deltas_90c18609ce4f.json.gz` | `a8b8175653ed1606327e0487750084efc54a615f7f5cecbe7ca2bf7a32b51b74` |
| `results/cross_model/olmo32b/summary_90c18609ce4f.json` | `e87a9e7d1ab452143a35cf48963fbeb95ed27ba98f01ccb727877b0c2ba40c52` |
| `results/cross_model/summary.json` | `01aba1e3a7f64e5bf299007e1ac1fb59155cbc13156d44be127b23a0daa44cdd` |
| `reproducibility/study_plan.json` | `c7e97625eb94c711e9767091eb935fd3c107962937505920800f1c34ec09c538` |
| `reproducibility/cross_model_plan.json` | `25591b2972bb1a8381be7ba748943d6ae94b5424a7da88b300e92e182d01d4cd` |
| `reproducibility/llama3b_environment.json` | `9bc653269800820b046b9c86eda6f14221835ba2ce112c854fcdbe2a43312559` |
| `reproducibility/cross_model_llama3b_environment.json` | `70d9ee2eb046df1e62d78695c358430859e0ceda57d6f52e88aea1c0ddb21911` |
| `reproducibility/cross_model_environment.json` | `51eafc94a1b2141af403ff5ac551cf9ea766d3d23cc6753472b9561245164c4f` |
| `paper/main.tex` | `1df923b07ec52833ebcbbea9f323fc46f84fe372c50be8b5bef4fdced7bb45e9` |
| `paper/refs.bib` | `44da042eae4b5de55c6c3ca78a3f928bca828bc2d5465b5179ef64ed1c1daacf` |
| `paper/fig_conversation_sensitivity.pdf` | `dbf95bac64f9aeb30ca8e66f510aee200fe95e9ee4470ce8634590bb5af3a607` |
| `paper/fig_cross_model.pdf` | `8807fd7a4c25c5465ab77d7961159432e9b81f2d246bf3238c5a7cccc2fd6904` |
| `paper/main.pdf` | `da75261abfbbaec7cfb2b8d87a9138013edc4f83306307cc2c25279208f65c50` |

<!-- ARTIFACT_HASHES_END -->
