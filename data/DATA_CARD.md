# Dataset card

## Summary

The release contains 1,025 controlled minimal pairs over 100 base messages.
Each pair changes one linguistic feature while preserving the underlying
request. The bases cover travel, cooking, small talk, technical support,
health, finance/shopping, and work/study.

| Category | Pairs |
|---|---:|
| Orthography | 184 |
| Explicit disclosure | 111 |
| Emoji | 100 |
| Grammatical complexity | 100 |
| Random-synonym control | 100 |
| Contraction/formality | 94 |
| Slang | 94 |
| Affect language | 82 |
| US-to-UK spelling | 82 |
| Price language | 78 |

The 3B experiment evaluates all 1,025 pairs. A fixed 505-pair panel over 48
base messages is evaluated in 3B, 8B, and 32B models to support like-for-like
cross-model comparisons.

## Files

- `minimal_pairs.jsonl` is the complete source dataset. Rows include pair and
  base identifiers, topic, category, edit description, both message variants,
  and tokenizer-length metadata.
- `conversation_minimal_pairs.jsonl` places every pair in four deterministic
  contexts: two independently worded current-turn shells, one turn later, and
  two turns later.
- `cross_model_minimal_pairs.jsonl` and
  `cross_model_conversation_minimal_pairs.jsonl` contain the shared evaluation
  panel used for all three checkpoints.
- `probes/` contains separate five-seed ensembles of the four demographic
  readouts for Llama 3.2 3B, Llama 3.1 8B, and OLMo 2 32B. Each ensemble
  includes probe metadata, a source-ID manifest, and complete training metrics.

In every rendered conversation, the variants differ at exactly one user turn.
All surrounding user and assistant messages are fixed and identical, so a
generated response cannot mediate the measured effect. The test suite verifies
this invariant for every released row.

## Intended use

The pairs are designed to measure changes in a fixed model readout under
controlled edits. They are not a demographic classifier benchmark and should
not be used to infer attributes of real people.

## Limitations

- The messages are authored rather than sampled from a natural population of
  conversations.
- Probe-training conversations are synthetic and label-conditioned. Held-out
  accuracy establishes that each fitted readout recovers that source task; it
  does not establish demographic validity on natural conversations.
- Some transformations necessarily change length as well as content.
- The conversation shells are deliberately neutral and do not cover the full
  variety of real dialogue.
- Demographic categories are coarse labels inherited from the probe-training
  task, not complete descriptions of identity.
