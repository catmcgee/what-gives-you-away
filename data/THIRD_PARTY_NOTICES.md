# Third-party notices

The MIT license covers the repository code and controlled-pair data.
It does not relicense model weights or source conversations.

- **TalkTuner training conversations.** MIT licensed, copyright Yida Chen.
  Probe training uses source revision
  `e0b97f1c6e8b75a976ece7dec829acb1b2f57e06`. The conversations are not
  redistributed; `probes/source_manifest.jsonl` records only source IDs and
  labels.
- **Llama 3.2.** Reference-model weights are downloaded from
  `unsloth/Llama-3.2-3B-Instruct` at revision
  `006f5dcd1393c3add266de40994ba96225e9689d` and remain subject to the Llama
  3.2 Community License. No base-model weights are included here.
- **Llama 3.1.** Model weights are downloaded from
  `unsloth/Meta-Llama-3.1-8B-Instruct` at revision
  `a2856192dd7c25b842431f39c179a6c2c2f627d1` and remain subject to the Llama
  3.1 Community License.
- **OLMo 2.** Model weights are downloaded from
  `allenai/OLMo-2-0325-32B-Instruct` at revision
  `b96024342a77a69aa0dda815c3454a671f477463` and remain subject to the Apache
  License 2.0.

The released probe files contain only fitted linear coefficients and metadata.
