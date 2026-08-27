"""Measure minimal-pair cue effects with conversation-trained probes.

All five demographic probe seeds are applied to the same cached hidden states.
The default elicitation suffix is evaluated in every conversation view; three
paraphrases are additionally evaluated in ``current_help``.

Usage:
    uv run scripts/run_conversation_sensitivity.py --batch-size 24
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wgya import config
from wgya.io_utils import file_sha256, read_jsonl, write_results
from wgya.probes import (
    SUFFIX_VARIANTS,
    ProbeSet,
    batch_reading_hidden_states,
    logits_to_probs,
    pick_device,
    probe_logits,
)

STATIC_ATTRIBUTES = ("age", "gender", "education", "socioeco")


def portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(path)


def message_key(messages: list[dict[str, str]]) -> str:
    return json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def suffix_conditions(rows: list[dict]) -> list[dict]:
    conditions = []
    for view in sorted({row["view"] for row in rows}):
        conditions.append({"view": view, "suffix_index": 0})
    for suffix_index in (1, 2, 3):
        conditions.append({"view": "current_help", "suffix_index": suffix_index})
    return conditions


def load_probe_ensemble(root: Path, seeds: list[int]) -> dict[str, ProbeSet]:
    ensemble = {}
    for seed in seeds:
        probe_dir = root / f"seed-{seed}"
        if not probe_dir.exists():
            raise FileNotFoundError(probe_dir)
        ensemble[str(seed)] = ProbeSet(probe_dir)
    models = {probes.meta["model"] for probes in ensemble.values()}
    if len(models) != 1:
        raise ValueError(f"probe seeds target different models: {sorted(models)}")
    for seed, probes in ensemble.items():
        missing = set(STATIC_ATTRIBUTES) - set(probes.probes)
        if missing:
            raise ValueError(f"seed {seed} missing static probes: {sorted(missing)}")
    return ensemble


def load_model(model_name: str, device: str, dtype_name: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    revision = config.model_revision(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if dtype_name == "auto":
        dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float32 if device == "cpu" else torch.float16)
        )
    else:
        dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=dtype,
        device_map=device if device != "cpu" else None,
    )
    model.eval()
    return model, tokenizer, revision, dtype


def reading_record(probe: dict, hidden_states: np.ndarray) -> dict:
    logits = probe_logits(probe, hidden_states[probe["layer"]])
    probs = logits_to_probs(logits)
    return {
        "layer": probe["layer"],
        "logits": [float(value) for value in logits],
        "probs": [float(value) for value in probs],
        "pred": probe["classes"][int(np.argmax(probs))],
    }


def measure_attribute(
    *,
    model,
    tokenizer,
    device: str,
    rows: list[dict],
    conditions: list[dict],
    attr: str,
    probes_by_seed: dict[str, ProbeSet],
    batch_size: int,
) -> dict[tuple[str, int, str], dict[str, dict]]:
    """Return readings keyed by ``(view, suffix_index, message_key)``."""

    outputs = {}
    for condition in conditions:
        view = condition["view"]
        suffix_index = condition["suffix_index"]
        view_rows = [row for row in rows if row["view"] == view]
        messages_by_key = {}
        for row in view_rows:
            for side in ("a", "b"):
                messages = row[f"messages_{side}"]
                messages_by_key.setdefault(message_key(messages), messages)
        keys = sorted(messages_by_key)
        suffix = SUFFIX_VARIANTS[attr][suffix_index]
        label = f"{attr}:{view}:suffix-{suffix_index}"
        for start in tqdm(range(0, len(keys), batch_size), desc=label):
            batch_keys = keys[start : start + batch_size]
            states = batch_reading_hidden_states(
                model,
                tokenizer,
                [messages_by_key[key] for key in batch_keys],
                suffix,
                device,
            )
            for index, key in enumerate(batch_keys):
                outputs[(view, suffix_index, key)] = {
                    seed: reading_record(probes.probes[attr], states[index])
                    for seed, probes in probes_by_seed.items()
                }
    return outputs


def combine_side_readings(
    side_a: dict[str, dict], side_b: dict[str, dict], classes: list[str]
) -> dict:
    seeds = {}
    for seed in sorted(side_a):
        record_a, record_b = side_a[seed], side_b[seed]
        logits_a = np.asarray(record_a["logits"], dtype=float)
        logits_b = np.asarray(record_b["logits"], dtype=float)
        seeds[seed] = {
            "layer": record_a["layer"],
            "logits_a": record_a["logits"],
            "logits_b": record_b["logits"],
            "dlogits": [float(value) for value in logits_b - logits_a],
            "probs_a": record_a["probs"],
            "probs_b": record_b["probs"],
            "pred_a": record_a["pred"],
            "pred_b": record_b["pred"],
            "flipped": record_a["pred"] != record_b["pred"],
        }
    mean_probs_a = np.mean([seed["probs_a"] for seed in seeds.values()], axis=0)
    mean_probs_b = np.mean([seed["probs_b"] for seed in seeds.values()], axis=0)
    return {
        "classes": classes,
        "seeds": seeds,
        "ensemble": {
            "probs_a": [float(value) for value in mean_probs_a],
            "probs_b": [float(value) for value in mean_probs_b],
            "pred_a": classes[int(np.argmax(mean_probs_a))],
            "pred_b": classes[int(np.argmax(mean_probs_b))],
            "flipped": int(np.argmax(mean_probs_a)) != int(np.argmax(mean_probs_b)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=Path,
        default=config.DATA_DIR / "conversation_minimal_pairs.jsonl",
    )
    parser.add_argument(
        "--static-root",
        type=Path,
        default=config.DATA_DIR / "probes",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--limit-pairs",
        type=int,
        default=None,
        help="limit source pair ids for a smoke test; omitted for the full run",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=config.RESULTS_DIR / "llama3b"
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=config.PROJECT_ROOT / "reproducibility" / "study_plan.json",
        help="study plan recorded in the run configuration",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.pairs)
    if args.limit_pairs is not None:
        pair_ids = list(dict.fromkeys(row["pair_id"] for row in rows))[
            : args.limit_pairs
        ]
        rows = [row for row in rows if row["pair_id"] in pair_ids]
    conditions = suffix_conditions(rows)
    static = load_probe_ensemble(args.static_root, args.seeds)
    model_name = next(iter(static.values())).meta["model"]

    device = args.device or pick_device()
    model, tokenizer, revision, dtype = load_model(model_name, device, args.dtype)
    print(
        f"{len(rows)} conversation rows, {len(conditions)} view/suffix conditions; "
        f"model={model_name} device={device} dtype={dtype}"
    )

    readings = {}
    for attr in STATIC_ATTRIBUTES:
        readings[attr] = measure_attribute(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=rows,
            conditions=conditions,
            attr=attr,
            probes_by_seed=static,
            batch_size=args.batch_size,
        )
    result_rows = []
    for condition in conditions:
        view, suffix_index = condition["view"], condition["suffix_index"]
        for row in (candidate for candidate in rows if candidate["view"] == view):
            key_a = message_key(row["messages_a"])
            key_b = message_key(row["messages_b"])
            attrs = {}
            for attr, attr_readings in readings.items():
                probe_set = static["0"]
                attrs[attr] = combine_side_readings(
                    attr_readings[(view, suffix_index, key_a)],
                    attr_readings[(view, suffix_index, key_b)],
                    probe_set.probes[attr]["classes"],
                )
            result_rows.append(
                {
                    **{
                        key: row.get(key)
                        for key in (
                            "conversation_pair_id",
                            "pair_id",
                            "base_id",
                            "topic",
                            "category",
                            "subtype",
                            "tok_delta",
                            "view",
                            "target_message_index",
                            "turns_after_target",
                        )
                    },
                    "suffix_index": suffix_index,
                    "attrs": attrs,
                }
            )

    probe_meta = {
        seed: file_sha256(args.static_root / f"seed-{seed}" / "meta.json")
        for seed in args.seeds
    }
    run_config = {
        "phase": "conversation_sensitivity",
        "model": model_name,
        "model_revision": revision,
        "device": device,
        "dtype": str(dtype),
        "pairs": portable_path(args.pairs),
        "pairs_sha": file_sha256(args.pairs),
        "plan": portable_path(args.plan),
        "plan_sha": file_sha256(args.plan),
        "static_probe_root": portable_path(args.static_root),
        "static_probe_meta_sha_by_seed": probe_meta,
        "views": sorted({row["view"] for row in rows}),
        "suffix_conditions": conditions,
        "batch_size": args.batch_size,
        "n_source_rows": len(rows),
        "n_result_rows": len(result_rows),
        "limit_pairs": args.limit_pairs,
    }
    output = write_results(args.out_dir, "deltas", run_config, result_rows)
    print(f"Wrote {output}")

    by_view = defaultdict(int)
    for row in result_rows:
        by_view[(row["view"], row["suffix_index"])] += 1
    for condition, count in sorted(by_view.items()):
        print(f"  {condition[0]:18s} suffix-{condition[1]}: {count}")


if __name__ == "__main__":
    main()
