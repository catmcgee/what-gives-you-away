"""Train the five-seed full-conversation probe ensemble.

Rows sharing a TalkTuner generator folder and numeric source index remain
together in 70/15/15 train/layer-selection/test partitions. Source
conversations are read from an authorized TalkTuner checkout and are not
redistributed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wgya.config import model_revision
from wgya.io_utils import write_jsonl
from wgya.metrics import classification_metrics
from wgya.probes import (
    ATTRIBUTES,
    READING_SUFFIX,
    ProbeSet,
    batch_reading_hidden_states,
    pick_device,
)

FOLDER_ATTR = {
    "age": "age",
    "gender": "gender",
    "education_three_classes": "education",
    "socioeconomic": "socioeco",
}
FILE_RE = re.compile(r"conversation_(\d+)_([a-z]+)_([a-z ]+)\.txt$")
SPEAKER_RE = re.compile(r"^(?:###\s*)?(HUMAN|Human|ASSISTANT|Assistant):\s*(.*)$")
TALKTUNER_REVISION = "e0b97f1c6e8b75a976ece7dec829acb1b2f57e06"


def parse_conversation(text):
    messages, role, buffer = [], None, []

    def flush():
        if role is not None:
            content = "\n".join(buffer).strip()
            if content:
                messages.append({"role": role, "content": content})

    for line in text.splitlines():
        match = SPEAKER_RE.match(line)
        if match:
            flush()
            role = "user" if match.group(1).lower() == "human" else "assistant"
            buffer = [match.group(2).strip()]
        elif role is not None:
            buffer.append(line)
    flush()
    return messages


def load_rows(dataset_root, attr, per_class, seed):
    # Every generator folder reuses numeric source indices across labels. Keep
    # each label-complete index set together so no generator/index signature
    # can cross partitions, even though the generated dialogues are not
    # literal minimal-pair versions of one scenario.
    by_group = defaultdict(dict)
    for folder in sorted(Path(dataset_root).iterdir()):
        if not folder.is_dir():
            continue
        match = re.match(r"(?:llama|openai)_(.+?)_\d+$", folder.name)
        if not match or FOLDER_ATTR.get(match.group(1)) != attr:
            continue
        for path in sorted(folder.glob("*.txt")):
            if path.name.startswith("._"):
                continue
            file_match = FILE_RE.fullmatch(path.name)
            if (
                not file_match
                or file_match.group(2) != attr
                or file_match.group(3) not in ATTRIBUTES[attr]
            ):
                continue
            messages = parse_conversation(
                path.read_text(encoding="utf-8", errors="replace")
            )
            if messages:
                group_id = f"{folder.name}/conversation_{file_match.group(1)}"
                by_group[group_id][file_match.group(3)] = {
                    "id": path.relative_to(dataset_root).as_posix(),
                    "group_id": group_id,
                    "messages": messages,
                    "label": file_match.group(3),
                }
    complete = [
        group for group in by_group.values() if set(group) == set(ATTRIBUTES[attr])
    ]
    rng = random.Random(seed)
    rng.shuffle(complete)
    if len(complete) < per_class:
        raise ValueError(
            f"{attr}: requested {per_class} complete groups, found {len(complete)}"
        )
    rows = [
        group[label] for group in complete[:per_class] for label in ATTRIBUTES[attr]
    ]
    rng.shuffle(rows)
    return rows


def grouped_split(rows, seed):
    groups = np.asarray(sorted({row["group_id"] for row in rows}))
    train_val_groups, test_groups = train_test_split(
        groups, test_size=0.15, random_state=seed
    )
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=0.17647058823529413,
        random_state=seed,
    )
    memberships = {
        "train": set(train_groups),
        "selection": set(val_groups),
        "test": set(test_groups),
    }
    indices = {
        name: np.asarray(
            [i for i, row in enumerate(rows) if row["group_id"] in selected]
        )
        for name, selected in memberships.items()
    }
    assert not (memberships["train"] & memberships["selection"])
    assert not (memberships["train"] & memberships["test"])
    assert not (memberships["selection"] & memberships["test"])
    return indices["train"], indices["selection"], indices["test"]


def cache_key(model, attr, rows, activation_dtype):
    payload = {
        "model": model,
        "attr": attr,
        "context": "full-conversation",
        "suffix": READING_SUFFIX[attr],
        "ids": [row["id"] for row in rows],
        "extractor": 4,
        "activation_dtype": activation_dtype,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def row_label(row, attr):
    return row.get("label", row.get(attr, row.get("labels", {}).get(attr)))


def extract_features(model, tokenizer, device, rows, attr, path, batch_size=1):
    from tqdm import tqdm

    features = []
    for start in tqdm(range(0, len(rows), batch_size), desc=f"extract {attr}"):
        chunk = rows[start : start + batch_size]
        suffix = batch_reading_hidden_states(
            model,
            tokenizer,
            [row["messages"] for row in chunk],
            READING_SUFFIX[attr],
            device,
        )
        features.extend(suffix.astype(np.float16))
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [row_label(row, attr) for row in rows]
    if any(label is None for label in labels):
        raise ValueError(f"missing {attr} label in activation rows")
    # Hidden states are high-entropy floats and compress slowly. Uncompressed
    # NPZ avoids unnecessary GPU idle time during feature extraction.
    np.savez(
        path,
        features=np.stack(features),
        labels=np.asarray(labels),
        ids=np.asarray([row["id"] for row in rows]),
    )


def ordered_probs(clf, x, classes):
    raw = clf.predict_proba(x)
    order = [list(clf.classes_).index(cls) for cls in classes]
    return raw[:, order]


def fit_layers(x, y, train_idx, val_idx, test_idx, classes, layer_range):
    curve = []
    best = None
    for layer in layer_range:
        selector = RidgeClassifier(alpha=10.0, solver="lsqr", tol=1e-3)
        selector.fit(x[train_idx, layer].astype(np.float32), y[train_idx])
        predicted = selector.predict(x[val_idx, layer].astype(np.float32))
        hard_probs = np.asarray(
            [[float(prediction == cls) for cls in classes] for prediction in predicted]
        )
        selection_metrics = classification_metrics(y[val_idx], hard_probs, classes)
        curve.append({"layer": layer, "selector": "ridge_lsqr", **selection_metrics})
        score = (
            selection_metrics["macro_f1"],
            selection_metrics["balanced_accuracy"],
            -layer,
        )
        if best is None or score > best[0]:
            best = (score, layer)
    _, layer = best
    clf = LogisticRegression(max_iter=500, tol=1e-3, C=0.1, solver="lbfgs")
    clf.fit(x[train_idx, layer].astype(np.float32), y[train_idx])
    val_probs = ordered_probs(clf, x[val_idx, layer].astype(np.float32), classes)
    val_metrics = classification_metrics(y[val_idx], val_probs, classes)
    test_probs = ordered_probs(clf, x[test_idx, layer].astype(np.float32), classes)
    return (
        layer,
        clf,
        val_metrics,
        classification_metrics(y[test_idx], test_probs, classes),
        curve,
    )


def save_probe(path, layer, clf, classes, val_metrics, test_metrics):
    if len(classes) > 2:
        order = [list(clf.classes_).index(cls) for cls in classes]
        coef, intercept = clf.coef_[order], clf.intercept_[order]
    else:
        coef, intercept = clf.coef_, clf.intercept_
    np.savez(
        path,
        layer=layer,
        val_acc=val_metrics["accuracy"],
        test_acc=test_metrics["accuracy"],
        classes=np.asarray(classes),
        coef=coef,
        intercept=intercept,
    )


def text_controls(rows, y, train_idx, test_idx, classes, seed):
    texts = [
        "\n".join(m["content"] for m in row["messages"] if m["role"] == "user")
        for row in rows
    ]
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), min_df=2, max_features=50000
    )
    train_x = vectorizer.fit_transform([texts[i] for i in train_idx])
    test_x = vectorizer.transform([texts[i] for i in test_idx])
    clf = LogisticRegression(max_iter=500, tol=1e-3, C=1.0)
    clf.fit(train_x, y[train_idx])
    metrics = classification_metrics(
        y[test_idx], ordered_probs(clf, test_x, classes), classes
    )

    rng = np.random.default_rng(seed)
    shuffled = y[train_idx].copy()
    rng.shuffle(shuffled)
    shuffle_clf = LogisticRegression(max_iter=500, tol=1e-3, C=1.0)
    shuffle_clf.fit(train_x, shuffled)
    shuffled_metrics = classification_metrics(
        y[test_idx], ordered_probs(shuffle_clf, test_x, classes), classes
    )
    return {"char_tfidf": metrics, "shuffled_label_char_tfidf": shuffled_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--attrs",
        nargs="+",
        choices=["age", "gender", "education", "socioeco"],
        default=["age", "gender", "education", "socioeco"],
    )
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "data" / "cache" / "activations"
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "probes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--source-revision", default=TALKTUNER_REVISION)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.extract_only and args.fit_only:
        parser.error("--extract-only and --fit-only are mutually exclusive")
    if args.smoke:
        args.per_class = min(args.per_class, 8)
        args.split_seeds = args.split_seeds[:1]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    if args.dtype == "auto":
        dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else (torch.float32 if device == "cpu" else torch.float16)
        )
    else:
        dtype = getattr(torch, args.dtype)
    revision = model_revision(args.model)
    model = tokenizer = None
    if not args.fit_only:
        print(f"loading {args.model}@{revision} on {device} as {dtype}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=revision,
            dtype=dtype,
            device_map="auto"
            if device == "cuda"
            else (device if device != "cpu" else None),
        )
        model.eval()

    slug = args.model.split("/")[-1].lower()
    base_out = args.out_root
    aggregate = {
        "model": args.model,
        "model_revision": revision,
        "context": "full-conversation",
        "sample_seed": args.sample_seed,
        "split_seeds": args.split_seeds,
        "per_class": args.per_class,
        "source": "yc015/TalkTuner-chatbot-llm-dashboard",
        "source_revision": args.source_revision,
        "activation_dtype": str(dtype).removeprefix("torch."),
        "activation_extractor_version": 4,
        "probe_fit": {"solver": "lbfgs", "C": 0.1, "max_iter": 500, "tol": 0.001},
        "layer_selector": {
            "classifier": "RidgeClassifier",
            "solver": "lsqr",
            "alpha": 10.0,
            "tol": 0.001,
        },
        "text_control_fit": {
            "solver": "lbfgs",
            "C": 1.0,
            "max_iter": 500,
            "tol": 0.001,
        },
        "attributes": {},
        "created_at": datetime.now(UTC).isoformat(),
    }
    if args.extract_only:
        for attr in args.attrs:
            rows = load_rows(args.dataset_root, attr, args.per_class, args.sample_seed)
            key = cache_key(
                args.model,
                attr,
                rows,
                str(dtype).removeprefix("torch."),
            )
            cache_path = args.cache_dir / f"{slug}_{attr}_{key}.npz"
            if not cache_path.exists():
                extract_features(
                    model, tokenizer, device, rows, attr, cache_path, args.batch_size
                )
            print(f"ready {cache_path}")
        return
    for attr in args.attrs:
        classes = ATTRIBUTES[attr]
        rows = load_rows(args.dataset_root, attr, args.per_class, args.sample_seed)
        key = cache_key(
            args.model,
            attr,
            rows,
            str(dtype).removeprefix("torch."),
        )
        cache_path = args.cache_dir / f"{slug}_{attr}_{key}.npz"
        if not cache_path.exists():
            if args.fit_only:
                raise FileNotFoundError(
                    f"missing activation cache for --fit-only: {cache_path}"
                )
            extract_features(
                model, tokenizer, device, rows, attr, cache_path, args.batch_size
            )
        data = np.load(cache_path)
        x, y = data["features"], data["labels"].astype(str)
        aggregate["attributes"][attr] = {
            "activation_cache": cache_path.name,
            "seeds": {},
        }

        for split_seed in args.split_seeds:
            train_idx, val_idx, test_idx = grouped_split(rows, split_seed)
            seed_dir = base_out / f"seed-{split_seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            layer, clf, val_metrics, test_metrics, curve = fit_layers(
                x, y, train_idx, val_idx, test_idx, classes, range(x.shape[1])
            )
            save_probe(
                seed_dir / f"{attr}.npz", layer, clf, classes, val_metrics, test_metrics
            )
            controls = text_controls(rows, y, train_idx, test_idx, classes, split_seed)
            aggregate["attributes"][attr]["seeds"][str(split_seed)] = {
                "reading_layer": layer,
                "selection": val_metrics,
                "test": test_metrics,
                "layer_curve": curve,
                "controls": controls,
                "split": {
                    "unit": "source_index_group",
                    "train_ids": [rows[i]["id"] for i in train_idx],
                    "selection_ids": [rows[i]["id"] for i in val_idx],
                    "test_ids": [rows[i]["id"] for i in test_idx],
                },
            }
            meta_path = seed_dir / "meta.json"
            meta = (
                json.loads(meta_path.read_text())
                if meta_path.exists()
                else {
                    "model": args.model,
                    "model_revision": revision,
                    "attributes": {},
                    "reading": {"suffix": {}, "strip_last_assistant": True},
                    "training": {
                        "recipe": "conversation-probes-v1",
                        "context": "full-conversation",
                        "split_seed": split_seed,
                        "independent_test": True,
                        "activation_dtype": str(dtype).removeprefix("torch."),
                        "activation_extractor_version": 4,
                        "layer_selector": "RidgeClassifier(alpha=10, solver='lsqr', tol=1e-3)",
                    },
                }
            )
            meta["model_revision"] = revision
            meta.setdefault("training", {}).update(
                recipe="conversation-probes-v1",
                context="full-conversation",
                split_seed=split_seed,
                independent_test=True,
                activation_dtype=str(dtype).removeprefix("torch."),
                activation_extractor_version=4,
                layer_selector="RidgeClassifier(alpha=10, solver='lsqr', tol=1e-3)",
            )
            meta["attributes"][attr] = {
                "layer": layer,
                "val_acc": val_metrics["accuracy"],
                "test_acc": test_metrics["accuracy"],
            }
            meta["reading"]["suffix"][attr] = READING_SUFFIX[attr]
            meta_path.write_text(json.dumps(meta, indent=2) + "\n")
            ProbeSet(seed_dir)
            print(
                f"{attr} seed={split_seed} layer={layer} "
                f"test balanced={test_metrics['balanced_accuracy']:.3f} "
                f"macro-F1={test_metrics['macro_f1']:.3f}"
            )

    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "training_report.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )
    source_rows = [
        {
            "id": row["id"],
            "group_id": row["group_id"],
            "attribute": attr,
            "label": row["label"],
            "messages": row["messages"],
        }
        for attr in args.attrs
        for row in load_rows(args.dataset_root, attr, args.per_class, args.sample_seed)
    ]
    # The manifest records exact source IDs and labels without redistributing
    # the source conversation text.
    write_jsonl(
        base_out / "source_manifest.jsonl",
        (
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "attribute": row["attribute"],
                "label": row["label"],
            }
            for row in source_rows
        ),
    )
    print(f"wrote {base_out / 'training_report.json'}")


if __name__ == "__main__":
    main()
