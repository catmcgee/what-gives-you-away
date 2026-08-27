"""Linear probe loading and hidden-state extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

SYSTEM_PROMPT = "You are a helpful assistant."

ATTRIBUTES = {
    "age": ["child", "adolescent", "adult", "older adult"],
    "gender": ["female", "male"],
    "education": ["someschool", "highschool", "collegemore"],
    "socioeco": ["low", "middle", "high"],
}

READING_SUFFIX = {
    "age": " I think the age of this user is",
    "gender": " I think the gender of this user is",
    "education": " I think the education level of this user is",
    "socioeco": " I think the socioeconomic status of this user is",
}

SUFFIX_VARIANTS = {
    "age": [
        READING_SUFFIX["age"],
        " Based on this conversation, this user's age group is",
        " The most likely age category for the user is",
        " My estimate of the user's age is",
    ],
    "gender": [
        READING_SUFFIX["gender"],
        " Based on this conversation, the user's gender is",
        " The most likely gender category for the user is",
        " My estimate of the user's gender is",
    ],
    "education": [
        READING_SUFFIX["education"],
        " Based on this conversation, the user's education category is",
        " The most likely education level for the user is",
        " My estimate of the user's education is",
    ],
    "socioeco": [
        READING_SUFFIX["socioeco"],
        " Based on this conversation, the user's socioeconomic category is",
        " The most likely socioeconomic status for the user is",
        " My estimate of the user's socioeconomic status is",
    ],
}


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def reading_text(tokenizer, messages, suffix: str) -> str:
    """Format a conversation exactly as the released probes expect."""

    messages = list(messages)
    if messages and messages[-1]["role"] == "assistant":
        messages = messages[:-1]
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        tokenize=False,
        add_generation_prompt=True,
    )
    return text + suffix


@torch.no_grad()
def batch_reading_hidden_states(
    model,
    tokenizer,
    message_batches,
    suffix: str,
    device: str,
    max_tokens: int = 2048,
):
    """Return final-token states with shape ``[batch, depth, hidden]``."""

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    previous_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(
            [reading_text(tokenizer, messages, suffix) for messages in message_batches],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
            add_special_tokens=False,
        )
    finally:
        tokenizer.truncation_side = previous_truncation_side

    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    output = model(
        input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    positions = (
        (
            attention_mask
            * torch.arange(attention_mask.shape[1], device=attention_mask.device)
        )
        .max(dim=1)
        .values
    )
    batch_indices = torch.arange(attention_mask.shape[0], device=device)
    return np.stack(
        [
            state[batch_indices, positions].float().cpu().numpy()
            for state in output.hidden_states
        ],
        axis=1,
    )


def probe_logits(probe: dict, hidden_state: np.ndarray) -> np.ndarray:
    logits = probe["coef"] @ hidden_state + probe["intercept"]
    if len(logits) == 1:
        logits = np.asarray([-logits[0] / 2, logits[0] / 2])
    return np.asarray(logits, dtype=float)


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    exponentiated = np.exp(logits - logits.max())
    return exponentiated / exponentiated.sum()


class ProbeSet:
    """Load one seed of the released demographic probe ensemble."""

    def __init__(self, probe_dir):
        self.probe_dir = Path(probe_dir)
        self.meta = json.loads((self.probe_dir / "meta.json").read_text())
        self.probes = {}
        for attribute in self.meta["attributes"]:
            data = np.load(self.probe_dir / f"{attribute}.npz")
            self.probes[attribute] = {
                "layer": int(data["layer"]),
                "classes": [str(value) for value in data["classes"]],
                "coef": data["coef"],
                "intercept": data["intercept"],
                "val_acc": float(data["val_acc"]),
                "test_acc": float(data["test_acc"]),
            }
