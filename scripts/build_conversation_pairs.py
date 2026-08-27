"""Embed the released minimal pairs in matched multi-turn conversations.

The demographic probes were trained on full conversations. This builder keeps
every controlled edit intact while varying
where it appears in a dialogue.  Assistant turns are fixed across variants so
the experiment measures context and persistence without adding separately
generated responses as a mediator.

Usage:
    uv run scripts/build_conversation_pairs.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wgya import config
from wgya.io_utils import read_jsonl, write_jsonl

VIEW_ORDER = (
    "current_help",
    "current_advice",
    "one_turn_later",
    "two_turns_later",
)


def conversation_for(text: str, view: str) -> list[dict[str, str]]:
    """Return one deterministic conversation shell containing ``text``."""

    if view == "current_help":
        return [
            {"role": "user", "content": "I could use your help with a question."},
            {
                "role": "assistant",
                "content": "Of course. What would you like help with?",
            },
            {"role": "user", "content": text},
        ]
    if view == "current_advice":
        return [
            {"role": "user", "content": "Can I ask you for some advice?"},
            {
                "role": "assistant",
                "content": "Absolutely. Tell me what you have in mind.",
            },
            {"role": "user", "content": text},
        ]
    if view == "one_turn_later":
        return [
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": "I can help with that. Is there anything else I should keep in mind?",
            },
            {
                "role": "user",
                "content": "No, that covers it. Please keep the answer practical and concise.",
            },
        ]
    if view == "two_turns_later":
        return [
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": "I can help with that. Is there anything else I should keep in mind?",
            },
            {
                "role": "user",
                "content": "No, that covers it. Please keep the answer practical and concise.",
            },
            {
                "role": "assistant",
                "content": "Understood. I will keep the response focused on practical steps.",
            },
            {"role": "user", "content": "Thanks. Please go ahead."},
        ]
    raise ValueError(f"unknown conversation view: {view}")


def build_rows(pairs: list[dict]) -> list[dict]:
    rows = []
    for pair in pairs:
        for view in VIEW_ORDER:
            messages_a = conversation_for(pair["text_a"], view)
            messages_b = conversation_for(pair["text_b"], view)
            differing = [
                index
                for index, (message_a, message_b) in enumerate(
                    zip(messages_a, messages_b)
                )
                if message_a != message_b
            ]
            if len(messages_a) != len(messages_b) or len(differing) != 1:
                raise ValueError(
                    f"conversation shell changed outside target: {pair['pair_id']} {view}"
                )
            target_index = differing[0]
            if messages_a[target_index]["role"] != "user":
                raise ValueError(f"target is not a user turn: {pair['pair_id']} {view}")
            rows.append(
                {
                    "conversation_pair_id": f"{pair['pair_id']}__{view}",
                    "pair_id": pair["pair_id"],
                    "base_id": pair["base_id"],
                    "topic": pair.get("topic"),
                    "category": pair["category"],
                    "subtype": pair.get("subtype"),
                    "tok_delta": pair.get("tok_delta"),
                    "view": view,
                    "target_message_index": target_index,
                    "turns_after_target": sum(
                        message["role"] == "user"
                        for message in messages_a[target_index + 1 :]
                    ),
                    "messages_a": messages_a,
                    "messages_b": messages_b,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=config.DATA_DIR / "minimal_pairs.jsonl"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.DATA_DIR / "conversation_minimal_pairs.jsonl",
    )
    args = parser.parse_args()
    pairs = read_jsonl(args.input)
    rows = build_rows(pairs)
    write_jsonl(args.output, rows)
    print(
        f"Wrote {len(rows)} rows ({len(pairs)} pairs x {len(VIEW_ORDER)} views) "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
