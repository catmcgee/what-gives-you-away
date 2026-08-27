from collections import Counter

from wgya.io_utils import read_jsonl

EXPECTED_CATEGORIES = {
    "affect",
    "control",
    "dialect",
    "disclosure",
    "emoji",
    "formality",
    "grammar",
    "orthography",
    "price",
    "slang",
}


def test_released_full_dataset_is_complete():
    pairs = read_jsonl("data/minimal_pairs.jsonl")
    conversations = read_jsonl("data/conversation_minimal_pairs.jsonl")

    assert len(pairs) == 1025
    assert len({row["base_id"] for row in pairs}) == 100
    assert len({row["pair_id"] for row in pairs}) == 1025
    assert {row["category"] for row in pairs} == EXPECTED_CATEGORIES
    assert Counter(row["pair_id"] for row in conversations) == {
        row["pair_id"]: 4 for row in pairs
    }


def test_every_conversation_pair_changes_one_user_turn():
    paths = (
        "data/conversation_minimal_pairs.jsonl",
        "data/cross_model_conversation_minimal_pairs.jsonl",
    )
    for path in paths:
        for row in read_jsonl(path):
            messages_a = row["messages_a"]
            messages_b = row["messages_b"]
            differences = [
                index
                for index, (message_a, message_b) in enumerate(
                    zip(messages_a, messages_b)
                )
                if message_a != message_b
            ]
            assert differences == [row["target_message_index"]]
            assert messages_a[differences[0]]["role"] == "user"


def test_cross_model_panel_is_a_fixed_subset():
    full_rows = read_jsonl("data/minimal_pairs.jsonl")
    full = {row["pair_id"]: row for row in full_rows}
    panel = read_jsonl("data/cross_model_minimal_pairs.jsonl")
    conversations = read_jsonl("data/cross_model_conversation_minimal_pairs.jsonl")

    assert len(panel) == 505
    assert len({row["base_id"] for row in panel}) == 48
    assert {row["category"] for row in panel} == EXPECTED_CATEGORIES
    assert all(full[row["pair_id"]] == row for row in panel)
    assert Counter(row["pair_id"] for row in conversations) == {
        row["pair_id"]: 4 for row in panel
    }
    base_order = list(dict.fromkeys(row["base_id"] for row in full_rows))
    assert {row["base_id"] for row in panel} == set(base_order[:48])
    assert panel == [row for row in full_rows if row["base_id"] in base_order[:48]]


def test_pair_metadata_and_controls_are_valid():
    pairs = read_jsonl("data/minimal_pairs.jsonl")
    categories_by_base = {}
    for row in pairs:
        categories_by_base.setdefault(row["base_id"], set()).add(row["category"])
        assert row["text_a"] != row["text_b"]
        assert row["n_tok_b"] - row["n_tok_a"] == row["tok_delta"]

    assert all("control" in categories for categories in categories_by_base.values())
