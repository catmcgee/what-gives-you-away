import gzip
import hashlib
import json
from pathlib import Path

from scripts.analyze_conversation_sensitivity import (
    base_differences,
    measurements_for,
    read_payload,
)
from scripts.analyze_cross_model import assess_probe_gate, compare
from scripts.build_conversation_pairs import VIEW_ORDER, build_rows
from scripts.run_conversation_sensitivity import suffix_conditions


def source_pair(pair_id="base__emoji", category="emoji", text_b="Please help 🙂"):
    return {
        "pair_id": pair_id,
        "base_id": "base",
        "topic": "test",
        "category": category,
        "subtype": None,
        "attrs_hint": None,
        "confound": None,
        "tok_delta": 1,
        "text_a": "Please help",
        "text_b": text_b,
    }


def test_conversation_builder_changes_only_one_user_turn():
    rows = build_rows([source_pair()])
    assert len(rows) == len(VIEW_ORDER)
    assert {row["view"] for row in rows} == set(VIEW_ORDER)
    assert len(suffix_conditions(rows)) == len(VIEW_ORDER) + 3
    for row in rows:
        messages_a, messages_b = row["messages_a"], row["messages_b"]
        differences = [
            index
            for index, pair in enumerate(zip(messages_a, messages_b))
            if pair[0] != pair[1]
        ]
        assert differences == [row["target_message_index"]]
        assert messages_a[differences[0]]["role"] == "user"
        assert [message["role"] for message in messages_a] == [
            "user" if index % 2 == 0 else "assistant"
            for index in range(len(messages_a))
        ]


def result_row(pair_id, base_id, category, view, seed_movements):
    seeds = {
        str(seed): {"dlogits": [movement, -movement]}
        for seed, movement in enumerate(seed_movements)
    }
    return {
        "pair_id": pair_id,
        "base_id": base_id,
        "category": category,
        "view": view,
        "suffix_index": 0,
        "attrs": {
            "gender": {
                "seeds": seeds,
                "ensemble": {"flipped": False},
            }
        },
    }


def test_analysis_averages_probe_seeds_and_shells_before_pairing():
    rows = [
        result_row("emoji", "base", "emoji", "current_help", [3.0, 5.0]),
        result_row("emoji", "base", "emoji", "current_advice", [5.0, 7.0]),
        result_row("control", "base", "control", "current_help", [1.0, 1.0]),
        result_row("control", "base", "control", "current_advice", [1.0, 1.0]),
    ]
    measurements = measurements_for(
        rows,
        [("current_help", 0), ("current_advice", 0)],
        "gender",
    )
    emoji = next(row for row in measurements if row["category"] == "emoji")
    assert emoji["seed_movements"] == {"0": 4.0, "1": 6.0}
    assert emoji["ensemble_movement"] == 5.0
    assert base_differences(measurements, "emoji") == {"base": 4.0}


def test_analysis_reads_compressed_raw_results(tmp_path):
    path = tmp_path / "deltas_test.json.gz"
    expected = {"config": {"phase": "test"}, "results": []}
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(expected, stream)
    assert read_payload(path) == expected


def test_cross_model_comparison_is_identity_on_same_summary():
    summary = json.loads(
        Path("results/cross_model/llama3b/summary_302951caa648.json").read_text()
    )
    result = compare(summary, summary)
    assert result["spearman_rho"] == 1.0
    assert result["direction_agreement"]["count"] == 30
    assert result["direction_agreement"]["fraction"] == 1.0
    assert result["direction_agreement"]["disagreements"] == []
    assert result["corrected_significance_overlap"]["count"] == 30
    assert result["corrected_significance_overlap"]["fraction"] == 1.0
    assert result["corrected_significance_overlap"]["not_significant"] == []
    assert result["dominant_readout_agreement"]["count"] == 9
    for view in ("one_turn_later", "two_turns_later"):
        assert result["persistence"][view]["spearman_rho"] == 1.0
        assert result["persistence"][view]["mean_absolute_difference"] == 0.0


def test_released_probe_ensembles_pass_quality_gate():
    reports = (
        Path("data/probes/training_report.json"),
        Path("data/probes/llama8b/training_report.json"),
        Path("data/probes/olmo32b/training_report.json"),
    )
    for path in reports:
        gate = assess_probe_gate(json.loads(path.read_text()))
        assert gate["all_attributes_pass"]
        assert set(gate["attributes"]) == {
            "age",
            "gender",
            "education",
            "socioeco",
        }


def test_released_cross_model_summary_matches_source_summaries():
    result = json.loads(Path("results/cross_model/summary.json").read_text())
    plan = Path(result["plan"])
    assert result["plan_sha256"] == hashlib.sha256(plan.read_bytes()).hexdigest()

    reference = json.loads(
        Path("results/cross_model/llama3b/summary_302951caa648.json").read_text()
    )
    candidate_paths = (
        Path("results/cross_model/llama8b/summary_e7901d719f28.json"),
        Path("results/cross_model/olmo32b/summary_90c18609ce4f.json"),
    )
    expected = [
        compare(reference, json.loads(path.read_text())) for path in candidate_paths
    ]
    assert result["comparisons"] == expected
