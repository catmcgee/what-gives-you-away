import gzip
import hashlib
import json
from pathlib import Path

from scipy.stats import rankdata, spearmanr

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
    assert result["rank_agreement"]["pooled_spearman_rho"] == 1.0
    assert {
        record["spearman_rho"]
        for record in result["rank_agreement"]["by_readout"].values()
    } == {1.0}
    assert result["direction_agreement"]["count"] == 30
    assert result["direction_agreement"]["fraction"] == 1.0
    assert result["direction_agreement"]["disagreements"] == []
    assert result["corrected_significance_overlap"]["count"] == 30
    assert result["corrected_significance_overlap"]["fraction"] == 1.0
    assert result["corrected_significance_overlap"]["not_significant"] == []
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
    source_summaries = [reference] + [
        json.loads(path.read_text()) for path in candidate_paths
    ]
    assert {
        summary["config"]["plan_sha"] for summary in source_summaries
    } == {result["plan_sha256"][:12]}


def test_manuscript_effect_table_matches_released_summary():
    summary = json.loads(
        Path("results/llama3b/summary_d61d45c5c48f.json").read_text()
    )
    matrix = summary["results"]["summaries"]["primary_current"]["matrix"]
    manuscript = Path("paper/main.tex").read_text()
    rows = (
        ("disclosure", "age", "Personal context", "age"),
        ("emoji", "age", "Emoji", "age"),
        ("disclosure", "gender", "Personal context", "gender"),
        ("emoji", "gender", "Emoji", "gender"),
        ("grammar", "education", "Grammatical complexity", "education"),
        ("disclosure", "education", "Personal context", "education"),
        ("price", "socioeco", "Price language", "socioeconomic status"),
        ("disclosure", "socioeco", "Personal context", "socioeconomic status"),
    )
    for category, attribute, cue_label, readout_label in rows:
        estimate = matrix[category][attribute]["paired_vs_control"]
        low, high = estimate["ci95"]
        expected = (
            f"{cue_label} & {readout_label} & {estimate['mean']:.3f} & "
            f"[{low:.3f}, {high:.3f}] \\\\"
        )
        assert expected in manuscript


def test_manuscript_cross_model_table_matches_released_summary():
    result = json.loads(Path("results/cross_model/summary.json").read_text())
    manuscript = Path("paper/main.tex").read_text()
    labels = ("Llama 3.1 8B", "OLMo 2 32B")
    for label, comparison in zip(labels, result["comparisons"], strict=True):
        rank = comparison["rank_agreement"]
        readout_rhos = [
            record["spearman_rho"] for record in rank["by_readout"].values()
        ]
        pooled = f"{rank['pooled_spearman_rho']:.3f}".removeprefix("0")
        low = f"{min(readout_rhos):.3f}".removeprefix("0")
        high = f"{max(readout_rhos):.3f}".removeprefix("0")
        expected = (
            f"{label} & {pooled} & {low}--{high} & "
            f"{comparison['direction_agreement']['count']}/"
            f"{comparison['reference_significant_cells']} & "
            f"{comparison['corrected_significance_overlap']['count']}/"
            f"{comparison['reference_significant_cells']} \\\\"
        )
        assert expected in manuscript


def test_manuscript_discloses_cross_model_analysis_change():
    manuscript = Path("paper/main.tex").read_text()
    assert "Departure from the planned cross-model endpoints" in manuscript
    assert "post-plan change" in manuscript
    assert "arbitrary scales" in manuscript


def test_manuscript_panel_coverage_matches_released_3b_summaries():
    paths = (
        "results/llama3b/summary_d61d45c5c48f.json",
        "results/cross_model/llama3b/summary_302951caa648.json",
    )
    matrices = [
        json.loads(Path(path).read_text())["results"]["summaries"]
        ["primary_current"]["matrix"]
        for path in paths
    ]
    attributes = ("age", "gender", "education", "socioeco")
    categories = sorted(category for category in matrices[0] if category != "control")

    rank_maps = []
    for matrix in matrices:
        ranks = {}
        for attribute in attributes:
            values = [
                matrix[category][attribute]["paired_vs_control"]["mean"]
                for category in categories
            ]
            ranks.update(
                {
                    (category, attribute): rank
                    for category, rank in zip(
                        categories, rankdata(values), strict=True
                    )
                }
            )
        rank_maps.append(ranks)

    keys = sorted(rank_maps[0])
    pooled = spearmanr(
        [rank_maps[0][key] for key in keys],
        [rank_maps[1][key] for key in keys],
    ).statistic
    by_readout = []
    for attribute in attributes:
        selected = [key for key in keys if key[1] == attribute]
        by_readout.append(
            spearmanr(
                [rank_maps[0][key] for key in selected],
                [rank_maps[1][key] for key in selected],
            ).statistic
        )

    assert round(pooled, 3) == 0.917
    assert round(min(by_readout), 3) == 0.833
    assert round(max(by_readout), 3) == 0.983
    manuscript = Path("paper/main.tex").read_text()
    assert "correlate $.917$ with the full 3B ranks" in manuscript
    assert "range from $.833$ to $.983$" in manuscript


def test_arxiv_source_references_only_flat_local_assets():
    manuscript = Path("paper/main.tex").read_text()
    expected_files = (
        "paper/main.tex",
        "paper/refs.bib",
        "paper/fig_conversation_sensitivity.pdf",
        "paper/fig_cross_model.pdf",
    )
    assert all(Path(path).is_file() for path in expected_files)
    assert "../results/" not in manuscript
    assert "{fig_conversation_sensitivity.pdf}" in manuscript
    assert "{fig_cross_model.pdf}" in manuscript
    assert "\\bibliography{refs}" in manuscript
    assert Path("paper/fig_conversation_sensitivity.pdf").read_bytes() == Path(
        "results/fig_conversation_sensitivity.pdf"
    ).read_bytes()
    assert Path("paper/fig_cross_model.pdf").read_bytes() == Path(
        "results/fig_cross_model.pdf"
    ).read_bytes()


def test_manuscript_has_stable_release_and_llama32_source():
    manuscript = Path("paper/main.tex").read_text()
    references = Path("paper/refs.bib").read_text()
    assert "releases/tag/v1.5.0" in manuscript
    assert "\\citep{meta2024llama32}" in manuscript
    assert "ai.meta.com/blog/llama-3-2" in references
