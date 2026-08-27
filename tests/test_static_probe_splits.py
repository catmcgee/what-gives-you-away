from scripts.train_probes import grouped_split, row_label


def test_grouped_split_keeps_source_index_sets_together():
    rows = [
        {"id": f"{group}/{label}", "group_id": group, "label": label}
        for group_index in range(40)
        for group in [f"group-{group_index}"]
        for label in ("female", "male")
    ]
    train, selection, test = grouped_split(rows, seed=0)
    partitions = [
        {rows[i]["group_id"] for i in indices} for indices in (train, selection, test)
    ]
    assert not (partitions[0] & partitions[1])
    assert not (partitions[0] & partitions[2])
    assert not (partitions[1] & partitions[2])
    assert sum(map(len, partitions)) == 40
    for indices in (train, selection, test):
        labels_by_group = {}
        for i in indices:
            labels_by_group.setdefault(rows[i]["group_id"], set()).add(rows[i]["label"])
        assert all(labels == {"female", "male"} for labels in labels_by_group.values())


def test_probe_trainer_accepts_normalized_labels():
    assert row_label({"label": "female"}, "gender") == "female"
    assert row_label({"labels": {"age": "adult"}}, "age") == "adult"
