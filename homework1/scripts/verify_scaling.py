#!/usr/bin/env python3
"""Recompute scaling-study scores and verify splits, budgets and selections."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_scaling import METRICS, read_json


def verify(directory):
    protocol = read_json(directory / "protocol.json")
    rows = read_json(directory / "results.json")
    planned = {row["run_id"]: row for row in protocol["runs"]}
    assert len(rows) == len(planned) == 204
    assert {row["run_id"] for row in rows} == set(planned)
    partitions = {}
    for seed in protocol["split_seeds"]:
        split_dir = directory / f"split{seed}"
        manifest = read_json(split_dir / "splits.json")
        names = manifest["label_names"]
        joined = np.concatenate([manifest[key] for key in ("train", "validation", "test")])
        np.testing.assert_array_equal(np.sort(joined), np.arange(manifest["n_documents"]))
        reference = pd.read_csv(split_dir / "test_rows.csv", keep_default_na=False)
        info = read_json(split_dir / "data_info.json")
        partitions[seed] = manifest, reference, info
    for row in rows:
        config = planned[row["run_id"]]
        for key in config:
            assert row[key] == config[key], (row["run_id"], key)
        manifest, reference, info = partitions[row["split_seed"]]
        names = manifest["label_names"]
        location = directory / row["run_directory"]
        predictions = pd.read_csv(location / "test_predictions.csv", keep_default_na=False)
        np.testing.assert_array_equal(predictions["row_id"], manifest["test"])
        np.testing.assert_array_equal(predictions["true_label"], reference["label"])
        truth, prediction = predictions["true_label"], predictions["predicted_label"]
        per_class_f1 = f1_score(truth, prediction, labels=names, average=None, zero_division=0)
        scores = accuracy_score(truth, prediction), float(per_class_f1.mean())
        np.testing.assert_allclose(scores, [row["test_accuracy"], row["test_macro_f1"]], atol=1e-12, rtol=0)
        saved = read_json(location / "metrics.json")
        np.testing.assert_allclose(scores, [saved["test_accuracy"], saved["test_macro_f1"]], atol=1e-12, rtol=0)
        report = read_json(location / "classification_report.json")
        np.testing.assert_allclose(per_class_f1, [report[name]["f1-score"] for name in names], atol=1e-12)
        matrix = pd.read_csv(location / "confusion_matrix.csv", index_col=0).to_numpy()
        np.testing.assert_array_equal(matrix, confusion_matrix(truth, prediction, labels=names))
        rare = [name for name in names if info["class_counts"]["train"][name] <= 100]
        assert row["rare_classes"] == rare
        np.testing.assert_allclose(row["rare_macro_f1"],
                                   np.mean([report[name]["f1-score"] for name in rare]), atol=1e-12)
        assert row["zero_f1_classes"] == int(np.sum(per_class_f1 == 0))
        if row["class_weighting"] == "inverse-frequency":
            weights = read_json(location / "class_weights.json")
            expected = [len(manifest["train"]) / (len(names) * info["class_counts"]["train"][name])
                        for name in names]
            np.testing.assert_allclose([weights[name] for name in names], expected, atol=1e-12)
        if row["family"] == "bert":
            history = read_json(location / "training_history.json")
            assert [point["epoch"] for point in history] == list(range(1, row["epochs"] + 1))
            best = max(history, key=lambda point: point["validation_macro_f1"])
            assert row["selected_epoch"] == best["epoch"]
            assert all(point["training_examples"] == len(manifest["train"]) for point in history)
            assert all(point["training_steps"] == int(np.ceil(len(manifest["train"]) / row["batch_size"]))
                       for point in history)
            for key in ("max_length", "epochs", "training_seed", "class_weighting"):
                assert saved[key] == row[key], (row["run_id"], key)
            assert saved.get("input_position", "prefix") == row["input_position"]
            architecture = read_json(location / "best_model/config.json")
            assert architecture["num_hidden_layers"] == 12
            assert architecture["hidden_size"] == 768 and architecture["num_attention_heads"] == 12
            assert row["pretrained_model"] == "google-bert/bert-base-uncased"
        else:
            trials = read_json(location / "validation_trials.json")
            assert [trial["C"] for trial in trials] == [0.1, 1.0, 10.0]
            best = max(trials, key=lambda trial: trial["macro_f1"])
            assert row["selected_C"] == best["C"]
        np.testing.assert_allclose([row["validation_accuracy"], row["validation_macro_f1"]],
                                   [best.get("validation_accuracy", best.get("accuracy")),
                                    best.get("validation_macro_f1", best.get("macro_f1"))], atol=1e-12)
    dimensions = ("family", "representation", "max_length", "epochs", "input_position", "class_weighting")
    summary = read_json(directory / "summary.json")
    assert summary["n_configurations"] == len(rows)
    assert summary["n_new_configurations"] == sum(not row["reused"] for row in rows) == 189
    for group in summary["groups"]:
        selected = [row for row in rows if all(row[key] == group[key] for key in dimensions)]
        assert len(selected) == group["n_runs"]
        for metric in METRICS:
            split_values, training_variation = [], []
            for point in group["splits"]:
                subset = [row[metric] for row in selected if row["split_seed"] == point["split_seed"]]
                mean = np.mean(subset)
                np.testing.assert_allclose(point[metric + "_mean"], mean, atol=1e-12)
                split_values.append(mean)
                if len(subset) > 1:
                    std = np.std(subset, ddof=1)
                    np.testing.assert_allclose(point[metric + "_training_std"], std, atol=1e-12)
                    training_variation.append(std)
                else:
                    assert point[metric + "_training_std"] is None
            np.testing.assert_allclose(group[metric + "_mean"], np.mean(split_values), atol=1e-12)
            if len(split_values) > 1:
                np.testing.assert_allclose(group[metric + "_split_std"],
                                           np.std(split_values, ddof=1), atol=1e-12)
            else:
                assert group[metric + "_split_std"] is None
            if training_variation:
                np.testing.assert_allclose(group[metric + "_training_std_mean"],
                                           np.mean(training_variation), atol=1e-12)
    print(f"Verified {len(rows)} configurations, five common partitions, complete epoch budgets, "
          "validation selections, training-only weights and both variance levels.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/scaling"))
    args = parser.parse_args()
    verify(args.results)


if __name__ == "__main__":
    main()
