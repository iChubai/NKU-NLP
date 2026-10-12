#!/usr/bin/env python3
"""Verify reported scores against predictions and the shared test partition."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

METHODS = ["bow_binary", "bow_count", "bow_tfidf", "glove",
           "word2vec_ag_news", "word2vec_nyt", "bert"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(directory):
    manifest = read_json(directory / "splits.json")
    labels = manifest["label_names"]
    results = {row["method"]: row for row in read_json(directory / "results.json")}
    reference = pd.read_csv(directory / "test_rows.csv", keep_default_na=False)
    partition = np.concatenate([manifest[key] for key in ("train", "validation", "test")])
    np.testing.assert_array_equal(np.sort(partition), np.arange(manifest["n_documents"]))
    assert set(results) == set(METHODS), "The seven required methods are not all present."
    for method in METHODS:
        predictions = pd.read_csv(directory / method / "test_predictions.csv", keep_default_na=False)
        np.testing.assert_array_equal(predictions["row_id"], manifest["test"])
        np.testing.assert_array_equal(predictions["true_label"].astype(str), reference["label"].astype(str))
        truth, prediction = predictions["true_label"].astype(str), predictions["predicted_label"].astype(str)
        accuracy = accuracy_score(truth, prediction)
        macro_f1 = f1_score(truth, prediction, labels=labels, average="macro", zero_division=0)
        np.testing.assert_allclose([accuracy, macro_f1],
                                   [results[method]["test_accuracy"], results[method]["test_macro_f1"]],
                                   rtol=0, atol=1e-12)
        saved_matrix = pd.read_csv(directory / method / "confusion_matrix.csv", index_col=0)
        np.testing.assert_array_equal(saved_matrix.to_numpy(), confusion_matrix(truth, prediction, labels=labels))
        per_class = read_json(directory / method / "classification_report.json")
        np.testing.assert_allclose(np.mean([per_class[label]["f1-score"] for label in labels]), macro_f1)
        print(f"{method}: Accuracy={accuracy:.6f}, Macro-F1={macro_f1:.6f}; {len(predictions)} test rows")
    bert = results["bert"]
    assert bert["epochs"] == 3 and bert["max_length"] == 64
    assert bert["pretrained_model"] == "google-bert/bert-base-uncased"
    history = read_json(directory / "bert/training_history.json")
    assert [entry["epoch"] for entry in history] == [1, 2, 3]
    best = max(history, key=lambda entry: entry["validation_macro_f1"])
    assert bert["selected_epoch"] == best["epoch"]
    for method in METHODS[:6]:
        trials = read_json(directory / method / "validation_trials.json")
        best = max(trials, key=lambda entry: entry["macro_f1"])
        assert results[method]["selected_C"] == best["C"]
    print("All seven methods use the same test partition; metrics and model selection agree with saved outputs.")


def verify_explorations(directory, base):
    manifest = read_json(directory / "splits.json")
    assert manifest == read_json(base / "splits.json")
    names = manifest["label_names"]
    reference = pd.read_csv(base / "test_rows.csv", keep_default_na=False)
    train = pd.read_csv(base / "train_rows.csv", keep_default_na=False)
    reference["label"] = reference["label"].astype(str)
    train["label"] = train["label"].astype(str)
    rows = read_json(directory / "results.json")
    expected = {
        *(f"{method}_prefix{length}" for method in METHODS[:3] for length in (64, 128, 256)),
        *(f"bert_len{length}_{weighting}_seed{seed}" for length in (64, 128, 256)
          for weighting in ("uniform", "balanced") for seed in (42, 43, 44)),
        *(f"{source}_{pooling}_seed{seed}" for source in METHODS[3:6]
          for pooling in ("mean", "tfidf") for seed in ((42,) if source == "glove" else (42, 43, 44))),
    }
    assert {row["method"] for row in rows} == expected
    assert len(rows) == len(expected) == 41
    summary = read_json(directory / "summary.json")
    rare_names = [name for name in names if train["label"].eq(name).sum() <= 100]
    assert summary["rare_classes"] == rare_names
    for row in rows:
        location = directory / row["run_directory"]
        predictions = pd.read_csv(location / "test_predictions.csv", keep_default_na=False)
        np.testing.assert_array_equal(predictions["row_id"], manifest["test"])
        np.testing.assert_array_equal(predictions["true_label"].astype(str), reference["label"])
        truth = predictions["true_label"].astype(str)
        prediction = predictions["predicted_label"].astype(str)
        scores = [accuracy_score(truth, prediction),
                  f1_score(truth, prediction, labels=names, average="macro", zero_division=0)]
        np.testing.assert_allclose(scores, [row["test_accuracy"], row["test_macro_f1"]],
                                   rtol=0, atol=1e-12, err_msg=row["method"])
        saved = read_json(location / "metrics.json")
        np.testing.assert_allclose(scores, [saved["test_accuracy"], saved["test_macro_f1"]],
                                   rtol=0, atol=1e-12)
        matrix = pd.read_csv(location / "confusion_matrix.csv", index_col=0)
        np.testing.assert_array_equal(matrix.to_numpy(), confusion_matrix(truth, prediction, labels=names))
        per_class = read_json(location / "classification_report.json")
        np.testing.assert_allclose(row["rare_macro_f1"],
                                   np.mean([per_class[name]["f1-score"] for name in rare_names]))
        assert row["zero_f1_classes"] == sum(per_class[name]["f1-score"] == 0 for name in names)
        if row["family"] == "bert":
            assert saved["max_length"] == row["max_length"] and saved["epochs"] == 3
            assert saved["pretrained_model"] == "google-bert/bert-base-uncased"
            history = read_json(location / "training_history.json")
            assert [entry["epoch"] for entry in history] == [1, 2, 3]
            assert row["selected_epoch"] == max(history, key=lambda entry: entry["validation_macro_f1"])["epoch"]
            if location.resolve() != (base / "bert").resolve():
                assert saved["training_seed"] == row["training_seed"]
                assert saved["class_weighting"] == row["class_weighting"]
                assert all(entry["training_examples"] == len(manifest["train"]) for entry in history)
            if row["class_weighting"] == "inverse-frequency":
                weights = read_json(location / "class_weights.json")
                expected_weights = [len(train) / (len(names) * train["label"].eq(name).sum()) for name in names]
                np.testing.assert_allclose([weights[name] for name in names], expected_weights)
        else:
            trials = read_json(location / "validation_trials.json")
            assert row["selected_C"] == max(trials, key=lambda trial: trial["macro_f1"])["C"]
            if row["family"] == "embedding":
                assert row["feature_dim"] == 100
    for group in summary["bert"]:
        selected = [row for row in rows if row["family"] == "bert"
                    and row["max_length"] == group["max_length"]
                    and row["class_weighting"] == group["class_weighting"]]
        assert sorted(row["training_seed"] for row in selected) == [42, 43, 44]
        assert group["n_seeds"] == 3
        for metric in ("test_accuracy", "test_macro_f1", "validation_macro_f1", "rare_macro_f1", "zero_f1_classes"):
            values = [row[metric] for row in selected]
            np.testing.assert_allclose(group[metric + "_mean"], np.mean(values))
            np.testing.assert_allclose(group[metric + "_std"], np.std(values, ddof=1))
    for group in summary["embedding_pooling"]:
        selected = [row for row in rows if row["family"] == "embedding"
                    and row["method"].startswith(group["source"] + "_") and row["pooling"] == group["pooling"]]
        assert group["n_seeds"] == len(selected)
        for metric in ("test_accuracy", "test_macro_f1"):
            np.testing.assert_allclose(group[metric + "_mean"], np.mean([row[metric] for row in selected]))
            if len(selected) > 1:
                np.testing.assert_allclose(group[metric + "_std"], np.std([row[metric] for row in selected], ddof=1))
    print("All 41 supplementary configurations agree with their predictions, shared split, training records and seed summaries.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/main"))
    parser.add_argument("--explorations", type=Path)
    args = parser.parse_args()
    verify(args.results)
    if args.explorations is not None:
        verify_explorations(args.explorations, args.results)


if __name__ == "__main__":
    main()
