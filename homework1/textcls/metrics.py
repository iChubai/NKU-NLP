"""Classification metrics and per-document predictions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data import write_json


def evaluate(y_true, y_pred, n_classes):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0
        )),
    }


def save_test_outputs(output, method, frame, indices, y_true, y_pred, names):
    directory = Path(output) / method
    directory.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame({
        "row_id": indices,
        "true_label": [names[i] for i in y_true],
        "predicted_label": [names[i] for i in y_pred],
        "correct": np.asarray(y_true) == np.asarray(y_pred),
        "text_preview": [text[:300] for text in frame.iloc[indices]["text"]],
    })
    rows.to_csv(directory / "test_predictions.csv", index=False)
    write_json(directory / "classification_report.json", classification_report(
        y_true, y_pred, labels=np.arange(len(names)), target_names=names,
        output_dict=True, zero_division=0
    ))
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(names)))
    pd.DataFrame(matrix, index=names, columns=names).to_csv(directory / "confusion_matrix.csv")
