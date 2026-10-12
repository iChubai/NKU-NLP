"""Load documents and share one shuffled 80/10/10 partition."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def read_table(path, require_labels=True):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, keep_default_na=False)
    elif path.suffix.lower() == ".jsonl":
        frame = pd.read_json(path, lines=True)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError("Dataset must be CSV, JSONL, or Parquet.")
    required = {"text", "label"} if require_labels else {"text"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if frame["text"].isna().any():
        raise ValueError("The text column contains missing values.")
    frame["text"] = frame["text"].astype(str).str.strip()
    if frame["text"].eq("").any():
        raise ValueError("The text column contains empty documents.")
    if require_labels:
        if frame["label"].isna().any():
            raise ValueError("The label column contains missing values.")
        frame["label"] = frame["label"].astype(str).str.strip()
        if frame["label"].eq("").any():
            raise ValueError("The label column contains empty labels.")
    return frame.reset_index(drop=True)


def shuffled_split(n_rows, seed=42):
    if n_rows < 10:
        raise ValueError("At least 10 documents are needed for an 80/10/10 split.")
    order = np.random.RandomState(seed).permutation(n_rows)
    train_end = int(0.8 * n_rows)
    val_end = train_end + int(0.1 * n_rows)
    return {
        "train": order[:train_end],
        "validation": order[train_end:val_end],
        "test": order[val_end:],
    }


def prepare_data(path, output, seed):
    raw_frame = read_table(path)
    label_counts = raw_frame.groupby("text")["label"].nunique()
    conflicting_texts = label_counts.index[label_counts > 1]
    conflicting_rows = raw_frame["text"].isin(conflicting_texts)
    unambiguous = raw_frame.loc[~conflicting_rows]
    duplicate_rows_removed = int(unambiguous["text"].duplicated().sum())
    frame = unambiguous.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame["label"])
    split = shuffled_split(len(frame), seed)
    names = encoder.classes_.tolist()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "n_documents": len(frame),
        "label_names": names,
        **{key: values.tolist() for key, values in split.items()},
    }
    split_path = output / "splits.json"
    if split_path.exists() and json.loads(split_path.read_text()) != manifest:
        raise ValueError("This output directory has a different split; use a new --output.")
    write_json(split_path, manifest)
    frame_with_ids = frame.assign(row_id=np.arange(len(frame)), label_id=labels)
    for name, indices in split.items():
        frame_with_ids.iloc[indices][["row_id", "label_id", "label"]].to_csv(
            output / f"{name}_rows.csv", index=False
        )
    token_counts = np.array([len(tokenize(text)) for text in frame["text"]])
    info = {
        "input_path": str(Path(path).resolve()),
        "n_raw_documents": len(raw_frame),
        "conflicting_text_groups_removed": len(conflicting_texts),
        "conflicting_rows_removed": int(conflicting_rows.sum()),
        "duplicate_rows_removed": duplicate_rows_removed,
        "n_documents": len(frame),
        "n_classes": len(names),
        "label_names": names,
        "mean_words": float(token_counts.mean()),
        "median_words": float(np.median(token_counts)),
        "duplicate_texts": int(frame["text"].duplicated().sum()),
        "class_counts": {
            name: {
                label: int(np.sum(labels[indices] == i))
                for i, label in enumerate(names)
            }
            for name, indices in split.items()
        },
        "split_sizes": {name: len(indices) for name, indices in split.items()},
    }
    write_json(output / "data_info.json", info)
    return frame, labels, names, split


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
