#!/usr/bin/env python3
"""Run context, class-weighting, pooling and training-seed experiments."""

import argparse
import importlib.metadata
import json
import os
import platform
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from transformers import AutoTokenizer

from textcls.classical import finish_classifier, mean_embeddings, run_bow
from textcls.data import prepare_data, tokenize, write_json
from textcls.neural import run_bert

ROOT = Path(__file__).resolve().parents[1]
LENGTHS = (64, 128, 256)
SEEDS = (42, 43, 44)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_rows(output, task, rows):
    write_json(output / f"results_{task}.json", rows)


def reference_row(base, output, source, method, **configuration):
    row = read_json(base / source / "metrics.json")
    return {**row, "method": method, "run_directory": os.path.relpath(base / source, output),
            "split_seed": 42, **configuration}


def run_prefix_experiments(frame, labels, names, split, args):
    output = Path(args.output)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    lengths = []
    prefix_texts = {length: [] for length in LENGTHS}
    for start in range(0, len(frame), 128):
        texts = frame["text"].iloc[start:start + 128].tolist()
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True, verbose=False)
        for text, offsets in zip(texts, encoded["offset_mapping"]):
            lengths.append(len(offsets))
            for length in LENGTHS:
                end = offsets[min(len(offsets), length - 2) - 1][1]
                prefix_texts[length].append(text[:end])
    lengths = np.asarray(lengths)
    statistics = {
        "median_wordpiece_tokens": float(np.median(lengths)),
        "mean_wordpiece_tokens": float(lengths.mean()),
        "lengths": {
            str(length): {
                "text_token_budget": length - 2,
                "truncated_document_fraction": float(np.mean(lengths > length - 2)),
                "mean_retained_token_fraction": float(np.mean(np.minimum(lengths, length - 2) / lengths)),
                "median_visible_regex_words": float(np.median([
                    len(tokenize(text)) for text in prefix_texts[length]
                ])),
            } for length in LENGTHS
        },
    }
    write_json(output / "input_statistics.json", statistics)
    rows = []
    for length in LENGTHS:
        prefix_frame = frame.copy()
        prefix_frame["text"] = prefix_texts[length]
        for result in run_bow(prefix_frame, labels, names, split, args,
                              method_suffix=f"_prefix{length}"):
            result.update(family="sparse-prefix", max_length=length,
                          training_seed=42, split_seed=42,
                          run_directory=result["method"])
            rows.append(result)
            save_rows(output, "sparse", rows)
    return rows


def run_pooling_experiments(frame, labels, names, split, args):
    base, output = Path(args.base_results), Path(args.output)
    documents = [tokenize(text) for text in frame["text"]]
    document_frequencies = Counter(word for i in split["train"] for word in set(documents[i]))
    n_train = len(split["train"])
    idf = {word: float(np.log((1 + n_train) / (1 + count)) + 1)
           for word, count in document_frequencies.items()}
    unseen_idf = float(np.log(1 + n_train) + 1)
    write_json(output / "pooling_config.json", {
        "idf_fit_on": "NYT train", "training_documents": n_train,
        "formula": "log((1 + N) / (1 + document_frequency)) + 1",
        "unseen_word_idf": unseen_idf, "embedding_dimension": 100,
        "word2vec": {"sg": 1, "window": 5, "min_count": 2, "negative": 5,
                     "sample": 0.001, "epochs": 10, "workers": 1},
    })
    rows = []
    ag_corpus = None
    for source in ("glove", "word2vec_ag_news", "word2vec_nyt"):
        source_seeds = (42,) if source == "glove" else SEEDS
        for seed in source_seeds:
            mean_method = f"{source}_mean_seed{seed}"
            if seed == 42:
                rows.append(reference_row(base, output, source, mean_method,
                                          family="embedding", pooling="mean", training_seed=seed))
                if source == "glove":
                    vectors = joblib.load(base / source / "glove_vectors.joblib")
                else:
                    vectors = Word2Vec.load(str(base / source / "word2vec.model")).wv
            else:
                start = time.perf_counter()
                if source == "word2vec_ag_news":
                    if ag_corpus is None:
                        ag_texts = pd.read_csv(args.ag_news, keep_default_na=False)["text"]
                        ag_corpus = [tokenize(text) for text in ag_texts]
                    corpus = ag_corpus
                else:
                    corpus = [documents[i] for i in split["train"]]
                print(f"Training {source}, seed {seed}: {len(corpus)} documents", flush=True)
                model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=2,
                                 sg=1, negative=5, sample=1e-3, epochs=10, workers=1, seed=seed)
                directory = output / mean_method
                directory.mkdir(parents=True, exist_ok=True)
                model.save(str(directory / "word2vec.model"))
                vectors = model.wv
                features, coverage = mean_embeddings(documents, vectors)
                current_args = SimpleNamespace(**{**vars(args), "seed": seed})
                result = finish_classifier(
                    mean_method, features, frame, labels, names, split, current_args, start,
                    {"feature_dim": 100, "embedding_source": source,
                     "vocabulary_size": len(vectors), "pooling": "mean", **coverage}, scale=True,
                )
                result.update(family="embedding", training_seed=seed, split_seed=42,
                              run_directory=mean_method)
                rows.append(result)
                print(f"{mean_method}: {result['test_accuracy']:.4f}, {result['test_macro_f1']:.4f}", flush=True)
            save_rows(output, "embeddings", rows)
            start = time.perf_counter()
            method = f"{source}_tfidf_seed{seed}"
            features, coverage = mean_embeddings(documents, vectors, idf=idf, unseen_idf=unseen_idf)
            current_args = SimpleNamespace(**{**vars(args), "seed": seed})
            result = finish_classifier(
                method, features, frame, labels, names, split, current_args, start,
                {"feature_dim": 100, "embedding_source": source, "pooling": "tfidf",
                 "vocabulary_size": len(vectors), **coverage}, scale=True,
            )
            result.update(family="embedding", training_seed=seed, split_seed=42,
                          run_directory=method)
            rows.append(result)
            save_rows(output, "embeddings", rows)
            print(f"{method}: {result['test_accuracy']:.4f}, {result['test_macro_f1']:.4f}", flush=True)
    return rows


def run_bert_experiments(frame, labels, names, split, args):
    base, output = Path(args.base_results), Path(args.output)
    rows = [reference_row(base, output, "bert", "bert_len64_uniform_seed42",
                          family="bert", max_length=64, class_weighting="uniform", training_seed=42)]
    save_rows(output, "bert", rows)
    for length in LENGTHS:
        for balanced in (False, True):
            weighting = "balanced" if balanced else "uniform"
            for seed in SEEDS:
                if (length, balanced, seed) == (64, False, 42):
                    continue
                method = f"bert_len{length}_{weighting}_seed{seed}"
                current_args = SimpleNamespace(**{**vars(args), "seed": seed})
                print(f"Starting {method}", flush=True)
                result = next(run_bert(frame, labels, names, split, current_args,
                                       method=method, max_length=length, balanced=balanced))
                result.update(family="bert", split_seed=42, run_directory=method)
                rows.append(result)
                save_rows(output, "bert", rows)
                print(f"{method}: {result['test_accuracy']:.4f}, {result['test_macro_f1']:.4f}", flush=True)
    return rows


def summarize(output, rows, names, split, labels):
    rare_ids = [i for i in range(len(names)) if np.sum(labels[split["train"]] == i) <= 100]
    rare_names = [names[i] for i in rare_ids]
    for row in rows:
        report = read_json(output / row["run_directory"] / "classification_report.json")
        row["rare_macro_f1"] = float(np.mean([report[name]["f1-score"] for name in rare_names]))
        row["zero_f1_classes"] = sum(report[name]["f1-score"] == 0 for name in names)
    write_json(output / "results.json", rows)
    pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
    groups = []
    configurations = sorted({(row["max_length"], row["class_weighting"])
                             for row in rows if row["family"] == "bert"})
    for length, weighting in configurations:
        selected = [row for row in rows if row["family"] == "bert" and row["max_length"] == length
                    and row["class_weighting"] == weighting]
        group = {"max_length": length, "class_weighting": weighting, "n_seeds": len(selected)}
        for metric in ("test_accuracy", "test_macro_f1", "validation_macro_f1", "rare_macro_f1", "zero_f1_classes"):
            values = [row[metric] for row in selected]
            group[metric + "_mean"] = float(np.mean(values))
            group[metric + "_std"] = float(np.std(values, ddof=1))
        groups.append(group)
    pooling = []
    for source in ("glove", "word2vec_ag_news", "word2vec_nyt"):
        for strategy in ("mean", "tfidf"):
            selected = [row for row in rows if row["family"] == "embedding"
                        and row["method"].startswith(source + "_") and row["pooling"] == strategy]
            group = {"source": source, "pooling": strategy, "n_seeds": len(selected)}
            for metric in ("test_accuracy", "test_macro_f1"):
                group[metric + "_mean"] = float(np.mean([row[metric] for row in selected]))
                group[metric + "_std"] = (float(np.std([row[metric] for row in selected], ddof=1))
                                          if len(selected) > 1 else None)
            pooling.append(group)
    paired_gains = []
    for seed in SEEDS:
        pair = {row["max_length"]: row for row in rows if row["family"] == "bert"
                and row["class_weighting"] == "uniform" and row["training_seed"] == seed}
        paired_gains.append({"training_seed": seed,
                             "accuracy_gain_256_vs_64": pair[256]["test_accuracy"] - pair[64]["test_accuracy"],
                             "macro_f1_gain_256_vs_64": pair[256]["test_macro_f1"] - pair[64]["test_macro_f1"]})
    write_json(output / "summary.json", {
        "training_seeds": list(SEEDS), "split_seed": 42,
        "rare_class_training_threshold": 100, "rare_classes": rare_names,
        "bert": groups, "embedding_pooling": pooling, "paired_context_gains": paired_gains,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", default="data/nyt.csv")
    parser.add_argument("--ag-news", default="data/ag_news.csv")
    parser.add_argument("--bert-model", default="data/bert-base-uncased")
    parser.add_argument("--base-results", default="outputs/main")
    parser.add_argument("--output", default="outputs/explorations")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    args = parser.parse_args()
    args.seed = 42
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    output = Path(args.output)
    frame, labels, names, split = prepare_data(args.nyt, output, 42)
    assert read_json(output / "splits.json") == read_json(Path(args.base_results) / "splits.json")
    write_json(output / "run_config.json", {**vars(args), "training_seeds": list(SEEDS),
                                           "input_lengths": list(LENGTHS), "epochs": 3,
                                           "class_weighting": ["uniform", "inverse-frequency"]})
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "gensim", "torch", "transformers", "huggingface-hub"]
    write_json(output / "environment.json", {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    })
    print(f"Supplementary experiments: {len(frame)} documents; shared split seed 42", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        sparse = pool.submit(run_prefix_experiments, frame, labels, names, split, args)
        embeddings = pool.submit(run_pooling_experiments, frame, labels, names, split, args)
        bert_rows = run_bert_experiments(frame, labels, names, split, args)
        rows = [*sparse.result(), *embeddings.result(), *bert_rows]
    summarize(output, rows, names, split, labels)
    print(f"Completed {len(rows)} supplementary configurations: {output / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
