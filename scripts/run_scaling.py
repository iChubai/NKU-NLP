#!/usr/bin/env python3
"""Run repeated-split, loss, context, training-budget and text-position studies."""

import argparse
import importlib.metadata
import json
import os
import platform
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from transformers import AutoTokenizer

from textcls.classical import finish_classifier, mean_embeddings, run_bow
from textcls.data import prepare_data, tokenize, write_json
from textcls.neural import encode_documents, run_bert

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEEDS = (42, 43, 44, 45, 46)
TRAIN_SEEDS = (42, 43, 44)
WEIGHTINGS = ("uniform", "inverse-frequency")
SPARSE = ("bow_binary", "bow_count", "bow_tfidf")
METRICS = ("test_accuracy", "test_macro_f1", "rare_macro_f1")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def weight_name(weighting):
    return "balanced" if weighting == "inverse-frequency" else "uniform"


def entry(family, split_seed, training_seed, weighting, length, epochs, position, representation):
    loss = weight_name(weighting)
    if family == "bert":
        method = f"bert_len{length}_{loss}_seed{training_seed}_e{epochs}_{position}"
    elif family == "sparse":
        view = f"prefix{length}" if length else "full"
        method = f"{representation}_{view}_{loss}"
    else:
        method = f"word2vec_nyt_tfidf_{loss}_seed{training_seed}"
    return {
        "method": method, "run_id": f"split{split_seed}/{method}", "family": family,
        "representation": representation, "split_seed": split_seed,
        "training_seed": training_seed, "class_weighting": weighting,
        "max_length": length, "epochs": epochs, "input_position": position,
    }


def planned_runs():
    runs = []
    for split_seed in SPLIT_SEEDS:
        for length in (256, 512):
            for weighting in WEIGHTINGS:
                for seed in TRAIN_SEEDS:
                    runs.append(entry("bert", split_seed, seed, weighting, length, 3, "prefix", "bert"))
        for length in (0, 256, 512):
            for weighting in WEIGHTINGS:
                for representation in SPARSE:
                    runs.append(entry("sparse", split_seed, 42, weighting, length, 0,
                                      "prefix" if length else "full", representation))
        for weighting in WEIGHTINGS:
            for seed in TRAIN_SEEDS:
                runs.append(entry("embedding", split_seed, seed, weighting, 0, 10, "full", "word2vec_nyt"))
    for length in (64, 256, 512):
        for weighting in WEIGHTINGS:
            for seed in TRAIN_SEEDS:
                runs.append(entry("bert", 42, seed, weighting, length, 10, "prefix", "bert"))
    for position in ("tail", "head_tail"):
        for seed in TRAIN_SEEDS:
            runs.append(entry("bert", 42, seed, "inverse-frequency", 256, 3, position, "bert"))
    return runs


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for seed in SPLIT_SEEDS:
        frame, labels, names, _ = prepare_data(args.nyt, output / f"split{seed}", seed)
    assert read_json(output / "split42/splits.json") == read_json(Path(args.base_results) / "splits.json")
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    joblib.dump((frame, labels, names), cache / "documents.joblib")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    views = {256: [], 512: []}
    lengths = []
    for start in range(0, len(frame), 128):
        texts = frame["text"].iloc[start:start + 128].tolist()
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False,
                            return_offsets_mapping=True, verbose=False)
        for text, offsets in zip(texts, encoded["offset_mapping"]):
            lengths.append(len(offsets))
            for length in views:
                end = offsets[min(len(offsets), length - 2) - 1][1]
                views[length].append(text[:end])
    joblib.dump(views, cache / "prefixes.joblib")
    lengths = np.asarray(lengths)
    write_json(output / "input_statistics.json", {
        "median_wordpiece_tokens": float(np.median(lengths)),
        "mean_wordpiece_tokens": float(lengths.mean()),
        "lengths": {
            str(length): {
                "text_token_budget": length - 2,
                "truncated_document_fraction": float(np.mean(lengths > length - 2)),
                "mean_retained_token_fraction": float(np.mean(np.minimum(lengths, length - 2) / lengths)),
                "median_visible_regex_words": float(np.median([len(tokenize(text)) for text in texts])),
            } for length, texts in views.items()
        },
    })
    write_json(output / "protocol.json", {
        "split_seeds": list(SPLIT_SEEDS), "training_seeds": list(TRAIN_SEEDS),
        "split_protocol": "shuffled 80/10/10; no stratification",
        "rare_class_training_threshold": 100, "runs": planned_runs(),
        "bert": {"model": "google-bert/bert-base-uncased", "batch_size": args.batch_size,
                 "eval_batch_size": args.eval_batch_size, "learning_rate": 2e-5,
                 "weight_decay": 0.01, "warmup_ratio": 0.1,
                 "selection": "highest validation Macro-F1"},
        "logistic": {"C": [0.1, 1, 10], "max_iter": 2000, "solver": "lbfgs",
                     "selection": "highest validation Macro-F1"},
        "word2vec": {"dimensions": 100, "sg": 1, "window": 5, "min_count": 2,
                     "negative": 5, "sample": 0.001, "epochs": 10, "workers": 1},
        "position": {"tail": "last L-2 WordPieces",
                     "head_tail": "first half and last half of the L-2 text-token budget"},
    })
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "gensim", "torch", "transformers")
    write_json(output / "environment.json", {
        "python": platform.python_version(),
        "packages": {package: importlib.metadata.version(package) for package in packages},
    })
    write_json(output / "run_config.json", vars(args))
    print(f"Prepared {len(planned_runs())} configurations across five splits", flush=True)


def partition(output, split_seed):
    frame, labels, names = joblib.load(output / "cache/documents.joblib")
    manifest = read_json(output / f"split{split_seed}/splits.json")
    split = {key: np.asarray(manifest[key]) for key in ("train", "validation", "test")}
    return frame, labels, names, split


def reference_location(config, args):
    if config["split_seed"] != 42 or config["input_position"] in ("tail", "head_tail"):
        return None
    base, explorations = Path(args.base_results), Path(args.explorations)
    family, length, weighting, seed = (
        config["family"], config["max_length"], config["class_weighting"], config["training_seed"]
    )
    if family == "bert" and config["epochs"] == 3 and length == 256:
        return explorations / f"bert_len256_{weight_name(weighting)}_seed{seed}"
    if family == "sparse" and weighting == "uniform":
        if length == 0:
            return base / config["representation"]
        if length == 256:
            return explorations / f"{config['representation']}_prefix256"
    if family == "embedding" and weighting == "uniform":
        return explorations / f"word2vec_nyt_tfidf_seed{seed}"
    return None


def complete_row(config, location, output, names, labels, split, reused=False):
    row = {**read_json(location / "metrics.json"), **config,
           "run_directory": os.path.relpath(location, output), "reused": reused}
    rare = [name for i, name in enumerate(names) if np.sum(labels[split["train"]] == i) <= 100]
    report = read_json(location / "classification_report.json")
    row["rare_macro_f1"] = float(np.mean([report[name]["f1-score"] for name in rare]))
    row["rare_classes"] = rare
    row["zero_f1_classes"] = sum(report[name]["f1-score"] == 0 for name in names)
    return row


def available_row(config, args, names, labels, split):
    output = Path(args.output)
    reference = reference_location(config, args)
    location = reference or output / config["run_id"]
    if (location / "metrics.json").exists():
        return complete_row(config, location, output, names, labels, split, reference is not None)
    return None


def save_worker(output, name, rows):
    write_json(output / f"results_{name}.json", rows)
    print(f"{name}: saved {len(rows)} completed configurations", flush=True)


def run_bert_matrix(args):
    output = Path(args.output)
    rows = []
    encodings = {}
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    datasets = {seed: partition(output, seed) for seed in SPLIT_SEEDS}
    configs = [row for row in planned_runs() if row["family"] == "bert"]
    configs = configs[args.gpu_worker_index::args.gpu_workers]
    worker = (f"bert_worker{args.gpu_worker_index}" if args.gpu_workers > 1 else "bert")
    for config in configs:
        frame, labels, names, split = datasets[config["split_seed"]]
        row = available_row(config, args, names, labels, split)
        if row is None:
            key = config["max_length"], config["input_position"]
            if key not in encodings:
                encodings[key] = encode_documents(frame["text"].tolist(), tokenizer, *key)
            current = SimpleNamespace(**{**vars(args), "seed": config["training_seed"],
                                          "output": str(output / f"split{config['split_seed']}")})
            print(f"Starting {config['run_id']}", flush=True)
            next(run_bert(frame, labels, names, split, current, method=config["method"],
                          max_length=config["max_length"], epochs=config["epochs"],
                          position=config["input_position"], encodings=encodings[key],
                          balanced=config["class_weighting"] == "inverse-frequency"))
            row = complete_row(config, output / config["run_id"], output, names, labels, split)
        rows.append(row)
        save_worker(output, worker, rows)
    return rows


def run_classical_split(args, split_seed):
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.cpu_threads)
    output = Path(args.output)
    frame, labels, names, split = partition(output, split_seed)
    views = joblib.load(output / "cache/prefixes.joblib")
    current = SimpleNamespace(**{**vars(args), "seed": 42,
                                  "output": str(output / f"split{split_seed}")})
    rows = []
    for length in (0, 256, 512):
        visible = frame.copy()
        if length:
            visible["text"] = views[length]
        for weighting in WEIGHTINGS:
            configs = [entry("sparse", split_seed, 42, weighting, length, 0,
                             "prefix" if length else "full", representation) for representation in SPARSE]
            found = [available_row(config, args, names, labels, split) for config in configs]
            if all(row is not None for row in found):
                rows.extend(found)
            else:
                view = f"prefix{length}" if length else "full"
                suffix = f"_{view}_{weight_name(weighting)}"
                for result, config in zip(run_bow(visible, labels, names, split, current,
                                                 method_suffix=suffix,
                                                 balanced=weighting == "inverse-frequency"), configs):
                    rows.append(complete_row(config, output / config["run_id"], output,
                                             names, labels, split))
                    save_worker(output, f"classical_split{split_seed}", rows)
            save_worker(output, f"classical_split{split_seed}", rows)
    return rows


def run_embedding_split(args, split_seed):
    from threadpoolctl import threadpool_limits
    threadpool_limits(args.cpu_threads)
    output = Path(args.output)
    frame, labels, names, split = partition(output, split_seed)
    documents = [tokenize(text) for text in frame["text"]]
    df = Counter(word for i in split["train"] for word in set(documents[i]))
    n_train = len(split["train"])
    idf = {word: float(np.log((1 + n_train) / (1 + count)) + 1) for word, count in df.items()}
    unseen_idf = float(np.log(1 + n_train) + 1)
    rows = []
    for seed in TRAIN_SEEDS:
        configs = [entry("embedding", split_seed, seed, weighting, 0, 10, "full", "word2vec_nyt")
                   for weighting in WEIGHTINGS]
        found = [available_row(config, args, names, labels, split) for config in configs]
        if all(row is not None for row in found):
            rows.extend(found)
            save_worker(output, f"embeddings_split{split_seed}", rows)
            continue
        directory = output / f"split{split_seed}/word2vec_seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "word2vec.model"
        start = time.perf_counter()
        if split_seed == 42:
            source = (Path(args.base_results) / "word2vec_nyt/word2vec.model" if seed == 42 else
                      Path(args.explorations) / f"word2vec_nyt_mean_seed{seed}/word2vec.model")
            model = Word2Vec.load(str(source))
        elif model_path.exists():
            model = Word2Vec.load(str(model_path))
        else:
            print(f"Training NYT Word2Vec split {split_seed}, seed {seed}", flush=True)
            model = Word2Vec(sentences=[documents[i] for i in split["train"]],
                             vector_size=100, window=5, min_count=2, sg=1, negative=5,
                             sample=0.001, epochs=10, workers=1, seed=seed)
            model.save(str(model_path))
        features, coverage = mean_embeddings(documents, model.wv, idf=idf, unseen_idf=unseen_idf)
        write_json(directory / "pooling_config.json", {
            "fit_on": "NYT train", "training_documents": n_train,
            "formula": "log((1 + N) / (1 + document_frequency)) + 1",
            "unseen_word_idf": unseen_idf, "vocabulary_size": len(model.wv),
        })
        for config, old_row in zip(configs, found):
            if old_row is not None:
                rows.append(old_row)
            else:
                current = SimpleNamespace(**{**vars(args), "seed": seed,
                                              "output": str(output / f"split{split_seed}")})
                finish_classifier(config["method"], features, frame, labels, names, split,
                                  current, start, {"feature_dim": 100, "pooling": "tfidf",
                                                   "embedding_source": "NYT train (text only)",
                                                   "vocabulary_size": len(model.wv), **coverage},
                                  scale=True, balanced=config["class_weighting"] == "inverse-frequency")
                rows.append(complete_row(config, output / config["run_id"], output, names, labels, split))
            save_worker(output, f"embeddings_split{split_seed}", rows)
    return rows


def summarize(output):
    workers = sorted(output.glob("results_bert_worker*.json"))
    rows = ([row for path in workers for row in read_json(path)] if workers else
            read_json(output / "results_bert.json"))
    for seed in SPLIT_SEEDS:
        rows += read_json(output / f"results_classical_split{seed}.json")
        rows += read_json(output / f"results_embeddings_split{seed}.json")
    expected = {row["run_id"] for row in planned_runs()}
    assert len(rows) == len(expected) and {row["run_id"] for row in rows} == expected
    rows.sort(key=lambda row: row["run_id"])
    write_json(output / "results.json", rows)
    pd.DataFrame([{key: value for key, value in row.items() if key != "rare_classes"}
                  for row in rows]).to_csv(output / "results.csv", index=False)
    dimensions = ("family", "representation", "max_length", "epochs", "input_position", "class_weighting")
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in dimensions), []).append(row)
    summary = []
    for key, selected in grouped.items():
        group = dict(zip(dimensions, key))
        group["n_splits"] = len({row["split_seed"] for row in selected})
        group["n_runs"] = len(selected)
        group["splits"] = []
        for split_seed in sorted({row["split_seed"] for row in selected}):
            subset = [row for row in selected if row["split_seed"] == split_seed]
            point = {"split_seed": split_seed, "n_training_seeds": len(subset)}
            for metric in METRICS:
                values = [row[metric] for row in subset]
                point[metric + "_mean"] = float(np.mean(values))
                point[metric + "_training_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            group["splits"].append(point)
        for metric in METRICS:
            values = [point[metric + "_mean"] for point in group["splits"]]
            group[metric + "_mean"] = float(np.mean(values))
            group[metric + "_split_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            group[metric + "_training_std_mean"] = (
                float(np.mean([point[metric + "_training_std"] for point in group["splits"]]))
                if group["splits"][0][metric + "_training_std"] is not None else None
            )
        summary.append(group)
    summary.sort(key=lambda group: tuple(str(group[key]) for key in dimensions))
    write_json(output / "summary.json", {
        "n_configurations": len(rows), "n_new_configurations": sum(not row["reused"] for row in rows),
        "split_seeds": list(SPLIT_SEEDS), "training_seeds": list(TRAIN_SEEDS), "groups": summary,
    })
    print(f"Completed {len(rows)} configurations ({sum(not row['reused'] for row in rows)} new)", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nyt", default="data/nyt.csv")
    parser.add_argument("--bert-model", default="data/bert-base-uncased")
    parser.add_argument("--base-results", default="outputs/main")
    parser.add_argument("--explorations", default="outputs/explorations")
    parser.add_argument("--output", default="outputs/scaling")
    parser.add_argument("--task", choices=("all", "prepare", "bert", "cpu", "classical", "embeddings", "summarize"),
                        default="all")
    parser.add_argument("--split-seed", type=int, choices=SPLIT_SEEDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--cpu-workers", type=int, default=3)
    parser.add_argument("--gpu-workers", type=int, default=1)
    parser.add_argument("--gpu-worker-index", type=int, default=0)
    args = parser.parse_args()
    args.batch_size, args.eval_batch_size = 32, 128
    output = Path(args.output)
    if args.task != "summarize" and (
            args.task in ("prepare", "all") or not (output / "cache/documents.joblib").exists()):
        prepare(args)
    if args.task == "prepare":
        return
    if args.task == "bert":
        run_bert_matrix(args)
    elif args.task in ("classical", "embeddings"):
        runner = run_classical_split if args.task == "classical" else run_embedding_split
        for seed in ((args.split_seed,) if args.split_seed is not None else SPLIT_SEEDS):
            runner(args, seed)
    elif args.task == "summarize":
        summarize(output)
    else:
        with ProcessPoolExecutor(max_workers=args.cpu_workers) as pool:
            futures = []
            for seed in SPLIT_SEEDS:
                futures.append(pool.submit(run_classical_split, args, seed))
                futures.append(pool.submit(run_embedding_split, args, seed))
            if args.task == "all":
                run_bert_matrix(args)
            for future in futures:
                future.result()
        if args.task == "all":
            summarize(output)


if __name__ == "__main__":
    main()
