"""Command line entry point for the seven shared-split experiments."""

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import pandas as pd

from .data import prepare_data, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the NYT text classification homework.")
    parser.add_argument("--nyt", default="data/nyt.csv")
    parser.add_argument("--ag-news", default="data/ag_news.csv")
    parser.add_argument("--glove", default="data/glove.6B.100d.txt")
    parser.add_argument("--bert-model", default="google-bert/bert-base-uncased")
    parser.add_argument("--output", default="outputs/main")
    parser.add_argument("--tasks", nargs="+", choices=("bow", "embeddings", "bert"),
                        default=["bow", "embeddings", "bert"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w2v-epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cpu or cuda:0.")
    return parser.parse_args(argv)


def save_result(output, result):
    path = Path(output) / "results.json"
    results = json.loads(path.read_text()) if path.exists() else []
    results = [item for item in results if item["method"] != result["method"]] + [result]
    order = ["bow_binary", "bow_count", "bow_tfidf", "glove",
             "word2vec_ag_news", "word2vec_nyt", "bert"]
    results.sort(key=lambda item: order.index(item["method"]))
    write_json(path, results)
    pd.DataFrame(results).to_csv(Path(output) / "results.csv", index=False)


def main(argv=None):
    args = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    frame, labels, names, split = prepare_data(args.nyt, args.output, args.seed)
    write_json(Path(args.output) / "run_config.json", vars(args))
    packages = ["numpy", "pandas", "scipy", "scikit-learn", "gensim",
                "torch", "transformers", "huggingface-hub"]
    write_json(Path(args.output) / "environment.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
    })
    print(f"NYT: {len(frame)} rows, {len(names)} classes; "
          f"splits={ {key: len(value) for key, value in split.items()} }", flush=True)
    for task in args.tasks:
        write_json(Path(args.output) / f"run_config_{task}.json", vars(args))
        if task == "bow":
            from .classical import run_bow
            runner = run_bow
        elif task == "embeddings":
            from .classical import run_embeddings
            runner = run_embeddings
        else:
            from .neural import run_bert
            runner = run_bert
        for result in runner(frame, labels, names, split, args):
            save_result(args.output, result)
    print(f"Results saved to {Path(args.output).resolve()}", flush=True)
