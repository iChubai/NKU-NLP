#!/usr/bin/env python3
"""Download public NYT, AG News, GloVe 6B, and BERT resources."""

import argparse
import gzip
import json
import shutil
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import snapshot_download

NYT_BASE = "https://raw.githubusercontent.com/yumeng5/WeSHClass/master/nyt/"
AG_NEWS_URL = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/train-00000-of-00001.parquet"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DOWNLOAD_URL = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"


def download(url, path):
    path = Path(path)
    if path.exists():
        print(f"Using {path}", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    print(f"Downloading {url}", flush=True)
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        received = 0
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                stream.write(chunk)
                received += len(chunk)
                if received // (128 * 1024 * 1024) > (received - len(chunk)) // (128 * 1024 * 1024):
                    print(f"  {received / 1024**2:.0f} MiB", flush=True)
    temporary.replace(path)


def prepare_nyt(directory):
    raw = directory / "nyt_source"
    for name in ("dataset.txt", "labels.txt", "label_hier.txt"):
        download(NYT_BASE + name, raw / name)
    texts = (raw / "dataset.txt").read_text(encoding="utf-8").splitlines()
    labels = (raw / "labels.txt").read_text(encoding="utf-8").splitlines()
    if len(texts) != len(labels):
        raise ValueError("The NYT documents and labels have different row counts.")
    pd.DataFrame({"text": texts, "label": labels}).to_csv(directory / "nyt.csv", index=False)
    return {"name": "NYT / WeSHClass", "url": NYT_BASE,
            "text_file": "dataset.txt", "label_file": "labels.txt",
            "label_level": "original topic labels", "rows": len(texts),
            "classes": len(set(labels)), "prepared_file": "nyt.csv"}


def prepare_ag_news(directory):
    path = directory / "ag_news_train.parquet"
    download(AG_NEWS_URL, path)
    frame = pd.read_parquet(path)
    frame[["text", "label"]].to_csv(directory / "ag_news.csv", index=False)
    return {"name": "AG News", "url": AG_NEWS_URL, "split": "train",
            "rows": len(frame), "prepared_file": "ag_news.csv",
            "usage": "Unsupervised Word2Vec training; labels unused"}


def prepare_glove(directory):
    target = directory / "glove.6B.100d.txt"
    if not target.exists():
        archive = directory / "glove-wiki-gigaword-100.gz"
        download(GLOVE_DOWNLOAD_URL, archive)
        temporary = target.with_name(target.name + ".part")
        with gzip.open(archive, "rb") as source:
            vocabulary, dimension = map(int, source.readline().split())
            if (vocabulary, dimension) != (400000, 100):
                raise ValueError("Expected the 400,000-word, 100-dimensional GloVe 6B resource.")
            with temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        temporary.replace(target)
    return {"name": "Stanford GloVe 6B", "url": GLOVE_URL,
            "download_url": GLOVE_DOWNLOAD_URL,
            "format": "Gensim mirror of Wikipedia 2014 + Gigaword 5; word2vec header removed",
            "dimension": 100, "prepared_file": target.name}


def prepare_bert(directory):
    snapshot_download(
        repo_id="google-bert/bert-base-uncased", local_dir=directory / "bert-base-uncased",
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                        "tokenizer_config.json", "vocab.txt"],
    )
    return {"name": "google-bert/bert-base-uncased",
            "url": "https://huggingface.co/google-bert/bert-base-uncased",
            "prepared_directory": "bert-base-uncased"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--assets", nargs="+", choices=("nyt", "ag-news", "glove", "bert"),
                        default=["nyt", "ag-news", "glove", "bert"])
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.data_dir / "sources.json"
    sources = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    runners = {"nyt": prepare_nyt, "ag-news": prepare_ag_news,
               "glove": prepare_glove, "bert": prepare_bert}
    for name in args.assets:
        sources[name] = runners[name](args.data_dir)
        manifest_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared {name}: {sources[name]}", flush=True)


if __name__ == "__main__":
    main()
