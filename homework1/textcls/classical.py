"""Bag-of-words and averaged 100-dimensional word vector classifiers."""

import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import tokenize, write_json
from .metrics import evaluate, save_test_outputs


def fit_logistic(features, labels, names, split, seed, scale=False, balanced=False):
    best_score = -1.0
    best_model = None
    trials = []
    for c in (0.1, 1.0, 10.0):
        classifier = LogisticRegression(C=c, max_iter=2000, solver="lbfgs", random_state=seed,
                                        class_weight="balanced" if balanced else None)
        model = make_pipeline(StandardScaler(), classifier) if scale else classifier
        model.fit(features[split["train"]], labels[split["train"]])
        prediction = model.predict(features[split["validation"]])
        metrics = evaluate(labels[split["validation"]], prediction, len(names))
        trials.append({"C": c, **metrics})
        if metrics["macro_f1"] > best_score:
            best_score, best_model, best_trial = metrics["macro_f1"], model, trials[-1]
    return best_model, best_trial, trials


def finish_classifier(method, features, frame, labels, names, split, args, start,
                      representation, scale=False, balanced=False):
    model, best, trials = fit_logistic(features, labels, names, split, args.seed,
                                     scale=scale, balanced=balanced)
    prediction = model.predict(features[split["test"]])
    test_metrics = evaluate(labels[split["test"]], prediction, len(names))
    directory = Path(args.output) / method
    directory.mkdir(parents=True, exist_ok=True)
    if balanced:
        counts = np.bincount(labels[split["train"]], minlength=len(names))
        weights = len(split["train"]) / (len(names) * counts)
        write_json(directory / "class_weights.json", dict(zip(names, weights.tolist())))
    joblib.dump(model, directory / "classifier.joblib")
    write_json(directory / "validation_trials.json", trials)
    save_test_outputs(args.output, method, frame, split["test"],
                      labels[split["test"]], prediction, names)
    result = {
        "method": method,
        "validation_accuracy": best["accuracy"],
        "validation_macro_f1": best["macro_f1"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "selected_C": best["C"],
        "seconds": time.perf_counter() - start,
        "class_weighting": "inverse-frequency" if balanced else "uniform",
        **representation,
    }
    write_json(directory / "metrics.json", result)
    return result


def run_bow(frame, labels, names, split, args, *, method_suffix="", balanced=False):
    for representation in ("bow_binary", "bow_count", "bow_tfidf"):
        method = representation + method_suffix
        start = time.perf_counter()
        params = dict(tokenizer=tokenize, token_pattern=None, lowercase=False)
        if representation == "bow_tfidf":
            vectorizer = TfidfVectorizer(**params, norm="l2", smooth_idf=True)
        else:
            vectorizer = CountVectorizer(**params, binary=representation == "bow_binary")
        vectorizer.fit(frame.iloc[split["train"]]["text"])
        features = vectorizer.transform(frame["text"])
        directory = Path(args.output) / method
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, directory / "vectorizer.joblib")
        result = finish_classifier(
            method, features, frame, labels, names, split, args, start,
            {"feature_dim": features.shape[1], "vocabulary_fit_on": "NYT train"},
            balanced=balanced,
        )
        print(f"{method}: {result}", flush=True)
        yield result


def load_glove(path, needed, dimension=100):
    vectors = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            word, _, numbers = line.partition(" ")
            if word in needed:
                vector = np.fromstring(numbers, sep=" ", dtype=np.float32)
                if len(vector) != dimension:
                    raise ValueError(f"GloVe vector for {word!r} is not {dimension} dimensional.")
                vectors[word] = vector
    if not vectors:
        raise ValueError("No document words occur in the GloVe file.")
    return vectors


def mean_embeddings(documents, vectors, dimension=100, *, idf=None, unseen_idf=1.0):
    features = np.zeros((len(documents), dimension), dtype=np.float32)
    valid, total, empty = 0, 0, 0
    for i, words in enumerate(documents):
        counts = Counter(words)
        known = [(word, count) for word, count in counts.items() if word in vectors]
        n_known = sum(count for _, count in known)
        if n_known:
            if idf is None:
                features[i] = sum(vectors[word] * count for word, count in known) / n_known
            else:
                weights = [(word, count * idf.get(word, unseen_idf)) for word, count in known]
                features[i] = sum(vectors[word] * weight for word, weight in weights) / sum(
                    weight for _, weight in weights
                )
        else:
            empty += 1
        valid += n_known
        total += len(words)
    return features, {
        "token_coverage": valid / total if total else 0.0,
        "zero_vector_documents": empty,
        "valid_tokens": valid,
        "total_tokens": total,
    }


def run_embeddings(frame, labels, names, split, args):
    documents = [tokenize(text) for text in frame["text"]]
    for method in ("glove", "word2vec_ag_news", "word2vec_nyt"):
        start = time.perf_counter()
        directory = Path(args.output) / method
        directory.mkdir(parents=True, exist_ok=True)
        if method == "glove":
            needed = {word for words in documents for word in words}
            vectors = load_glove(args.glove, needed)
            source = "Stanford glove.6B.100d (fixed pretrained vectors)"
            vocabulary_size = len(vectors)
            joblib.dump(vectors, directory / "glove_vectors.joblib")
        else:
            if method == "word2vec_ag_news":
                from .data import read_table
                ag_news = read_table(args.ag_news, require_labels=False)
                corpus = [tokenize(text) for text in ag_news["text"]]
                source = "AG News train (text only)"
            else:
                corpus = [documents[i] for i in split["train"]]
                source = "NYT train (text only)"
            print(f"Training {method}: {len(corpus)} documents, 100 dimensions, "
                  f"{args.w2v_epochs} epochs", flush=True)
            embedding_model = Word2Vec(
                sentences=corpus, vector_size=100, window=5, min_count=2,
                sg=1, negative=5, sample=1e-3, epochs=args.w2v_epochs,
                workers=args.workers, seed=args.seed,
            )
            embedding_model.save(str(directory / "word2vec.model"))
            vectors = embedding_model.wv
            vocabulary_size = len(vectors)
            del corpus
        features, coverage = mean_embeddings(documents, vectors)
        coverage_by_split = {
            key: mean_embeddings([documents[i] for i in indices], vectors)[1]
            for key, indices in split.items()
        }
        np.save(directory / "document_vectors.npy", features)
        write_json(directory / "coverage.json", coverage_by_split)
        result = finish_classifier(
            method, features, frame, labels, names, split, args, start,
            {"feature_dim": 100, "embedding_source": source,
             "vocabulary_size": vocabulary_size, **coverage}, scale=True
        )
        print(f"{method}: {result}", flush=True)
        yield result
