"""Check split isolation, embedding averaging, and executable classifiers."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from textcls.classical import fit_logistic, mean_embeddings, run_bow, run_embeddings
from textcls.data import prepare_data, shuffled_split, tokenize
from textcls.metrics import evaluate
from textcls.neural import training_class_weights


class ExperimentTests(unittest.TestCase):
    def test_partition_is_reproducible_and_exhaustive(self):
        split = shuffled_split(100, 42)
        self.assertEqual([len(split[key]) for key in split], [80, 10, 10])
        joined = np.concatenate(list(split.values()))
        np.testing.assert_array_equal(np.sort(joined), np.arange(100))
        np.testing.assert_array_equal(split["train"], shuffled_split(100, 42)["train"])

    def test_average_counts_repeated_tokens_and_excludes_unknowns(self):
        vectors = {"apple": np.ones(100, dtype=np.float32),
                   "banana": np.full(100, 4.0, dtype=np.float32)}
        features, coverage = mean_embeddings([
            ["apple", "apple", "banana", "unknown"], ["unknown"]
        ], vectors)
        np.testing.assert_allclose(features[0], 2.0)
        np.testing.assert_allclose(features[1], 0.0)
        self.assertEqual(coverage["zero_vector_documents"], 1)
        self.assertEqual(coverage["token_coverage"], 3 / 5)

    def test_metrics_average_all_classes(self):
        scores = evaluate([0, 0, 1, 1], [0, 0, 0, 0], 2)
        self.assertEqual(scores["accuracy"], 0.5)
        self.assertAlmostEqual(scores["macro_f1"], 1 / 3)

    def test_tfidf_weighted_average_preserves_occurrences(self):
        vectors = {"apple": np.ones(100, dtype=np.float32),
                   "banana": np.full(100, 4.0, dtype=np.float32)}
        features, coverage = mean_embeddings(
            [["apple", "apple", "banana", "unknown"]], vectors,
            idf={"apple": 1.0, "banana": 2.0}, unseen_idf=3.0,
        )
        np.testing.assert_allclose(features[0], 2.5)
        self.assertEqual(coverage["valid_tokens"], 3)

    def test_class_weights_use_training_labels_only(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        training = np.array([0, 1, 2, 3])
        expected = np.array([2 / 3, 2])
        np.testing.assert_allclose(training_class_weights(labels, training, 2), expected)
        labels[4:] = 0
        np.testing.assert_allclose(training_class_weights(labels, training, 2), expected)

    def test_test_labels_do_not_affect_model_selection(self):
        rng = np.random.RandomState(42)
        features = rng.normal(size=(100, 5))
        labels = (features[:, 0] > 0).astype(int)
        split = shuffled_split(len(labels))
        first, best_first, _ = fit_logistic(features, labels, ["a", "b"], split, 42)
        altered = labels.copy()
        altered[split["test"]] = 1 - altered[split["test"]]
        second, best_second, _ = fit_logistic(features, altered, ["a", "b"], split, 42)
        self.assertEqual(best_first, best_second)
        np.testing.assert_array_equal(first.coef_, second.coef_)

    def test_classical_pipeline_excludes_held_out_training_words(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = shuffled_split(90)
            texts = []
            labels = []
            for i in range(90):
                topic = ("sport", "science", "finance")[i % 3]
                texts.append(f"{topic} {topic} news report article entry {i}")
                labels.append(topic)
            for i in np.concatenate([split["validation"], split["test"]]):
                texts[i] += " heldoutmarker heldoutmarker"
            pd.DataFrame({"text": texts, "label": labels}).to_csv(root / "nyt.csv", index=False)
            pd.DataFrame({"text": texts[:9]}).to_csv(root / "ag.csv", index=False)
            with (root / "glove.txt").open("w") as stream:
                for i, word in enumerate(("sport", "science", "finance", "news", "report", "article")):
                    vector = np.zeros(100)
                    vector[i] = 1
                    stream.write(word + " " + " ".join(map(str, vector)) + "\n")
            args = SimpleNamespace(output=str(root / "out"), seed=42,
                                   glove=str(root / "glove.txt"), ag_news=str(root / "ag.csv"),
                                   w2v_epochs=2, workers=1)
            frame, y, names, partition = prepare_data(root / "nyt.csv", args.output, 42)
            results = list(run_bow(frame, y, names, partition, args))
            results += list(run_embeddings(frame, y, names, partition, args))
            self.assertEqual(len(results), 6)
            import joblib
            vectorizer = joblib.load(root / "out/bow_count/vectorizer.joblib")
            self.assertNotIn("heldoutmarker", vectorizer.vocabulary_)
            vectors = Word2Vec.load(str(root / "out/word2vec_nyt/word2vec.model"))
            self.assertNotIn("heldoutmarker", vectors.wv)
            for result in results:
                predictions = pd.read_csv(root / "out" / result["method"] / "test_predictions.csv")
                np.testing.assert_array_equal(predictions["row_id"], partition["test"])
                if result["method"].startswith("bow"):
                    self.assertEqual(result["test_accuracy"], 1.0)

    def test_duplicate_and_conflicting_news_are_removed_before_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [{"text": f"News article {i}", "label": str(i % 2)} for i in range(20)]
            rows += [rows[0].copy(), {"text": "Ambiguous news", "label": "0"},
                     {"text": "Ambiguous news", "label": "1"}]
            pd.DataFrame(rows).to_csv(root / "nyt.csv", index=False)
            frame, _, _, _ = prepare_data(root / "nyt.csv", root / "out", 42)
            self.assertEqual(len(frame), 20)
            self.assertEqual(frame["text"].duplicated().sum(), 0)
            self.assertNotIn("Ambiguous news", frame["text"].tolist())
            import json
            info = json.loads((root / "out/data_info.json").read_text())
            self.assertEqual(info["conflicting_rows_removed"], 2)
            self.assertEqual(info["duplicate_rows_removed"], 1)


if __name__ == "__main__":
    unittest.main()
