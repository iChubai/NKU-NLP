"""Check balanced classifier fitting and exact token-position construction."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizerFast

from textcls.classical import fit_logistic
from textcls.neural import encode_documents


class ScalingTests(unittest.TestCase):
    def test_balanced_logistic_matches_explicit_training_weights(self):
        random = np.random.RandomState(17)
        features = random.normal(size=(60, 4))
        labels = np.array([0] * 30 + [1] * 10 + [0, 1] * 10)
        split = {"train": np.arange(40), "validation": np.arange(40, 50), "test": np.arange(50, 60)}
        model, selected, _ = fit_logistic(features, labels, ["a", "b"], split, 42,
                                         scale=True, balanced=True)
        counts = np.bincount(labels[split["train"]])
        weights = 40 / (2 * counts[labels[split["train"]]])
        reference = make_pipeline(StandardScaler(), LogisticRegression(
            C=selected["C"], max_iter=2000, solver="lbfgs", random_state=42))
        reference.fit(features[split["train"]], labels[split["train"]],
                      logisticregression__sample_weight=weights)
        np.testing.assert_allclose(model[-1].coef_, reference[-1].coef_, rtol=1e-12, atol=1e-12)
        altered = labels.copy()
        altered[split["test"]] = 0
        second, other_selection, _ = fit_logistic(features, altered, ["a", "b"], split, 42,
                                                 scale=True, balanced=True)
        self.assertEqual(selected, other_selection)
        np.testing.assert_array_equal(model[-1].coef_, second[-1].coef_)

    def test_position_views_preserve_budget_special_tokens_and_short_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            vocab = Path(temporary) / "vocab.txt"
            vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                        "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
                                        "eta", "theta", "iota", "kappa"]) + "\n", encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
            texts = ["alpha beta gamma delta epsilon zeta eta theta iota kappa", "alpha beta"]
            tokens = tokenizer(texts[0], add_special_tokens=False)["input_ids"]
            expected = {"prefix": tokens[:6], "tail": tokens[-6:],
                        "head_tail": tokens[:3] + tokens[-3:]}
            for position in expected:
                encoded = encode_documents(texts, tokenizer, 8, position)
                self.assertEqual(tuple(encoded["input_ids"].shape), (2, 8))
                self.assertEqual(encoded["input_ids"][0].tolist(),
                                 [tokenizer.cls_token_id, *expected[position], tokenizer.sep_token_id])
                self.assertEqual(encoded["attention_mask"][1].tolist(), [1, 1, 1, 1, 0, 0, 0, 0])
                self.assertEqual(encoded["input_ids"][1, :4].tolist(),
                                 tokenizer(texts[1])["input_ids"])


if __name__ == "__main__":
    unittest.main()
