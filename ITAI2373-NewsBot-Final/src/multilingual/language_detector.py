"""
Language identification (Module C, course Module 11).

A character n-gram classifier trained on the Webz multilingual news corpus
(our own model, no external API): TF-IDF over 1–4 character n-grams +
multinomial logistic regression. It returns calibrated confidence, works on
short snippets and can split a mixed-language text into per-sentence
segments (code-switching). ``langdetect`` is used only as an optional
baseline for comparison.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_processing.text_preprocessor import TextPreprocessor
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German",
                  "it": "Italian", "pt": "Portuguese", "und": "Undetermined"}


class LanguageDetector:
    """Character n-gram language identifier with confidence scores."""

    def __init__(self, min_confidence: float = 0.6, max_chars: int = 600):
        """Create the object; parameters are described in the class docstring."""
        self.min_confidence = min_confidence
        self.max_chars = max_chars
        self.model = Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=3,
                                      max_features=150000, sublinear_tf=True, lowercase=True)),
            ("lr", LogisticRegression(C=10, max_iter=2000)),
        ])
        self.fitted = False

    # -------------------------------------------------------------- train
    @staticmethod
    def make_training_snippets(texts: Iterable[str], labels: Iterable[str], per_text: int = 3,
                               seed: int = 42) -> tuple:
        """Cut articles into sentence-sized snippets so the model also learns short inputs."""
        rng = np.random.default_rng(seed)
        X, y = [], []
        for text, lab in zip(texts, labels):
            sents = [s for s in TextPreprocessor.split_sentences(text) if len(s) > 25]
            if not sents:
                continue
            for i in rng.choice(len(sents), size=min(per_text, len(sents)), replace=False):
                X.append(sents[i][:300])
                y.append(lab)
        return X, y

    def train(self, texts: List[str], labels: List[str]) -> "LanguageDetector":
        """Fit the model on texts and labels; returns ``self`` for chaining."""
        self.model.fit([t[: self.max_chars] for t in texts], labels)
        self.fitted = True
        log.info("Language detector trained on %d samples (%s)", len(texts),
                 dict(Counter(labels)))
        return self

    # ------------------------------------------------------------- detect
    def detect_language(self, text: str) -> dict:
        """Most likely language, confidence and the top alternatives."""
        if not self.fitted:
            raise RuntimeError("LanguageDetector is not trained – call train() or load()")
        snippet = (text or "").strip()[: self.max_chars]
        if len(snippet) < 3:
            return {"language": "und", "name": LANGUAGE_NAMES["und"], "confidence": 0.0, "alternatives": {}}
        probs = self.model.predict_proba([snippet])[0]
        classes = [str(c) for c in self.model.classes_]  # plain str, not numpy.str_
        order = np.argsort(-probs)
        lang = classes[order[0]] if probs[order[0]] >= self.min_confidence else "und"
        return {"language": lang, "name": LANGUAGE_NAMES.get(lang, lang),
                "confidence": round(float(probs[order[0]]), 4),
                "alternatives": {classes[i]: round(float(probs[i]), 4) for i in order[1:3]},
                "reliable": bool(probs[order[0]] >= self.min_confidence and len(snippet) >= 20)}

    def detect_batch(self, texts: Iterable[str]) -> List[str]:
        """Most likely language for each text (no confidence filtering)."""
        return [str(p) for p in self.model.predict([(t or "")[: self.max_chars] for t in texts])]

    def detect_segments(self, text: str) -> dict:
        """
        Per-sentence detection for code-switched / mixed-language text.
        Returns the segments and the language shares.
        """
        segs = []
        for s in TextPreprocessor.split_sentences(text):
            r = self.detect_language(s)
            segs.append({"sentence": s, "language": r["language"], "confidence": r["confidence"]})
        shares = Counter(str(s["language"]) for s in segs)
        total = sum(shares.values()) or 1
        return {"mixed": len([l for l in shares if l != "und"]) > 1,
                "shares": {k: round(v / total, 3) for k, v in shares.items()}, "segments": segs}

    # ----------------------------------------------------------- evaluate
    def evaluate(self, texts: List[str], labels: List[str], lengths: Iterable[int] = (20, 50, 100, 300)) -> dict:
        """Accuracy overall and when only the first N characters are available."""
        from sklearn.metrics import accuracy_score, f1_score
        out = {}
        for n in lengths:
            pred = self.model.predict([t[:n] for t in texts])
            out[f"first_{n}_chars"] = {"accuracy": round(accuracy_score(labels, pred), 4),
                                       "macro_f1": round(f1_score(labels, pred, average="macro"), 4)}
        try:  # optional baseline
            from langdetect import DetectorFactory, detect
            DetectorFactory.seed = 0
            base = []
            for t in texts:
                try:
                    base.append(detect(t[:100]))
                except Exception:
                    base.append("und")
            out["langdetect_baseline_first_100_chars"] = {"accuracy": round(accuracy_score(labels, base), 4)}
        except ImportError:
            pass
        return out

    def save(self, path) -> None:
        """Serialise this object to ``path`` with joblib (compressed)."""
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "LanguageDetector":
        """Load an object previously written with :meth:`save`."""
        return joblib.load(path)
