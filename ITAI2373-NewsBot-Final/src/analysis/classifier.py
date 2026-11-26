"""
Enhanced news classification (Module A).

Improvements over the midterm Linear SVM:
* **Soft-voting ensemble** – calibrated Linear SVM + Logistic Regression +
  Complement Naive Bayes on word *and* character TF-IDF features;
* **Confidence scoring** – calibrated probabilities, margin between the top
  two classes, a "needs human review" flag;
* **Multi-label output** – secondary categories above a threshold (a story
  about a football club's finances is both *sport* and *business*);
* **Explanations** – the words that pushed the decision, plus the most
  similar training articles;
* **Optional transformer head** – logistic regression on multilingual
  sentence embeddings, so non-English articles can be classified too.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_processing.feature_extractor import build_tfidf_union
from src.data_processing.text_preprocessor import TextPreprocessor
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


@dataclass
class Prediction:
    """Result of :meth:`NewsClassifier.predict_with_confidence`."""
    label: str
    confidence: float
    margin: float
    probabilities: Dict[str, float]
    secondary_labels: List[str]
    needs_review: bool

    def to_dict(self) -> dict:
        """Plain ``dict`` representation (JSON-friendly)."""
        return self.__dict__.copy()


class NewsClassifier:
    """
    Ensemble news classifier with confidence, multi-label and explanations.

    Example
    -------
    >>> clf = NewsClassifier().train(train_texts, train_labels)
    >>> clf.predict_with_confidence("Apple unveils a new chip ...").label
    'tech'
    """

    def __init__(self, low_confidence: float = 0.55, secondary_min: float = 0.20,
                 random_state: int = 42):
        """Create the object; parameters are described in the class docstring."""
        self.low_confidence = low_confidence
        self.secondary_min = secondary_min
        self.random_state = random_state
        self.pre = TextPreprocessor()
        self.model: Optional[Pipeline] = None
        self.explainer: Optional[Pipeline] = None  # word-level LR for explanations
        self.classes_: List[str] = []
        self._train_vectors = None
        self._train_texts: List[str] = []
        self._train_labels: List[str] = []
        self.train_seconds: float = 0.0

    # ---------------------------------------------------------------- build
    def _build(self) -> Pipeline:
        svm = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced",
                                               random_state=self.random_state), cv=3)
        lr = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
        nb = ComplementNB(alpha=0.3)
        ensemble = VotingClassifier([("svm", svm), ("lr", lr), ("nb", nb)],
                                    voting="soft", weights=[2, 1, 1])
        return Pipeline([("features", build_tfidf_union()), ("clf", ensemble)])

    def train(self, texts: Sequence[str], labels: Sequence[str]) -> "NewsClassifier":
        """Fit the ensemble and the explanation model on raw article texts."""
        start = time.time()
        norm = self.pre.batch_normalize(texts)
        self.model = self._build().fit(norm, labels)
        self.classes_ = [str(c) for c in self.model.classes_]
        self.explainer = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("lr", LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")),
        ]).fit(norm, labels)
        self._train_vectors = self.explainer.named_steps["tfidf"].transform(norm)
        self._train_texts = list(texts)
        self._train_labels = list(labels)
        self.train_seconds = time.time() - start
        log.info("Classifier trained on %d articles in %.1fs", len(norm), self.train_seconds)
        return self

    # -------------------------------------------------------------- predict
    def predict(self, texts: Sequence[str]) -> List[str]:
        """Predicted label for each raw text."""
        return list(self.model.predict(self.pre.batch_normalize(texts)))

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Class probabilities for each raw text (columns follow ``classes_``)."""
        return self.model.predict_proba(self.pre.batch_normalize(texts))

    def predict_with_confidence(self, text: str) -> Prediction:
        """Primary label, calibrated confidence, alternatives and a review flag."""
        if not text or not text.strip():
            raise ValueError("Cannot classify an empty article")
        probs = self.predict_proba([text])[0]
        order = np.argsort(-probs)
        top, second = probs[order[0]], probs[order[1]]
        classes = [str(c) for c in self.classes_]
        prob_map = {classes[i]: round(float(probs[i]), 4) for i in order}
        secondary = [classes[i] for i in order[1:] if probs[i] >= self.secondary_min]
        return Prediction(
            label=classes[order[0]],
            confidence=round(float(top), 4),
            margin=round(float(top - second), 4),
            probabilities=prob_map,
            secondary_labels=secondary,
            needs_review=bool(top < self.low_confidence or (top - second) < 0.15),
        )

    # -------------------------------------------------------------- explain
    def explain_prediction(self, text: str, top_n: int = 8, n_similar: int = 3) -> dict:
        """
        Explain a decision with (1) the terms that contributed most to the
        predicted class in the word-level model and (2) similar training
        articles (nearest neighbours in TF-IDF space).
        """
        pred = self.predict_with_confidence(text)
        norm = self.pre.normalize(text)
        tfidf = self.explainer.named_steps["tfidf"]
        lr = self.explainer.named_steps["lr"]
        vec = tfidf.transform([norm])
        cls_idx = list(lr.classes_).index(pred.label)
        contrib = vec.multiply(lr.coef_[cls_idx]).toarray()[0]
        names = tfidf.get_feature_names_out()
        top_idx = [i for i in np.argsort(-contrib)[:top_n] if contrib[i] > 0]
        sims = (self._train_vectors @ vec.T).toarray().ravel()
        nn = np.argsort(-sims)[:n_similar]
        return {
            "prediction": pred.to_dict(),
            "key_terms": [(names[i], round(float(contrib[i]), 4)) for i in top_idx],
            "similar_training_articles": [
                {"label": self._train_labels[i], "similarity": round(float(sims[i]), 3),
                 "snippet": self._train_texts[i][:160] + "..."} for i in nn],
        }

    # ------------------------------------------------------------- evaluate
    def evaluate(self, texts: Sequence[str], labels: Sequence[str]) -> dict:
        """Accuracy, macro-F1, per-class report and confusion matrix."""
        start = time.time()
        pred = self.predict(texts)
        secs = time.time() - start
        labels_sorted = sorted(set(labels) | set(pred))
        return {
            "n": len(labels),
            "accuracy": round(accuracy_score(labels, pred), 4),
            "macro_f1": round(f1_score(labels, pred, average="macro"), 4),
            "report": classification_report(labels, pred, output_dict=True, zero_division=0),
            "labels": labels_sorted,
            "confusion_matrix": confusion_matrix(labels, pred, labels=labels_sorted).tolist(),
            "ms_per_article": round(1000 * secs / max(len(labels), 1), 2),
        }

    def calibration_table(self, texts: Sequence[str], labels: Sequence[str], bins: int = 5) -> pd.DataFrame:
        """Reliability table: accuracy within confidence bins (is 0.9 really 90%?)."""
        probs = self.predict_proba(texts)
        conf = probs.max(axis=1)
        pred = np.array(self.classes_)[probs.argmax(axis=1)]
        correct = pred == np.asarray(labels)
        edges = np.linspace(0, 1, bins + 1)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                             "mean_confidence": round(float(conf[mask].mean()), 3),
                             "accuracy": round(float(correct[mask].mean()), 3)})
        return pd.DataFrame(rows)

    def cross_validate(self, texts: Sequence[str], labels: Sequence[str], folds: int = 5) -> dict:
        """Stratified k-fold macro-F1 of the full pipeline (no leakage)."""
        norm = self.pre.batch_normalize(texts)
        cv = StratifiedKFold(folds, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self._build(), norm, labels, cv=cv, scoring="f1_macro", n_jobs=1)
        return {"folds": folds, "mean": round(float(scores.mean()), 4), "std": round(float(scores.std()), 4)}

    # -------------------------------------------------------------- persist
    def save(self, path) -> None:
        """Serialise this object to ``path`` with joblib (compressed)."""
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "NewsClassifier":
        """Load an object previously written with :meth:`save`."""
        return joblib.load(path)


def compare_baselines(train_texts, train_labels, test_texts, test_labels) -> pd.DataFrame:
    """Compare single models against the ensemble on the same split."""
    pre = TextPreprocessor()
    Xtr, Xte = pre.batch_normalize(train_texts), pre.batch_normalize(test_texts)
    candidates = {
        "Naive Bayes (word TF-IDF)": Pipeline([("t", TfidfVectorizer(sublinear_tf=True, min_df=2)),
                                                ("m", ComplementNB(alpha=0.3))]),
        "Logistic Regression (word TF-IDF)": Pipeline([("t", TfidfVectorizer(sublinear_tf=True, min_df=2)),
                                                       ("m", LogisticRegression(C=8, max_iter=3000))]),
        "Linear SVM (word TF-IDF) – midterm style": Pipeline([
            ("t", TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2))),
            ("m", LinearSVC(C=1, class_weight="balanced"))]),
        "Linear SVM (word+char TF-IDF)": Pipeline([("t", build_tfidf_union()),
                                                   ("m", LinearSVC(C=0.5, class_weight="balanced"))]),
    }
    rows = []
    for name, pipe in candidates.items():
        t0 = time.time()
        pipe.fit(Xtr, train_labels)
        pred = pipe.predict(Xte)
        rows.append({"model": name, "accuracy": accuracy_score(test_labels, pred),
                     "macro_f1": f1_score(test_labels, pred, average="macro"),
                     "train_s": time.time() - t0})
    return pd.DataFrame(rows).round(4)


class EmbeddingClassifier:
    """
    Logistic regression on multilingual sentence embeddings. Because the
    embedding space is shared across languages, a model trained on English
    BBC articles can label Spanish/German/French news (zero-shot transfer).
    """

    def __init__(self, embedder):
        """Create the object; parameters are described in the class docstring."""
        self.embedder = embedder
        self.model = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")

    def train(self, texts, labels) -> "EmbeddingClassifier":
        """Fit the model on texts and labels; returns ``self`` for chaining."""
        self.model.fit(self.embedder.encode(list(texts)), labels)
        return self

    def predict(self, texts) -> List[str]:
        """Predicted label for each raw text."""
        return list(self.model.predict(self.embedder.encode(list(texts))))

    def predict_proba(self, texts) -> np.ndarray:
        """Class probabilities for each raw text (columns follow ``classes_``)."""
        return self.model.predict_proba(self.embedder.encode(list(texts)))
