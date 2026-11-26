"""
Feature extraction: TF-IDF (word + character), handcrafted style/sentiment
features and dense document vectors (LSA, or transformer embeddings via
``src.language_models.embeddings``).
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


def build_tfidf_union(max_word_features: int = 40000, max_char_features: int = 40000) -> FeatureUnion:
    """
    Word 1–2 grams + character 3–5 grams. Character n-grams make the
    classifier robust to spelling variants, unseen inflections and OCR noise
    (an improvement over the word-only midterm model).
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                                 max_features=max_word_features)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True, max_features=max_char_features)),
    ])


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Length, punctuation and numeric-density features from the *clean* text."""

    names = ["log_chars", "log_words", "avg_word_len", "digit_ratio", "quote_ratio",
             "question_ratio", "upper_ratio"]

    def fit(self, X, y=None):
        """No-op fit (stateless transformer); returns ``self``."""
        return self

    def transform(self, X: Iterable[str]):
        """Transform the input texts into feature vectors."""
        rows = []
        for t in X:
            t = t or ""
            n_chars = max(len(t), 1)
            words = t.split()
            n_words = max(len(words), 1)
            rows.append([
                np.log1p(n_chars), np.log1p(n_words),
                sum(len(w) for w in words) / n_words,
                sum(c.isdigit() for c in t) / n_chars,
                t.count('"') / n_words,
                t.count("?") / n_words,
                sum(c.isupper() for c in t) / n_chars,
            ])
        return sparse.csr_matrix(np.asarray(rows, dtype=float))


class LSAEmbedder:
    """
    Dense document vectors via TF-IDF + truncated SVD (Latent Semantic Analysis).
    This is the offline fallback for transformer sentence embeddings.
    """

    def __init__(self, n_components: int = 256, random_state: int = 42):
        """Create the object; parameters are described in the class docstring."""
        self.n_components = n_components
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True,
                                      max_features=60000)),
            ("svd", TruncatedSVD(n_components=n_components, random_state=random_state)),
        ])
        self.fitted = False

    def fit(self, texts: List[str]) -> "LSAEmbedder":
        """Learn the TF-IDF vocabulary and SVD projection from ``texts``."""
        n = min(self.n_components, max(2, len(texts) - 1))
        self.pipeline.set_params(svd__n_components=n)
        self.pipeline.fit(texts)
        self.fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform the input texts into feature vectors."""
        vecs = self.pipeline.transform(texts)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-9, None)

    def vocabulary_neighbours(self, term: str, k: int = 5) -> List[str]:
        """Terms closest to ``term`` in the LSA space (used for query expansion)."""
        tfidf: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        svd: TruncatedSVD = self.pipeline.named_steps["svd"]
        vocab = tfidf.vocabulary_
        if term not in vocab:
            return []
        term_vecs = svd.components_.T  # (n_terms, k)
        norms = np.linalg.norm(term_vecs, axis=1)
        q = term_vecs[vocab[term]]
        sims = term_vecs @ q / (np.clip(norms, 1e-9, None) * max(np.linalg.norm(q), 1e-9))
        inv = tfidf.get_feature_names_out()
        order = np.argsort(-sims)
        return [inv[i] for i in order if inv[i] != term and " " not in inv[i]][:k]


def describe_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Per-category summary of the style features (for EDA notebooks)."""
    feats = StyleFeatures().transform(df[text_col]).toarray()
    out = pd.DataFrame(feats, columns=StyleFeatures.names)
    out["category"] = df["category"].values
    return out.groupby("category").mean().round(4)
