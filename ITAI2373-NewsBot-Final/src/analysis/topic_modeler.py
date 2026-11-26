"""
Topic modelling and content discovery (Module A, course Module 9).

* LDA (on term counts) and NMF (on TF-IDF) with the same interface;
* model selection over the number of topics using **NPMI coherence**
  (implemented here from document co-occurrence) and **topic diversity**;
* topic distribution for new articles, dominant-topic clustering;
* **topic evolution**: topic share per period (requires dated articles);
* interactive Plotly visualisations (intertopic distance map, topic trends).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import MDS

from src.data_processing.text_preprocessor import TextPreprocessor
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Scraped-web boilerplate that otherwise forms its own "topics".
WEB_STOP_WORDS = set("""read article articles ago navigation menu app apps click subscribe subscription
newsletter cookies cookie privacy policy terms photo photos image images video videos getty copyright
reserved rights share facebook twitter whatsapp email print comments comment login sign updated
minutes minute hours hour min sec published editor editorial advertisement ad ads skip content
main toggle search home page news follow like just don make want really going know think thing things
way got use used people time good new factuality ground conglomerate blindspot ownership kam fe dd
leave check need stream""".split())


class TopicModeler:
    """
    Parameters
    ----------
    n_topics : int
    method : ``"lda"`` or ``"nmf"``
    max_features : vocabulary size
    """

    def __init__(self, n_topics: int = 10, method: str = "lda", max_features: int = 8000,
                 random_state: int = 42, language: str = "en"):
        """Create the object; parameters are described in the class docstring."""
        if method not in ("lda", "nmf"):
            raise ValueError("method must be 'lda' or 'nmf'")
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        self.random_state = random_state
        self.pre = TextPreprocessor(language=language)
        self.vectorizer = None
        self.model = None
        self.doc_topic_: Optional[np.ndarray] = None
        self._binary = None  # binary doc-term matrix for coherence
        self.labels_: List[str] = []

    # ------------------------------------------------------------------ fit
    def _vectorizer(self):
        common = dict(max_features=self.max_features, min_df=5, max_df=0.5, ngram_range=(1, 2),
                      stop_words=sorted(WEB_STOP_WORDS),
                      token_pattern=r"(?u)\b[^\W\d_][^\W\d_]+\b")
        return CountVectorizer(**common) if self.method == "lda" else TfidfVectorizer(sublinear_tf=True, **common)

    def fit_transform(self, documents: Iterable[str], normalized: bool = False) -> np.ndarray:
        """Train the topic model; returns the document-topic matrix."""
        docs = list(documents) if normalized else self.pre.batch_normalize(documents)
        self.vectorizer = self._vectorizer()
        X = self.vectorizer.fit_transform(docs)
        if self.method == "lda":
            self.model = LatentDirichletAllocation(n_components=self.n_topics, learning_method="batch",
                                                   max_iter=25, random_state=self.random_state,
                                                   doc_topic_prior=0.1, topic_word_prior=0.01, n_jobs=1)
        else:
            self.model = NMF(n_components=self.n_topics, init="nndsvda", random_state=self.random_state,
                             max_iter=400, l1_ratio=0.1)
        W = self.model.fit_transform(X)
        W = W / np.clip(W.sum(axis=1, keepdims=True), 1e-12, None)
        self.doc_topic_ = W
        self._binary = (X > 0).astype(np.int8).tocsc()
        self.labels_ = [" / ".join(self.get_topic_words(k, 3)) for k in range(self.n_topics)]
        return W

    # ------------------------------------------------------------- inspect
    def get_topic_words(self, topic_id: int, n_words: int = 10) -> List[str]:
        """Top ``n_words`` terms of topic ``topic_id``."""
        names = self.vectorizer.get_feature_names_out()
        comp = self.model.components_[topic_id]
        return [names[i] for i in np.argsort(-comp)[:n_words]]

    def topics_table(self, n_words: int = 10) -> pd.DataFrame:
        """One row per topic: label, top words, prevalence and NPMI coherence."""
        prevalence = self.doc_topic_.mean(axis=0)
        coh = self.coherence_per_topic(n_words)
        return pd.DataFrame({
            "topic": range(self.n_topics), "label": self.labels_,
            "top_words": [", ".join(self.get_topic_words(k, n_words)) for k in range(self.n_topics)],
            "prevalence": np.round(prevalence, 4), "npmi": np.round(coh, 4),
        }).sort_values("prevalence", ascending=False).reset_index(drop=True)

    def get_article_topics(self, article_text: str, top_n: int = 3) -> List[dict]:
        """Topic distribution of an unseen article (top ``top_n`` topics)."""
        X = self.vectorizer.transform([self.pre.normalize(article_text)])
        if X.nnz == 0:
            return []
        w = self.model.transform(X)[0]
        w = w / max(w.sum(), 1e-12)
        return [{"topic": int(k), "label": self.labels_[k], "weight": round(float(w[k]), 4)}
                for k in np.argsort(-w)[:top_n]]

    # ------------------------------------------------------------ quality
    def coherence_per_topic(self, n_words: int = 10) -> np.ndarray:
        """
        NPMI coherence (Bouma 2009; Röder et al. 2015) of each topic's top
        words, using document co-occurrence in the training corpus.
        Range -1..1; higher = words that really appear together.
        """
        vocab = self.vectorizer.vocabulary_
        B = self._binary
        n_docs = B.shape[0]
        df = np.asarray(B.sum(axis=0)).ravel()
        scores = []
        for k in range(self.n_topics):
            idx = [vocab[w] for w in self.get_topic_words(k, n_words)]
            vals = []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    co = B[:, idx[i]].multiply(B[:, idx[j]]).sum()
                    if co == 0:
                        vals.append(-1.0)
                        continue
                    p_ij = co / n_docs
                    p_i, p_j = df[idx[i]] / n_docs, df[idx[j]] / n_docs
                    pmi = math.log(p_ij / (p_i * p_j))
                    vals.append(pmi / -math.log(p_ij) if p_ij < 1 else 1.0)
            scores.append(float(np.mean(vals)) if vals else 0.0)
        return np.array(scores)

    def coherence(self, n_words: int = 10) -> float:
        """Mean NPMI coherence over all topics."""
        return float(self.coherence_per_topic(n_words).mean())

    def diversity(self, n_words: int = 25) -> float:
        """Share of unique words among all topics' top words (Dieng et al. 2020)."""
        words = [w for k in range(self.n_topics) for w in self.get_topic_words(k, n_words)]
        return len(set(words)) / len(words)

    # ------------------------------------------------------- clustering
    def dominant_topics(self) -> np.ndarray:
        """Index of the strongest topic for every training document."""
        return self.doc_topic_.argmax(axis=1)

    def cluster_purity(self, labels: Iterable[str]) -> float:
        """How well dominant topics align with known categories (purity)."""
        tab = pd.crosstab(self.dominant_topics(), np.asarray(list(labels)))
        return float(tab.max(axis=1).sum() / tab.values.sum())

    # ------------------------------------------------------------ evolution
    def track_topic_trends(self, dates: Iterable, freq: str = "Q", groups: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """
        Average topic share per period (long format). With ``groups`` (e.g.
        language) the trend is computed per group as well.
        """
        dates = pd.to_datetime(pd.Series(list(dates)), utc=True, errors="coerce").dt.tz_convert(None)
        df = pd.DataFrame(self.doc_topic_, columns=self.labels_)
        df["period"] = dates.dt.to_period(freq).astype(str).values
        keys = ["period"]
        if groups is not None:
            df["group"] = list(groups)
            keys.append("group")
        long = df.groupby(keys).mean().reset_index().melt(id_vars=keys, var_name="topic", value_name="share")
        return long

    @staticmethod
    def emerging_topics(trends: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        """Topics whose share grew most between the first and last period."""
        pivot = trends.pivot_table(index="topic", columns="period", values="share")
        first, last = pivot.columns.min(), pivot.columns.max()
        change = (pivot[last] - pivot[first]).sort_values(ascending=False)
        return pd.DataFrame({"topic": change.index, "first_share": pivot.loc[change.index, first].round(4).values,
                             "last_share": pivot.loc[change.index, last].round(4).values,
                             "change": change.round(4).values}).head(top_n)

    # ------------------------------------------------------- visualisation
    def visualize_topics(self, title: str = "Intertopic distance map"):
        """Interactive bubble chart: topics placed by similarity, sized by prevalence."""
        import plotly.express as px
        comp = self.model.components_ / self.model.components_.sum(axis=1, keepdims=True)
        # Jensen–Shannon distance between topic-word distributions
        k = len(comp)
        dist = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                m = 0.5 * (comp[i] + comp[j])
                js = 0.5 * (np.sum(comp[i] * np.log((comp[i] + 1e-12) / (m + 1e-12))) +
                            np.sum(comp[j] * np.log((comp[j] + 1e-12) / (m + 1e-12))))
                dist[i, j] = dist[j, i] = math.sqrt(max(js, 0))
        xy = MDS(n_components=2, dissimilarity="precomputed", random_state=self.random_state,
                 normalized_stress="auto").fit_transform(dist)
        df = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "topic": self.labels_,
                           "prevalence": self.doc_topic_.mean(axis=0),
                           "words": [", ".join(self.get_topic_words(i, 8)) for i in range(k)]})
        fig = px.scatter(df, x="x", y="y", size="prevalence", text="topic", hover_data=["words"],
                         title=title, size_max=60)
        fig.update_traces(textposition="middle center", marker=dict(opacity=0.55))
        fig.update_layout(xaxis_title="", yaxis_title="", showlegend=False, height=600)
        return fig


def select_n_topics(documents: List[str], candidates: Iterable[int] = (6, 8, 10, 12, 15),
                    method: str = "nmf", random_state: int = 42) -> pd.DataFrame:
    """Fit one model per candidate k and report coherence and diversity."""
    pre = TextPreprocessor()
    docs = pre.batch_normalize(documents)
    rows = []
    for k in candidates:
        tm = TopicModeler(k, method, random_state=random_state)
        tm.fit_transform(docs, normalized=True)
        rows.append({"n_topics": k, "method": method, "npmi": round(tm.coherence(), 4),
                     "diversity": round(tm.diversity(), 4)})
        log.info("k=%d %s npmi=%.3f diversity=%.3f", k, method, rows[-1]["npmi"], rows[-1]["diversity"])
    out = pd.DataFrame(rows)
    out["score"] = (out["npmi"] - out["npmi"].min()) / (np.ptp(out["npmi"]) or 1) + \
                   (out["diversity"] - out["diversity"].min()) / (np.ptp(out["diversity"]) or 1)
    return out
