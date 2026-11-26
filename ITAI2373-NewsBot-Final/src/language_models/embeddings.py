"""
Semantic embeddings and semantic search (Module B, course Module 10).

``SemanticEmbedder`` uses a multilingual Sentence-Transformer
(``paraphrase-multilingual-MiniLM-L12-v2``) when available, so an English
query also finds Spanish or German articles (cross-lingual retrieval). When
the model cannot be loaded (offline machine) it falls back to LSA vectors
(TF-IDF + SVD) and logs the switch – the rest of the system is unchanged.

``SemanticSearchEngine`` combines dense similarity with sparse TF-IDF
matching (hybrid search), supports metadata filters, query expansion,
"more like this" and clustering of results into story groups.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data_processing.feature_extractor import LSAEmbedder
from src.data_processing.text_preprocessor import TextPreprocessor
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Small hand-made expansion map for news vocabulary (used with LSA neighbours).
SYNONYMS = {
    "ai": ["artificial intelligence", "machine learning", "chatbot"],
    "economy": ["economic", "gdp", "growth", "inflation"],
    "climate": ["emissions", "global warming", "carbon"],
    "election": ["vote", "poll", "campaign", "ballot"],
    "football": ["soccer", "premier league", "goal"],
    "stocks": ["shares", "market", "investors"],
    "regulation": ["rules", "law", "regulator", "antitrust"],
    "phone": ["smartphone", "mobile", "iphone"],
    "war": ["conflict", "military", "troops"],
}


class SemanticEmbedder:
    """Text -> L2-normalised vector, transformer or LSA backend."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 use_transformer: bool = True, lsa_components: int = 256):
        """Create the object; parameters are described in the class docstring."""
        self.model_name = model_name
        self.backend = "lsa"
        self._st = None
        self._lsa = LSAEmbedder(lsa_components)
        if use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(model_name)
                self.backend = "sentence-transformers"
                log.info("Loaded embedding model %s", model_name)
            except Exception as exc:
                log.warning("Sentence-Transformer unavailable (%s) – falling back to LSA", exc)

    @property
    def multilingual(self) -> bool:
        """True when the active backend embeds all languages in one space."""
        return self.backend == "sentence-transformers" and "multilingual" in self.model_name

    def fit(self, texts: List[str]) -> "SemanticEmbedder":
        """Only needed for the LSA backend (learns the vocabulary)."""
        if self.backend == "lsa":
            self._lsa.fit(TextPreprocessor().batch_normalize(texts))
        return self

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Embed texts as L2-normalised vectors (transformer or LSA backend)."""
        texts = [t[:2000] for t in texts]
        if self.backend == "sentence-transformers":
            return np.asarray(self._st.encode(texts, batch_size=batch_size, show_progress_bar=False,
                                              normalize_embeddings=True))
        if not self._lsa.fitted:
            raise RuntimeError("LSA backend must be fitted first: call fit(texts)")
        return self._lsa.transform(TextPreprocessor().batch_normalize(texts))

    def neighbours(self, term: str, k: int = 5) -> List[str]:
        """Vocabulary terms closest to ``term`` (LSA space) for query expansion."""
        return self._lsa.vocabulary_neighbours(term, k) if self._lsa.fitted else []


class SemanticSearchEngine:
    """
    Hybrid semantic search over an article DataFrame.

    The DataFrame must have ``doc_id, title, text``; optional columns used as
    filters: ``category, language, sentiment_label, published, month``.
    """

    def __init__(self, embedder: Optional[SemanticEmbedder] = None, alpha: float = 0.6):
        """Create the object; parameters are described in the class docstring."""
        self.embedder = embedder or SemanticEmbedder(use_transformer=False)
        self.alpha = alpha  # weight of dense vs sparse score
        self.df: Optional[pd.DataFrame] = None
        self.vectors: Optional[np.ndarray] = None
        self.sparse = None
        self.tfidf = None
        self.pre = TextPreprocessor()

    def encode_documents(self, df: pd.DataFrame) -> "SemanticSearchEngine":
        """Index the articles (dense + sparse)."""
        self.df = df.reset_index(drop=True).copy()
        docs = (self.df["title"].fillna("") + ". " + self.df["text"].fillna("")).tolist()
        if self.embedder.backend == "lsa" and not self.embedder._lsa.fitted:
            self.embedder.fit(docs)
        self.vectors = self.embedder.encode(docs)
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 2), max_features=100000)
        self.sparse = self.tfidf.fit_transform(self.pre.batch_normalize(docs, self.df.get("language")))
        log.info("Indexed %d articles with %s backend", len(self.df), self.embedder.backend)
        return self

    # ---------------------------------------------------------- expansion
    def expand_query(self, query: str, k: int = 2) -> Dict[str, List[str]]:
        """Related terms for each query word (synonym map + LSA neighbours)."""
        out = {}
        for tok in self.pre.normalize(query).split():
            related = list(SYNONYMS.get(tok, []))
            related += [n for n in self.embedder.neighbours(tok, k + 2) if n not in related][:k]
            if related:
                out[tok] = related[: k + 2]
        return out

    # -------------------------------------------------------------- search
    def _filter_mask(self, filters: Optional[dict]) -> np.ndarray:
        mask = np.ones(len(self.df), dtype=bool)
        if not filters:
            return mask
        for key, value in filters.items():
            if value in (None, "", []):
                continue
            if key == "date_from":
                mask &= pd.to_datetime(self.df["published"], utc=True) >= pd.Timestamp(value, tz="UTC")
            elif key == "date_to":
                mask &= pd.to_datetime(self.df["published"], utc=True) <= pd.Timestamp(value, tz="UTC")
            elif key in self.df.columns:
                vals = value if isinstance(value, (list, tuple, set)) else [value]
                mask &= self.df[key].isin(vals).values
        return mask

    def semantic_search(self, query: str, top_k: int = 5, filters: Optional[dict] = None,
                        expand: bool = True) -> pd.DataFrame:
        """
        Natural-language search ("technology companies facing regulation").
        Returns the top articles with dense, sparse and combined scores.
        """
        if self.df is None:
            raise RuntimeError("Call encode_documents(df) first")
        q_vec = self.embedder.encode([query])[0]
        dense = self.vectors @ q_vec
        q_text = query
        if expand:
            extra = [t for terms in self.expand_query(query).values() for t in terms]
            q_text = query + " " + " ".join(extra)
        q_sparse = self.tfidf.transform([self.pre.normalize(q_text)])
        sparse_scores = (self.sparse @ q_sparse.T).toarray().ravel()
        score = self.alpha * dense + (1 - self.alpha) * sparse_scores
        mask = self._filter_mask(filters)
        score = np.where(mask, score, -np.inf)
        idx = [i for i in np.argsort(-score)[:top_k] if np.isfinite(score[i])]
        res = self.df.iloc[idx].copy()
        res["dense_score"] = dense[idx].round(4)
        res["sparse_score"] = sparse_scores[idx].round(4)
        res["score"] = score[idx].round(4)
        return res

    def find_similar_articles(self, query_article: str, top_k: int = 5,
                              exclude_doc_id: Optional[str] = None, filters: Optional[dict] = None) -> pd.DataFrame:
        """"More like this": nearest neighbours of an article in embedding space."""
        vec = self.embedder.encode([query_article])[0]
        sims = self.vectors @ vec
        mask = self._filter_mask(filters)
        if exclude_doc_id is not None:
            mask &= (self.df["doc_id"] != exclude_doc_id).values
        sims = np.where(mask, sims, -np.inf)
        idx = [i for i in np.argsort(-sims)[: top_k + 1] if np.isfinite(sims[i]) and sims[i] < 0.999][:top_k]
        res = self.df.iloc[idx].copy()
        res["similarity"] = sims[idx].round(4)
        return res

    def cluster_similar_content(self, n_clusters: int = 8, subset: Optional[np.ndarray] = None,
                                random_state: int = 42) -> pd.DataFrame:
        """K-means story clusters with their most central article and top terms."""
        idx = np.arange(len(self.df)) if subset is None else np.asarray(subset)
        X = self.vectors[idx]
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(X)
        names = self.tfidf.get_feature_names_out()
        rows = []
        for c in range(n_clusters):
            members = idx[km.labels_ == c]
            centre_sim = self.vectors[members] @ km.cluster_centers_[c]
            central = members[int(np.argmax(centre_sim))]
            terms = np.asarray(self.sparse[members].mean(axis=0)).ravel()
            rows.append({"cluster": c, "size": len(members),
                         "top_terms": ", ".join(names[i] for i in np.argsort(-terms)[:6]),
                         "central_title": self.df.loc[central, "title"]})
        self.df.loc[idx, "cluster"] = km.labels_
        return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)

    def detect_near_duplicates(self, threshold: float = 0.95, max_pairs: int = 50) -> List[tuple]:
        """Pairs of articles that are (almost) the same story – syndication detection."""
        sims = self.vectors @ self.vectors.T
        np.fill_diagonal(sims, 0)
        i, j = np.where(np.triu(sims) >= threshold)
        pairs = sorted(zip(i, j, sims[i, j]), key=lambda x: -x[2])[:max_pairs]
        return [(self.df.loc[a, "doc_id"], self.df.loc[b, "doc_id"], round(float(s), 3)) for a, b, s in pairs]


def retrieval_eval(engine: SemanticSearchEngine, label_col: str = "category", k: int = 10,
                   n_queries: int = 200, random_state: int = 42) -> dict:
    """
    Precision@k using articles as queries: a retrieved article counts as
    relevant when it shares the query article's category.
    """
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(engine.df), size=min(n_queries, len(engine.df)), replace=False)
    precisions = []
    for i in idx:
        sims = engine.vectors @ engine.vectors[i]
        sims[i] = -np.inf
        top = np.argsort(-sims)[:k]
        precisions.append(float((engine.df.loc[top, label_col] == engine.df.loc[i, label_col]).mean()))
    return {"k": k, "queries": len(idx), "precision_at_k": round(float(np.mean(precisions)), 4),
            "backend": engine.embedder.backend}
