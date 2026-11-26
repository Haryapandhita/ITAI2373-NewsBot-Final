"""
Intelligent summarisation (Module B).

Three strategies behind one interface:

* ``lead``        – first sentences (strong news baseline);
* ``textrank``    – extractive, implemented here: sentence graph with
                    TF-IDF cosine edges + PageRank, position prior and MMR
                    redundancy removal;
* ``abstractive`` – DistilBART fine-tuned on CNN/DailyMail (optional, needs
                    ``transformers``; falls back to TextRank automatically).

Also: multi-document summaries, headline generation and quality metrics
(ROUGE against a reference, compression, entity coverage, redundancy).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_processing.text_preprocessor import TextPreprocessor
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

LENGTHS = {"brief": 2, "balanced": 3, "detailed": 5}


class IntelligentSummarizer:
    """
    Parameters
    ----------
    use_transformer : bool
        Load DistilBART for abstractive summaries/headlines when possible.
    model_name : str
        HuggingFace summarisation model.
    """

    def __init__(self, use_transformer: bool = False, model_name: str = "sshleifer/distilbart-cnn-12-6"):
        """Create the object; parameters are described in the class docstring."""
        self.pre = TextPreprocessor()
        self.model_name = model_name
        self._pipe = None  # Seq2SeqModel when the abstractive model is loaded
        if use_transformer:
            try:
                from src.language_models.seq2seq import Seq2SeqModel
                self._pipe = Seq2SeqModel(model_name)
                log.info("Loaded abstractive model %s on %s", model_name, self._pipe.device)
            except Exception as exc:
                log.warning("Abstractive model unavailable (%s) – using TextRank", exc)

    @property
    def abstractive_available(self) -> bool:
        """True when the DistilBART pipeline was loaded."""
        return self._pipe is not None

    # --------------------------------------------------------- extractive
    def lead(self, text: str, n: int = 3) -> str:
        """First ``n`` sentences – the classic news baseline."""
        return " ".join(self.pre.split_sentences(text)[:n])

    def textrank(self, text: str, n: int = 3, position_weight: float = 0.15, mmr_lambda: float = 0.7) -> str:
        """Graph-based extractive summary; sentences kept in original order."""
        sents = self.pre.split_sentences(text)
        if len(sents) <= n:
            return " ".join(sents)
        norm = [self.pre.normalize(s) or s.lower() for s in sents]
        try:
            X = TfidfVectorizer().fit_transform(norm)
        except ValueError:  # all stop words
            return self.lead(text, n)
        sim = cosine_similarity(X)
        np.fill_diagonal(sim, 0)
        graph = nx.from_numpy_array(sim)
        pr = nx.pagerank(graph, max_iter=200, tol=1e-6) if graph.number_of_edges() else {i: 1 for i in range(len(sents))}
        rank = np.array([pr[i] for i in range(len(sents))])
        rank = rank / max(rank.max(), 1e-12)
        position = np.array([1.0 / (1 + i) for i in range(len(sents))])  # news: early = important
        score = (1 - position_weight) * rank + position_weight * position
        # Maximal Marginal Relevance to avoid near-duplicate sentences
        chosen: List[int] = []
        candidates = list(range(len(sents)))
        while candidates and len(chosen) < n:
            def mmr(i):
                """Maximal-marginal-relevance score of candidate sentence ``i``."""
                redundancy = max((sim[i, j] for j in chosen), default=0.0)
                return mmr_lambda * score[i] - (1 - mmr_lambda) * redundancy
            best = max(candidates, key=mmr)
            chosen.append(best)
            candidates.remove(best)
        return " ".join(sents[i] for i in sorted(chosen))

    # ------------------------------------------------------- abstractive
    def abstractive(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        """Abstractive summary with DistilBART (falls back to TextRank)."""
        if self._pipe is None:
            return self.textrank(text)
        chunk = " ".join(text.split()[:700])  # model limit ~1024 tokens
        return self._pipe.generate([chunk], max_new_tokens=max_length,
                                   min_new_tokens=min(min_length, max_length - 5))[0]

    # ---------------------------------------------------------- public API
    def summarize_article(self, text: str, summary_type: str = "balanced", method: str = "auto") -> dict:
        """
        Summarise one article.

        ``summary_type``: brief / balanced / detailed. ``method``: auto /
        lead / textrank / abstractive ("auto" = abstractive if available).
        """
        if not text or len(text.split()) < 15:
            return {"summary": text.strip(), "method": "passthrough", "compression": 1.0}
        n = LENGTHS.get(summary_type, 3)
        if method == "auto":
            method = "abstractive" if self._pipe else "textrank"
        if method == "lead":
            summary = self.lead(text, n)
        elif method == "abstractive":
            summary = self.abstractive(text, max_length=40 * n + 20, min_length=15 * n)
            method = "abstractive" if self._pipe else "textrank"
        else:
            summary = self.textrank(text, n)
            method = "textrank"
        return {"summary": summary, "method": method,
                "compression": round(len(summary.split()) / max(len(text.split()), 1), 3)}

    def summarize_multiple_articles(self, texts: List[str], n_sentences: int = 5,
                                    focus_topic: Optional[str] = None) -> str:
        """
        Multi-document summary: pool the lead sentences of every article,
        optionally re-weight them by similarity to ``focus_topic``, then run
        TextRank+MMR across the pool so repeated facts appear only once.
        """
        pool = []
        for t in texts:
            pool.extend(self.pre.split_sentences(t)[:6])
        if not pool:
            return ""
        if focus_topic:
            vec = TfidfVectorizer().fit(pool + [focus_topic])
            rel = cosine_similarity(vec.transform(pool), vec.transform([focus_topic])).ravel()
            keep = np.argsort(-rel)[: max(25, n_sentences * 4)]
            pool = [pool[i] for i in sorted(keep)]
        return self.textrank(" ".join(pool), n=n_sentences, position_weight=0.0, mmr_lambda=0.6)

    def generate_headlines(self, text: str) -> Dict[str, str]:
        """
        Headline variants: informative (abstractive or compressed lead),
        keyword (top TF-IDF noun phrases) and social (short + hook).
        """
        sents = self.pre.split_sentences(text)
        first = sents[0] if sents else text[:120]
        if self._pipe is not None:
            informative = self.abstractive(text, max_length=24, min_length=6)
            informative = informative.split(".")[0]
        else:
            informative = re.sub(r"\s*(,|\bwhich\b|\bwho\b|\bafter\b|\bas\b|\bwhile\b)\s.*$", "", first)
            informative = " ".join(informative.split()[:14])
        tokens = [t for t in self.pre.normalize(text).split() if len(t) > 3]
        top = [w for w, _ in sorted({t: tokens.count(t) for t in set(tokens)}.items(), key=lambda kv: -kv[1])[:4]]
        keyword = " | ".join(w.capitalize() for w in top)
        social = f"{informative.rstrip('.')} – here's what you need to know"
        return {"informative": informative.strip(), "keyword": keyword, "social": social}

    # ------------------------------------------------------------ quality
    @staticmethod
    def assess_summary_quality(original: str, summary: str, reference: Optional[str] = None,
                               entities: Optional[List[str]] = None) -> dict:
        """
        Reference-based (ROUGE-1/2/L F1 when ``reference`` is given) and
        reference-free metrics: compression, entity coverage, redundancy
        (share of repeated bigrams) and novel-word ratio (abstractiveness).
        """
        out = {"compression": round(len(summary.split()) / max(len(original.split()), 1), 3)}
        if reference:
            try:
                from rouge_score import rouge_scorer
                sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                r = sc.score(reference, summary)
                out.update({k: round(v.fmeasure, 4) for k, v in r.items()})
            except ImportError:
                out["rouge"] = "rouge-score not installed"
        words = re.findall(r"\w+", summary.lower())
        bigrams = list(zip(words, words[1:]))
        out["redundancy"] = round(1 - len(set(bigrams)) / max(len(bigrams), 1), 3)
        orig_words = set(re.findall(r"\w+", original.lower()))
        out["novel_word_ratio"] = round(sum(w not in orig_words for w in words) / max(len(words), 1), 3)
        if entities:
            ents = set(entities)
            out["entity_coverage"] = round(sum(e.lower() in summary.lower() for e in ents) / len(ents), 3)
        return out
