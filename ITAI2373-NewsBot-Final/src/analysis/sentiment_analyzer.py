"""
Advanced sentiment analysis (Module A).

* **Sentence-averaged VADER** – the midterm showed whole-article VADER
  saturates near ±1 for 70% of articles, so scores are averaged per sentence;
* **Optional transformer** – multilingual XLM-RoBERTa sentiment model
  (``cardiffnlp/twitter-xlm-roberta-base-sentiment``) for non-English text;
* **NRC emotions** – eight Plutchik emotions from the NRC lexicon;
* **Entity/aspect sentiment** – sentiment of the sentences that mention an entity;
* **Temporal tracking + anomaly detection** – sentiment per period and
  category with z-score alerts for sudden shifts.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data_processing.text_preprocessor import TextPreprocessor
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

# News-domain corrections to VADER: words that are neutral in news reporting.
NEWS_LEXICON_FIXES = {
    "party": 0.0, "parties": 0.0, "cancer": -1.5, "penalty": 0.0, "penalties": 0.0,
    "goal": 0.5, "slam": 0.0, "grand": 0.0, "share": 0.0, "shares": 0.0, "interest": 0.0,
    "vote": 0.0, "votes": 0.0, "championship": 0.8, "crisis": -2.0, "recession": -2.2,
    "growth": 1.2, "profit": 1.3, "profits": 1.3, "loss": -1.3, "losses": -1.3,
    "war": -2.5, "killed": -3.0, "lead": 0.0, "leads": 0.0, "defeat": -1.6, "beat": 0.6,
}


def _load_nrc() -> Dict[str, List[str]]:
    """Load the NRC emotion lexicon bundled with the ``nrclex`` package."""
    try:
        import nrclex
        path = os.path.join(os.path.dirname(nrclex.__file__), "data", "nrc_en.json")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # tiny fallback so the module still works
        log.warning("nrclex not installed – using a tiny fallback emotion lexicon")
        return {"win": ["joy", "anticipation"], "attack": ["anger", "fear"], "trust": ["trust"],
                "death": ["sadness", "fear"], "celebrate": ["joy"], "scandal": ["disgust", "anger"],
                "hope": ["anticipation", "joy", "trust"], "crisis": ["fear", "sadness"]}


class SentimentAnalyzer:
    """
    Multi-dimensional sentiment analysis for news.

    Parameters
    ----------
    use_transformer : bool
        Try to load a multilingual transformer for non-English text. If the
        model cannot be downloaded the analyser falls back to VADER on the
        (optionally translated) text.
    """

    TRANSFORMER = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    def __init__(self, use_transformer: bool = False):
        """Create the object; parameters are described in the class docstring."""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.vader = SentimentIntensityAnalyzer()
        self.vader.lexicon.update(NEWS_LEXICON_FIXES)
        self.nrc = _load_nrc()
        self.pre = TextPreprocessor()
        self._pipe = None
        if use_transformer:
            try:
                from transformers import pipeline
                self._pipe = pipeline("sentiment-analysis", model=self.TRANSFORMER,
                                      truncation=True, max_length=512)
                log.info("Loaded transformer sentiment model %s", self.TRANSFORMER)
            except Exception as exc:
                log.warning("Transformer sentiment unavailable (%s) – using VADER", exc)

    # ------------------------------------------------------------ core API
    @staticmethod
    def label(score: float, threshold: float = 0.05) -> str:
        """Map a compound score to positive / neutral / negative."""
        if score >= threshold:
            return "positive"
        if score <= -threshold:
            return "negative"
        return "neutral"

    def sentence_scores(self, text: str) -> List[float]:
        """VADER compound score for every sentence of ``text``."""
        return [self.vader.polarity_scores(s)["compound"] for s in self.pre.split_sentences(text)]

    def analyze(self, text: str, language: str = "en") -> dict:
        """
        Full sentiment profile of one article.

        Returns overall label/score (sentence average), whole-text VADER,
        share of positive/negative sentences, a confidence value, the most
        positive and negative sentences and the NRC emotion distribution.
        """
        if language != "en" and self._pipe is not None:
            return self._analyze_transformer(text, language)
        sents = self.pre.split_sentences(text) or [text]
        scores = np.array([self.vader.polarity_scores(s)["compound"] for s in sents])
        avg = float(scores.mean()) if len(scores) else 0.0
        pos_share = float((scores >= 0.05).mean())
        neg_share = float((scores <= -0.05).mean())
        # Confidence = agreement of sentences with the overall direction.
        lab = self.label(avg)
        agree = {"positive": pos_share, "negative": neg_share,
                 "neutral": 1 - pos_share - neg_share}[lab]
        order = np.argsort(scores)
        return {
            "label": lab,
            "score": round(avg, 4),
            "whole_text_vader": round(self.vader.polarity_scores(text[:5000])["compound"], 4),
            "confidence": round(float(0.5 + 0.5 * agree), 3),
            "positive_sentence_share": round(pos_share, 3),
            "negative_sentence_share": round(neg_share, 3),
            "most_positive_sentence": sents[order[-1]] if len(sents) else "",
            "most_negative_sentence": sents[order[0]] if len(sents) else "",
            "emotions": self.emotions(text),
            "method": "vader-sentence-average",
        }

    def _analyze_transformer(self, text: str, language: str) -> dict:
        sents = self.pre.split_sentences(text)[:30] or [text[:1000]]
        outputs = self._pipe(sents)
        mapping = {"positive": 1, "negative": -1, "neutral": 0}
        vals = [mapping.get(o["label"].lower(), 0) * o["score"] for o in outputs]
        avg = float(np.mean(vals))
        return {"label": self.label(avg, 0.1), "score": round(avg, 4),
                "confidence": round(float(np.mean([o["score"] for o in outputs])), 3),
                "emotions": self.emotions(text) if language == "en" else {},
                "method": f"transformer:{self.TRANSFORMER}", "language": language}

    def emotions(self, text: str) -> Dict[str, float]:
        """Relative frequency of the eight NRC emotions (sums to 1, or all 0)."""
        tokens = re.findall(r"[a-z]+", text.lower())
        counts = Counter()
        for tok in tokens:
            for emo in self.nrc.get(tok, []):
                if emo in EMOTIONS:
                    counts[emo] += 1
        total = sum(counts.values())
        return {e: round(counts[e] / total, 4) if total else 0.0 for e in EMOTIONS}

    def batch_scores(self, texts: Iterable[str]) -> pd.DataFrame:
        """Fast batch scoring (sentence-average + emotions) for many articles."""
        rows = []
        for t in texts:
            sc = self.sentence_scores(t) or [0.0]
            row = {"sentiment": float(np.mean(sc)), "pos_share": float(np.mean(np.array(sc) >= 0.05)),
                   "neg_share": float(np.mean(np.array(sc) <= -0.05))}
            row.update(self.emotions(t))
            rows.append(row)
        df = pd.DataFrame(rows)
        df["sentiment_label"] = df["sentiment"].apply(self.label)
        return df

    # ------------------------------------------------ entity / aspect level
    def entity_sentiment(self, text: str, entities: Iterable[str]) -> Dict[str, dict]:
        """Average sentiment of the sentences that mention each entity."""
        sents = self.pre.split_sentences(text)
        out = {}
        for ent in set(entities):
            hits = [self.vader.polarity_scores(s)["compound"] for s in sents if ent.lower() in s.lower()]
            if hits:
                out[ent] = {"mentions": len(hits), "score": round(float(np.mean(hits)), 3),
                            "label": self.label(float(np.mean(hits)))}
        return out

    # ------------------------------------------------------------- temporal
    @staticmethod
    def track_over_time(df: pd.DataFrame, date_col: str = "published", score_col: str = "sentiment",
                        group_col: Optional[str] = "category", freq: str = "Q") -> pd.DataFrame:
        """Mean sentiment and article count per period (and per group)."""
        data = df.dropna(subset=[date_col]).copy()
        data["period"] = pd.to_datetime(data[date_col], utc=True).dt.tz_convert(None).dt.to_period(freq).astype(str)
        keys = ["period"] + ([group_col] if group_col else [])
        out = data.groupby(keys)[score_col].agg(["mean", "count", "std"]).reset_index()
        return out.rename(columns={"mean": "mean_sentiment", "count": "n_articles"})

    @staticmethod
    def detect_anomalies(timeline: pd.DataFrame, value_col: str = "mean_sentiment",
                         group_col: Optional[str] = "category", z: float = 1.5) -> pd.DataFrame:
        """
        Flag periods whose sentiment deviates more than ``z`` standard
        deviations from that group's own average – candidate "events".
        """
        frames = []
        groups = timeline.groupby(group_col) if group_col else [("all", timeline)]
        for _, g in groups:
            g = g.copy()
            mu, sd = g[value_col].mean(), g[value_col].std(ddof=0)
            g["z_score"] = (g[value_col] - mu) / (sd if sd > 0 else 1)
            g["anomaly"] = g["z_score"].abs() >= z
            frames.append(g)
        return pd.concat(frames, ignore_index=True)

    def evaluate_against_labels(self, texts: Iterable[str], gold: Iterable[str]) -> dict:
        """Compare predicted polarity with external labels (e.g. Webz sentiment)."""
        from sklearn.metrics import accuracy_score, f1_score
        gold = list(gold)
        scores = [float(np.mean(self.sentence_scores(t) or [0])) for t in texts]
        # binary evaluation: positive vs negative (neutral -> sign of score)
        pred = ["positive" if s >= 0 else "negative" for s in scores]
        mask = [g in ("positive", "negative") for g in gold]
        g2 = [g for g, m in zip(gold, mask) if m]
        p2 = [p for p, m in zip(pred, mask) if m]
        return {"n": len(g2), "accuracy": round(accuracy_score(g2, p2), 4),
                "macro_f1": round(f1_score(g2, p2, average="macro"), 4)}
