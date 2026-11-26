"""
Content enhancement and insight generation (Module B).

Everything here is *grounded*: generated text is assembled from facts the
system computed (entity graph, search results, statistics). An optional
local LLM (Ollama) can rephrase the grounded draft into smoother prose, but
it never adds facts – if Ollama is not running the template text is used.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)


class OllamaClient:
    """Minimal client for a local Ollama server (optional)."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2", timeout: int = 60):
        """Create the object; parameters are described in the class docstring."""
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._available: Optional[bool] = None

    def available(self) -> bool:
        """True if an Ollama server answers at ``host`` (checked once)."""
        if self._available is None:
            try:
                with urllib.request.urlopen(self.host + "/api/tags", timeout=2) as r:
                    self._available = r.status == 200
            except Exception:
                self._available = False
        return self._available

    def rewrite(self, draft: str, instruction: str = "Rewrite as a friendly, concise answer.") -> str:
        """Ask the LLM to rephrase ``draft`` without adding facts; returns draft on failure."""
        if not self.available():
            return draft
        prompt = (f"{instruction} Use ONLY the facts below, do not add any new facts, numbers or names.\n\n"
                  f"FACTS:\n{draft}\n\nANSWER:")
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0.2}}).encode()
        try:
            req = urllib.request.Request(self.host + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())["response"].strip() or draft
        except Exception as exc:
            log.warning("Ollama call failed (%s) – using template text", exc)
            return draft


NUMBER_FACT = re.compile(
    r"(?P<value>[£$€]?\d[\d,.]*\s?(?:%|percent|bn|billion|m|million|k)?)\s+(?P<context>\w+(?:\s\w+){0,3})",
    re.IGNORECASE)


class ContentEnhancer:
    """
    Adds context to articles and produces corpus-level insights.

    Parameters
    ----------
    entity_mapper : EntityRelationshipMapper with a built knowledge graph
    search_engine : SemanticSearchEngine with indexed articles
    corpus_stats : DataFrame with per-article ``category`` and ``sentiment`` columns
    """

    def __init__(self, entity_mapper=None, search_engine=None, corpus_stats: Optional[pd.DataFrame] = None,
                 llm: Optional[OllamaClient] = None):
        """Create the object; parameters are described in the class docstring."""
        self.mapper = entity_mapper
        self.search = search_engine
        self.stats = corpus_stats
        self.llm = llm

    # ------------------------------------------------------------ enhance
    def enhance_article(self, text: str, entities: Optional[List[dict]] = None, category: Optional[str] = None,
                        sentiment: Optional[float] = None, k_related: int = 3) -> dict:
        """
        Context pack for one article: background on its key entities (from
        the knowledge graph), related coverage, how its tone compares with
        the category average, and a short "why it matters" line.
        """
        out: Dict[str, object] = {}
        if self.mapper is not None and entities:
            counts = Counter(e["canonical"] for e in entities if e["label"] in ("PERSON", "ORG", "GPE", "PER"))
            backgrounds = []
            for name, _ in counts.most_common(4):
                prof = self.mapper.entity_profile(name)
                if prof.get("found"):
                    top_cat = max(prof["categories"], key=prof["categories"].get) if prof["categories"] else "n/a"
                    links = ", ".join(n for n, _, _ in prof["top_connections"][:3])
                    backgrounds.append({"entity": prof["name"], "type": prof["type"],
                                        "mentions_in_corpus": prof["mentions"], "mostly_in": top_cat,
                                        "often_with": links})
            out["entity_background"] = backgrounds
        if self.search is not None:
            rel = self.search.find_similar_articles(text, top_k=k_related)
            out["related_articles"] = rel[["doc_id", "title", "similarity"]].to_dict("records")
        if self.stats is not None and category is not None and sentiment is not None:
            cat = self.stats[self.stats["category"] == category]["sentiment"]
            if len(cat):
                pct = float((cat < sentiment).mean() * 100)
                out["tone_context"] = (f"More positive than {pct:.0f}% of {category} articles "
                                       f"(category average {cat.mean():+.2f}, this article {sentiment:+.2f}).")
        out["numeric_facts"] = self.extract_numeric_facts(text)[:6]
        return out

    @staticmethod
    def extract_numeric_facts(text: str) -> List[dict]:
        """Numbers with a few words of context (amounts, percentages, scores)."""
        facts = []
        for m in NUMBER_FACT.finditer(text):
            val = m.group("value").strip()
            if re.fullmatch(r"\d{4}", val):  # skip years
                continue
            facts.append({"value": val, "context": m.group("context")})
        return facts

    def cross_reference_facts(self, text: str, k: int = 5) -> List[dict]:
        """
        Lightweight consistency check: find related articles and report
        numeric facts whose context words match but whose values differ –
        candidates for a human fact-checker (not an automatic verdict).
        """
        if self.search is None:
            return []
        mine = self.extract_numeric_facts(text)
        related = self.search.find_similar_articles(text, top_k=k)
        flags = []
        for _, row in related.iterrows():
            for other in self.extract_numeric_facts(row["text"]):
                for f in mine:
                    if f["context"].lower() == other["context"].lower() and f["value"] != other["value"]:
                        flags.append({"context": f["context"], "this_article": f["value"],
                                      "other_article": other["value"], "other_title": row["title"]})
        return flags[:10]

    # ------------------------------------------------------------ insights
    def generate_insights(self, df: pd.DataFrame, top_n: int = 5) -> List[str]:
        """
        Plain-English findings from an analysed article collection. Expects
        columns ``category`` and ``sentiment`` (optionally ``language``,
        ``published``, ``entities``).
        """
        ins = []
        counts = df["category"].value_counts(normalize=True)
        ins.append(f"{len(df):,} articles analysed; largest category is {counts.index[0]} "
                   f"({counts.iloc[0]:.0%} of coverage).")
        if "sentiment" in df:
            by_cat = df.groupby("category")["sentiment"].mean().sort_values()
            ins.append(f"Most negative coverage: {by_cat.index[0]} ({by_cat.iloc[0]:+.2f}); "
                       f"most positive: {by_cat.index[-1]} ({by_cat.iloc[-1]:+.2f}).")
        if "language" in df and df["language"].nunique() > 1 and "sentiment" in df:
            by_lang = df.groupby("language")["sentiment"].mean().sort_values()
            ins.append(f"Tone differs by language: {by_lang.index[0]} sources are the most negative "
                       f"({by_lang.iloc[0]:+.2f}), {by_lang.index[-1]} the most positive ({by_lang.iloc[-1]:+.2f}).")
        if "published" in df and "sentiment" in df:
            per = df.dropna(subset=["published"]).copy()
            per["q"] = pd.to_datetime(per["published"], utc=True).dt.tz_convert(None).dt.to_period("Q").astype(str)
            trend = per.groupby("q")["sentiment"].mean()
            if len(trend) >= 2:
                direction = "improved" if trend.iloc[-1] > trend.iloc[0] else "worsened"
                ins.append(f"Overall tone {direction} from {trend.index[0]} ({trend.iloc[0]:+.2f}) "
                           f"to {trend.index[-1]} ({trend.iloc[-1]:+.2f}).")
        if "entities" in df:
            ent = Counter(e for lst in df["entities"] for e in lst)
            top = ", ".join(f"{e} ({c})" for e, c in ent.most_common(top_n))
            ins.append(f"Most mentioned entities: {top}.")
        return ins

    def detect_information_gaps(self, df: pd.DataFrame, topic_mask: Iterable[bool],
                                group_col: str = "language") -> dict:
        """
        For one topic, which groups (languages/sources) under-cover it
        compared with their overall share – possible blind spots.
        """
        mask = np.asarray(list(topic_mask))
        overall = df[group_col].value_counts(normalize=True)
        topic = df.loc[mask, group_col].value_counts(normalize=True).reindex(overall.index).fillna(0)
        ratio = (topic / overall).round(2).sort_values()
        return {"coverage_ratio": ratio.to_dict(),
                "under_covered": [g for g, r in ratio.items() if r < 0.6],
                "over_covered": [g for g, r in ratio.items() if r > 1.5]}

    def polish(self, draft: str) -> str:
        """Optionally rephrase a grounded draft with the local LLM."""
        return self.llm.rewrite(draft) if self.llm else draft
