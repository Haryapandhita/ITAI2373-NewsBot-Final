"""
Cross-lingual analysis and cultural context (Module C).

Compares how news in different languages covers the same world:

* **coverage** – share of each category per language;
* **tone** – sentiment per language and category (Webz labels are produced
  by one multilingual model, so they are comparable across languages);
* **entity perspective** – which people/places each language talks about,
  overlap between languages, and tone towards a shared entity;
* **story linking** – the same event reported in two languages, matched by
  shared named entities + publication window (works offline) or by
  multilingual embeddings when available;
* **cultural context** – country focus (domestic vs foreign), currency and
  institution references that need explaining to an international reader.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.analysis.ner_extractor import normalize_entity
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

HOME_COUNTRIES = {"en": {"US", "GB", "AU", "CA", "IE", "NZ", "IN"}, "es": {"ES", "MX", "AR", "CO", "CL", "PE", "VE"},
                  "fr": {"FR", "BE", "CH", "CA"}, "de": {"DE", "AT", "CH"}, "it": {"IT", "CH"},
                  "pt": {"BR", "PT"}}

# Culture-specific references a foreign reader may need explained.
CULTURAL_REFERENCES = {
    "es": {"La Liga": "Spanish top football division", "Moncloa": "seat of the Spanish government",
           "Congreso de los Diputados": "Spanish lower house", "BCE": "European Central Bank (Spanish acronym)",
           "Ibex": "Spanish stock index (IBEX 35)", "PSOE": "Spanish Socialist party", "PP": "Spanish People's Party",
           "Sheinbaum": "President of Mexico", "AMLO": "former Mexican president López Obrador"},
    "fr": {"Élysée": "French presidency", "Matignon": "French prime minister's office",
           "Assemblée nationale": "French lower house", "CAC 40": "French stock index", "Ligue 1": "French football league",
           "RN": "Rassemblement National party"},
    "de": {"Bundestag": "German parliament", "Kanzleramt": "German chancellery", "DAX": "German stock index",
           "Bundesliga": "German football league", "CDU": "Christian Democratic Union", "SPD": "Social Democratic Party",
           "AfD": "Alternative für Deutschland party", "Ampel": "former three-party 'traffic light' coalition"},
    "it": {"Palazzo Chigi": "Italian prime minister's office", "Serie A": "Italian football league",
           "FTSE MIB": "Italian stock index", "Quirinale": "Italian presidency"},
    "pt": {"Planalto": "Brazilian presidency", "Brasileirão": "Brazilian football league",
           "Ibovespa": "Brazilian stock index", "Lula": "President of Brazil", "STF": "Brazilian Supreme Court"},
    "en": {"FTSE": "UK stock index", "Downing Street": "UK prime minister's office", "Fed": "US Federal Reserve",
           "Capitol Hill": "US Congress", "Premier League": "English football league", "Wall Street": "US financial markets"},
}
CURRENCIES = {"€": "EUR", "euro": "EUR", "euros": "EUR", "$": "USD", "dólares": "USD", "dollars": "USD",
              "£": "GBP", "libras": "GBP", "pounds": "GBP", "reais": "BRL", "R$": "BRL", "pesos": "MXN/ARS",
              "yuan": "CNY", "yen": "JPY", "rupees": "INR"}


class CrossLingualAnalyzer:
    """Comparative analysis of a multilingual, analysed article DataFrame."""

    def __init__(self, df: pd.DataFrame, entity_col: str = "entities"):
        """Create the object; parameters are described in the class docstring."""
        self.df = df.reset_index(drop=True)
        self.entity_col = entity_col

    # ------------------------------------------------------------ coverage
    def coverage_by_language(self) -> pd.DataFrame:
        """Category share within each language (rows sum to 1)."""
        return pd.crosstab(self.df["language"], self.df["category"], normalize="index").round(3)

    def sentiment_by_language(self, label_col: str = "webz_sentiment") -> pd.DataFrame:
        """Share of positive articles per language × category (+ overall)."""
        d = self.df.dropna(subset=[label_col]).copy()
        d["is_pos"] = (d[label_col] == "positive").astype(float)
        table = d.pivot_table(index="language", columns="category", values="is_pos", aggfunc="mean")
        table["overall"] = d.groupby("language")["is_pos"].mean()
        table["n"] = d.groupby("language").size()
        return table.round(3)

    # ------------------------------------------------------------ entities
    def entities_by_language(self, top_n: int = 15) -> Dict[str, List[tuple]]:
        """Most frequent entities (document frequency) per language."""
        out = {}
        for lang, g in self.df.groupby("language"):
            c = Counter(e for lst in g[self.entity_col] for e in set(lst))
            out[lang] = c.most_common(top_n)
        return out

    def entity_overlap(self, top_n: int = 50) -> pd.DataFrame:
        """Jaccard overlap of the top-N entities between every pair of languages."""
        tops = {l: {e for e, _ in v} for l, v in self.entities_by_language(top_n).items()}
        langs = sorted(tops)
        mat = pd.DataFrame(index=langs, columns=langs, dtype=float)
        for a, b in itertools.product(langs, langs):
            inter, union = len(tops[a] & tops[b]), len(tops[a] | tops[b]) or 1
            mat.loc[a, b] = round(inter / union, 3)
        return mat

    def perspective_on(self, entity: str, label_col: str = "webz_sentiment", min_articles: int = 5) -> pd.DataFrame:
        """How each language covers one entity: volume, positive share, top co-mentions."""
        target = normalize_entity(entity)
        rows = []
        for lang, g in self.df.groupby("language"):
            hits = g[g[self.entity_col].apply(lambda lst: target in lst)]
            if len(hits) < min_articles:
                continue
            co = Counter(e for lst in hits[self.entity_col] for e in set(lst) if e != target)
            rows.append({"language": lang, "articles": len(hits),
                         "share_of_language": round(len(hits) / len(g), 4),
                         "positive_share": round(float((hits[label_col] == "positive").mean()), 3),
                         "top_co_mentions": ", ".join(e for e, _ in co.most_common(5))})
        return pd.DataFrame(rows).sort_values("articles", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()

    # -------------------------------------------------------- story links
    def link_stories(self, min_shared: int = 3, window_days: int = 7, max_links: int = 30) -> pd.DataFrame:
        """
        Same story in different languages: articles that share at least
        ``min_shared`` named entities and were published within ``window_days``.
        """
        d = self.df.dropna(subset=["published"])
        index = defaultdict(list)
        for i, lst in zip(d.index, d[self.entity_col]):
            for e in set(lst):
                index[e].append(i)
        pair_count = Counter()
        for e, ids in index.items():
            if 2 <= len(ids) <= 200:  # skip ultra-common entities (e.g. United States)
                for a, b in itertools.combinations(ids, 2):
                    if self.df.at[a, "language"] != self.df.at[b, "language"]:
                        pair_count[(a, b)] += 1
        rows = []
        for (a, b), n in pair_count.most_common():
            if n < min_shared:
                break
            gap = abs((self.df.at[a, "published"] - self.df.at[b, "published"]).days)
            if gap > window_days:
                continue
            shared = sorted(set(self.df.at[a, self.entity_col]) & set(self.df.at[b, self.entity_col]))
            rows.append({"lang_a": self.df.at[a, "language"], "title_a": self.df.at[a, "title"][:90],
                         "lang_b": self.df.at[b, "language"], "title_b": self.df.at[b, "title"][:90],
                         "shared_entities": ", ".join(shared[:6]), "n_shared": n, "days_apart": gap})
            if len(rows) >= max_links:
                break
        return pd.DataFrame(rows)

    # ---------------------------------------------------- cultural context
    def country_focus(self) -> pd.DataFrame:
        """Share of each language's articles coming from its 'home' countries."""
        rows = []
        for lang, g in self.df.dropna(subset=["country"]).groupby("language"):
            home = HOME_COUNTRIES.get(lang, set())
            rows.append({"language": lang, "articles_with_country": len(g),
                         "home_country_share": round(float(g["country"].isin(home).mean()), 3),
                         "top_source_countries": ", ".join(g["country"].value_counts().head(4).index)})
        return pd.DataFrame(rows)

    @staticmethod
    def extract_cultural_context(text: str, source_language: str) -> dict:
        """Culture-specific references (institutions, leagues, indices) and currencies in a text."""
        refs = {}
        for lang_refs in (CULTURAL_REFERENCES.get(source_language, {}), CULTURAL_REFERENCES["en"]):
            for term, expl in lang_refs.items():
                if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text):
                    refs[term] = expl
        currencies = sorted({code for sym, code in CURRENCIES.items()
                             if re.search(r"(?<!\w)" + re.escape(sym) + r"(?!\w)" if sym.isalpha() else re.escape(sym),
                                          text, flags=re.IGNORECASE)})
        return {"language": source_language, "references": refs, "currencies": currencies}
