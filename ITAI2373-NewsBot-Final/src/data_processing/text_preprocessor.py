"""
Text preprocessing (enhanced from the midterm).

Midterm lessons kept here:
* keep TWO versions of every text – ``clean`` (casing/punctuation kept, for
  spaCy, VADER, summarisation) and ``normalized`` (for TF-IDF / topics);
* repair contractions, turn money amounts into tokens (``£168m`` ->
  ``currency_gbp``), keep sentiment-bearing words such as "won"/"not".

New in 2.0:
* language-aware stop-word lists for the six supported languages;
* optional spaCy lemmatisation (off by default because it is slow);
* boilerplate stripping for scraped web news (cookie banners, share links).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

CONTRACTIONS = {
    r"\bcan't\b": "can not", r"\bwon't\b": "will not", r"\bn't\b": " not",
    r"\b(\w+)'re\b": r"\1 are", r"\b(\w+)'ve\b": r"\1 have", r"\b(\w+)'ll\b": r"\1 will",
    r"\b(\w+)'d\b": r"\1 would", r"\b(i)'m\b": r"\1 am", r"\b(\w+)'s\b": r"\1",
}
CURRENCY = [
    (r"£\s?\d[\d,.]*\s?(bn|m|k|billion|million)?", " currency_gbp "),
    (r"\$\s?\d[\d,.]*\s?(bn|m|k|billion|million)?", " currency_usd "),
    (r"€\s?\d[\d,.]*\s?(bn|m|k|billion|million)?", " currency_eur "),
]
BOILERPLATE = re.compile(
    r"(cookie|cookies|subscribe|newsletter|sign up|all rights reserved|click here|"
    r"share this|follow us|advertisement|read more)\b.*", re.IGNORECASE)

# Words that carry meaning for news sentiment/classification – never removed.
KEEP_WORDS = {"not", "no", "won", "against", "up", "down", "off", "over", "under"}

# Compact stop-word lists for the non-English languages (most frequent function words).
STOP_WORDS = {
    "en": set(ENGLISH_STOP_WORDS) - KEEP_WORDS | {"said", "says", "mr", "mrs", "ms", "year", "years"},
    "es": set("de la que el en y a los se del las un por con no una su para es al lo como más "
              "o pero sus le ha me si sin sobre este ya entre cuando todo esta ser son dos también "
              "fue había era muy años hasta desde está mi porque qué sólo han yo hay vez puede".split()),
    "fr": set("de la le et les des en un du une que est pour qui dans a par plus pas au sur ne se "
              "ce il sont la avec son ses aux elle nous vous leur été être ont cette mais ou comme "
              "fait sa dont tout aussi même deux ans".split()),
    "de": set("der die und in den von zu das mit sich des auf für ist im dem nicht ein eine als "
              "auch es an werden aus er hat dass sie nach wird bei einer um am sind noch wie einem "
              "über einen so zum war haben nur oder aber vor zur bis mehr durch man".split()),
    "it": set("di e il la che in a per un del è non una con i le si da al dei della sono come "
              "più anche nel alla ha gli lo ma delle se ci questo su nella degli tra loro".split()),
    "pt": set("de a o que e do da em um para é com não uma os no se na por mais as dos como mas "
              "foi ao ele das tem à seu sua ou ser quando muito há nos já está também só pelo".split()),
}


class TextPreprocessor:
    """
    Configurable text cleaning pipeline.

    Parameters
    ----------
    language : str
        ISO code used to choose the stop-word list (default ``"en"``).
    lemmatize : bool
        If True, lemmatise English text with spaCy (slower, slightly better topics).
    min_token_len : int
        Tokens shorter than this are dropped in the normalised version.
    """

    def __init__(self, language: str = "en", lemmatize: bool = False, min_token_len: int = 2):
        """Create the object; parameters are described in the class docstring."""
        self.language = language
        self.lemmatize = lemmatize
        self.min_token_len = min_token_len
        self._nlp = None

    # ------------------------------------------------------------- cleaning
    @staticmethod
    def clean(text: str) -> str:
        """Light cleaning that keeps casing and punctuation (for spaCy/VADER)."""
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+\.\w+", " ", text)
        lines = [ln for ln in text.splitlines() if not BOILERPLATE.match(ln.strip())]
        text = " ".join(lines)
        text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
        return re.sub(r"\s+", " ", text).strip()

    def normalize(self, text: str, language: Optional[str] = None) -> str:
        """Aggressive normalisation for bag-of-words models (TF-IDF, LDA)."""
        lang = language or self.language
        text = self.clean(text).lower()
        for pattern, repl in CURRENCY:
            text = re.sub(pattern, repl, text)
        if lang == "en":
            for pattern, repl in CONTRACTIONS.items():
                text = re.sub(pattern, repl, text)
            # BBC-style stripped apostrophes: "don t" -> "do not"
            text = re.sub(r"\b(do|does|did|is|was|were|are|has|have|had|could|would|should)n t\b",
                          r"\1 not", text)
        text = re.sub(r"\d+([.,]\d+)?%", " percent_value ", text)
        text = re.sub(r"\b\d+([.,]\d+)?\b", " ", text)
        tokens = re.findall(r"[^\W\d_]+(?:_[^\W\d_]+)?", text, flags=re.UNICODE)
        stops = STOP_WORDS.get(lang, set())
        tokens = [t for t in tokens if len(t) >= self.min_token_len and t not in stops]
        if self.lemmatize and lang == "en":
            tokens = self._lemmatize(tokens)
        return " ".join(tokens)

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        return [t.lemma_ for t in self._nlp(" ".join(tokens))]

    def batch_normalize(self, texts: Iterable[str], languages: Optional[Iterable[str]] = None) -> List[str]:
        """Normalise many texts; ``languages`` can give a code per text."""
        texts = list(texts)
        langs = list(languages) if languages is not None else [self.language] * len(texts)
        return [self.normalize(t, lang) for t, lang in zip(texts, langs)]

    # ------------------------------------------------------------ sentences
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Rule-based sentence splitter (fast, language-agnostic, no downloads)."""
        out: List[str] = []
        # paragraphs / line breaks are sentence boundaries in scraped news
        for line in (text or "").splitlines():
            line = TextPreprocessor.clean(line)
            if not line:
                continue
            # protect common abbreviations
            line = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|St|Inc|Ltd|Co|vs|U\.S|U\.K|Sr|Jr)\.", r"\1<DOT>", line)
            parts = re.split(r"(?<=[.!?])\s+(?=[\"'¿¡A-ZÀ-ÖØ-Þ0-9])", line)
            out.extend(p.replace("<DOT>", ".").strip() for p in parts if len(p.strip()) > 1)
        return out
