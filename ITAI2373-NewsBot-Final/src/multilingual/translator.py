"""
Translation integration (Module C).

Primary engine: MarianMT models from Helsinki-NLP (open source, run locally
in Colab, no API key). Translation quality is estimated **without a
reference** by round-trip translation (source -> English -> source) scored
with chrF, plus simple sanity checks (length ratio, untranslated tokens).

If ``transformers`` / model downloads are unavailable, the translator
degrades gracefully: it optionally tries ``deep-translator`` and otherwise
returns the original text with ``translated=False`` so downstream modules
can fall back to language-agnostic features.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List, Optional

from src.data_processing.text_preprocessor import TextPreprocessor
from src.utils.logging_utils import get_logger

log = get_logger(__name__)

# Marian model per direction. Portuguese has no dedicated pt->en model, the
# multi-source Romance model covers it.
MODEL_NAMES = {
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en", ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en", ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("pt", "en"): "Helsinki-NLP/opus-mt-ROMANCE-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es", ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de", ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-ROMANCE",
}
TARGET_PREFIX = {("en", "pt"): ">>pt<< "}  # multi-target model needs a language token


class Translator:
    """
    Parameters
    ----------
    use_transformer : bool
        Load MarianMT models on demand (one per language pair, cached).
    max_sentences : int
        Long articles are translated sentence by sentence up to this limit.
    """

    def __init__(self, use_transformer: bool = True, max_sentences: int = 40, device: Optional[int] = None):
        """Create the object; parameters are described in the class docstring."""
        self.use_transformer = use_transformer
        self.max_sentences = max_sentences
        self.device = device
        self._pipes: Dict[tuple, object] = {}
        self._failed: set = set()

    def _pipeline(self, src: str, tgt: str):
        key = (src, tgt)
        if key in self._pipes:
            return self._pipes[key]
        if not self.use_transformer or key in self._failed or key not in MODEL_NAMES:
            return None
        try:
            from transformers import pipeline
            kwargs = {"device": self.device} if self.device is not None else {}
            self._pipes[key] = pipeline("translation", model=MODEL_NAMES[key], **kwargs)
            log.info("Loaded translation model %s", MODEL_NAMES[key])
            return self._pipes[key]
        except Exception as exc:
            log.warning("Translation model %s unavailable (%s)", MODEL_NAMES[key], exc)
            self._failed.add(key)
            return None

    @staticmethod
    def _deep_translate(text: str, src: str, tgt: str) -> Optional[str]:
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source=src, target=tgt).translate(text[:4500])
        except Exception:
            return None

    def translate_text(self, text: str, source_language: str, target_language: str = "en") -> dict:
        """
        Translate ``text``; returns ``{"text", "translated", "engine"}``.
        Same-language input is returned unchanged.
        """
        if not text or source_language == target_language:
            return {"text": text, "translated": False, "engine": "identity"}
        pipe = self._pipeline(source_language, target_language)
        if pipe is not None:
            sents = TextPreprocessor.split_sentences(text)[: self.max_sentences] or [text]
            prefix = TARGET_PREFIX.get((source_language, target_language), "")
            outs = pipe([prefix + s[:600] for s in sents], max_length=512, truncation=True)
            return {"text": " ".join(o["translation_text"] for o in outs), "translated": True,
                    "engine": MODEL_NAMES[(source_language, target_language)]}
        alt = self._deep_translate(text, source_language, target_language)
        if alt:
            return {"text": alt, "translated": True, "engine": "deep-translator/google"}
        return {"text": text, "translated": False, "engine": "unavailable"}

    def translate_batch(self, texts: List[str], languages: List[str], target_language: str = "en") -> List[dict]:
        """Translate many texts, one language code per text."""
        return [self.translate_text(t, l, target_language) for t, l in zip(texts, languages)]

    # ------------------------------------------------------------ quality
    def assess_quality(self, source: str, source_language: str, translation: Optional[str] = None) -> dict:
        """
        Reference-free quality estimate. ``round_trip_chrf`` compares the
        source with its back-translation (0–100, higher is better).
        """
        if translation is None:
            translation = self.translate_text(source, source_language)["text"]
        out = {"length_ratio": round(len(translation.split()) / max(len(source.split()), 1), 3)}
        src_tokens = set(re.findall(r"[^\W\d_]{4,}", source.lower()))
        tr_tokens = re.findall(r"[^\W\d_]{4,}", translation.lower())
        out["copied_token_share"] = round(sum(t in src_tokens for t in tr_tokens) / max(len(tr_tokens), 1), 3)
        back = self.translate_text(translation, "en", source_language)
        if back["translated"]:
            try:
                import sacrebleu
                out["round_trip_chrf"] = round(sacrebleu.sentence_chrf(back["text"], [source]).score, 2)
            except ImportError:
                out["round_trip_chrf"] = None
            out["back_translation"] = back["text"]
        return out

    @staticmethod
    def corpus_scores(hypotheses: List[str], references: List[str]) -> dict:
        """BLEU and chrF for a set of translations against references."""
        import sacrebleu
        return {"bleu": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2),
                "chrf": round(sacrebleu.corpus_chrf(hypotheses, [references]).score, 2)}
