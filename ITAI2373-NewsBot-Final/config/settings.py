"""
Central configuration for NewsBot 2.0.

All paths, model names, thresholds and data sources live here so that the
rest of the code never hard-codes a setting. Values can be overridden with
environment variables (see ``config/api_keys_template.txt``) or by passing a
modified ``NewsBotConfig`` object to ``NewsBot2``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Project root = folder that contains config/, src/, data/ ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass
class NewsBotConfig:
    """All tunable settings of the system in one object."""

    # ------------------------------------------------------------------ paths
    root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    models_dir: Path = PROJECT_ROOT / "data" / "models"
    results_dir: Path = PROJECT_ROOT / "data" / "results"
    figures_dir: Path = PROJECT_ROOT / "data" / "results" / "figures"

    # ------------------------------------------------------------ data sources
    # BBC News (Greene & Cunningham, 2006) with ORIGINAL casing and titles.
    # The midterm used the lower-cased Kaggle copy; this fixes the NER problem.
    bbc_git_url: str = "https://github.com/jonathanporta/bbc-news-article-archive.git"
    bbc_zip_url: str = (
        "https://codeload.github.com/jonathanporta/bbc-news-article-archive/zip/refs/heads/master"
    )
    # Webz.io free news datasets: multilingual, dated, labelled news (2023-2026).
    webz_raw_base: str = (
        "https://raw.githubusercontent.com/Webhose/free-news-datasets/master/News_Datasets/"
    )
    webz_snapshots: List[str] = field(default_factory=lambda: [
        "Economy, Business and Finance_positive_20240128131642.zip",
        "Economy, Business and Finance_negative_20250622073519.zip",
        "Economy, Business and Finance_negative_20260118073515.zip",
        "Economy, Business and Finance_positive_20260927071815.zip",
        "Economy, Business and Finance_positive_20261116065641.zip",
        "Politics_negative_20240128132206.zip",
        "Politics_negative_20240728070020.zip",
        "Politics_positive_20240811070033.zip",
        "Politics_negative_20241124070053.zip",
        "Politics_positive_20260920071818.zip",
        "Science and Technology_positive_20240128132252.zip",
        "Science and Technology_negative_20241215070200.zip",
        "Science and Technology_positive_20250504073516.zip",
        "Science and Technology_negative_20250824073515.zip",
        "Science and Technology_positive_20260816074915.zip",
        "Sport_negative_20240128131809.zip",
        "Sport_positive_20250216070035.zip",
        "Sport_positive_20251130073516.zip",
        "Sport_negative_20260412073515.zip",
        "Sport_negative_20261102065645.zip",
    ])
    # Webz category prefix -> BBC label space
    webz_category_map: Dict[str, str] = field(default_factory=lambda: {
        "Economy, Business and Finance": "business",
        "Politics": "politics",
        "Science and Technology": "tech",
        "Sport": "sport",
    })
    # Languages kept for the multilingual module (ISO 639-1 -> Webz name)
    languages: Dict[str, str] = field(default_factory=lambda: {
        "en": "english", "es": "spanish", "fr": "french",
        "de": "german", "it": "italian", "pt": "portuguese",
    })
    categories: List[str] = field(default_factory=lambda: [
        "business", "entertainment", "politics", "sport", "tech"])
    min_words: int = 40          # drop stubs shorter than this
    max_chars: int = 12000       # truncate very long scraped pages
    random_state: int = 42
    test_size: float = 0.2

    # ------------------------------------------------------------ NLP models
    spacy_model: str = "en_core_web_sm"
    spacy_multilingual_model: str = "xx_ent_wiki_sm"
    embedding_model: str = _env(
        "NEWSBOT_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    summarization_model: str = _env("NEWSBOT_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
    translation_model_template: str = "Helsinki-NLP/opus-mt-{src}-{tgt}"
    # Transformer components are optional; set NEWSBOT_USE_TRANSFORMERS=0 to
    # force the lightweight classical fallbacks (useful on slow machines).
    use_transformers: bool = _env("NEWSBOT_USE_TRANSFORMERS", "1") == "1"

    # Optional local LLM (Ollama) used ONLY to polish chatbot wording.
    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = _env("OLLAMA_MODEL", "llama3.2")
    use_llm: bool = _env("NEWSBOT_USE_LLM", "0") == "1"

    # ------------------------------------------------------------ thresholds
    low_confidence: float = 0.55        # flag classification for human review
    secondary_label_min: float = 0.20   # extra labels for multi-label output
    langid_min_confidence: float = 0.60
    n_topics: int = 15
    search_top_k: int = 5

    def ensure_dirs(self) -> None:
        """Create every output directory if it does not exist yet."""
        for path in (self.raw_dir, self.processed_dir, self.models_dir,
                     self.results_dir, self.figures_dir):
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = NewsBotConfig()
