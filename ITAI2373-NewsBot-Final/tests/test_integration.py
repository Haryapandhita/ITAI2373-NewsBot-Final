"""
Integration tests.

* conversation pipeline with a stub backend (always runs);
* the full ``NewsBot2`` system on a sample of the real data (runs when the
  datasets have been downloaded to ``data/processed``; skipped otherwise).
"""

import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from config.settings import DEFAULT_CONFIG
from src.conversation.query_processor import ConversationContext, QueryProcessor
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer
from src.utils.export import export_results

HAS_DATA = (DEFAULT_CONFIG.processed_dir / "bbc.pkl").exists() and (DEFAULT_CONFIG.processed_dir / "webz.pkl").exists()


def test_follow_up_inherits_filters():
    qp = QueryProcessor(reference_date=pd.Timestamp("2026-11-20", tz="UTC"))
    ctx = ConversationContext()
    first = qp.parse("Show me positive tech news from this year", ctx)[0]
    assert first.intent == "search" and first.slots["category"] == ["tech"]
    ctx.remember(first, ["a", "b"])
    nxt = qp.parse("what about last year?", ctx)[0]
    assert nxt.is_follow_up and nxt.intent == "search"
    assert nxt.slots["category"] == ["tech"] and nxt.slots["date_from"] == "2025-01-01"


def test_multi_part_query():
    qp = QueryProcessor(reference_date=pd.Timestamp("2026-11-20", tz="UTC"))
    parts = qp.parse("show German business news and then summarize them")
    assert [p.intent for p in parts] == ["search", "summarize"]
    assert parts[1].slots["language"] == ["de"]


def test_cultural_context():
    out = CrossLingualAnalyzer.extract_cultural_context("El Ibex cayó y el BCE subió los tipos 2 €", "es")
    assert "Ibex" in out["references"] and "EUR" in out["currencies"]


def test_export_formats(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    for ext in ("csv", "json", "md", "html"):
        p = export_results(df, tmp_path / f"out.{ext}")
        assert p.exists() and p.stat().st_size > 0


@pytest.mark.skipif(not HAS_DATA, reason="datasets not downloaded (run notebook 01 first)")
def test_full_system_end_to_end(tmp_path):
    from src.data_processing.data_loader import load_all, sample_articles
    from src.newsbot import NewsBot2
    cfg = dataclasses.replace(DEFAULT_CONFIG, models_dir=tmp_path / "models", processed_dir=tmp_path / "proc",
                              results_dir=tmp_path / "res", figures_dir=tmp_path / "res" / "fig")
    data = load_all(DEFAULT_CONFIG)
    bot = NewsBot2(cfg, use_transformers=False).build(data, max_webz=1500, kg_docs=300)
    art = sample_articles()[0]
    res = bot.analyze_article(art["text"], art["title"])
    assert not res["errors"], res["errors"]
    assert res["classification"]["label"] in DEFAULT_CONFIG.categories
    assert res["language"]["language"] == "en"
    es = bot.analyze_article(sample_articles()[3]["text"])
    assert es["language"]["language"] == "es"
    reply = bot.chat("show me sports news")
    assert "1." in reply
    assert "Comparison" in bot.chat("compare Apple and Google")
    batch = bot.batch_analysis([a["text"] for a in sample_articles()], progress=False)
    assert len(batch) == 4 and batch["category"].notna().all()
    assert "# NewsBot 2.0" in bot.generate_insights_report()
