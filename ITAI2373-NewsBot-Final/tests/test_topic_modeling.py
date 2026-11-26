"""Tests for topic modelling, summarisation and semantic search."""

import pandas as pd
import pytest

from src.analysis.topic_modeler import TopicModeler
from src.language_models.embeddings import SemanticEmbedder, SemanticSearchEngine
from src.language_models.summarizer import IntelligentSummarizer

DOCS = (["football team match goal striker league season win"] * 8 +
        ["stock market shares investors bank profit economy growth"] * 8 +
        ["election vote party minister government campaign parliament"] * 8 +
        ["smartphone software chip computer internet technology app"] * 8)
DOCS = [d + f" filler{i % 4}" for i, d in enumerate(DOCS)]


@pytest.mark.parametrize("method", ["lda", "nmf"])
def test_topic_model_fits(method):
    tm = TopicModeler(4, method, max_features=500)
    tm.fit_transform(DOCS)
    assert tm.doc_topic_.shape == (len(DOCS), 4)
    assert abs(tm.doc_topic_.sum(axis=1) - 1).max() < 1e-6
    words = " ".join(w for k in range(4) for w in tm.get_topic_words(k, 8))
    assert "football" in words and "bank" in words and "election" in words
    assert -1 <= tm.coherence() <= 1
    assert 0 < tm.diversity() <= 1


def test_article_topics_and_trends():
    tm = TopicModeler(4, "nmf", max_features=500)
    tm.fit_transform(DOCS)
    topics = tm.get_article_topics("the striker scored a goal for the football team")
    assert topics and "football" in tm.get_topic_words(topics[0]["topic"], 8)
    dates = pd.date_range("2024-01-01", periods=len(DOCS), freq="10D")
    tr = tm.track_topic_trends(dates)
    assert {"period", "topic", "share"} <= set(tr.columns)


def test_textrank_summary_length():
    text = ("Apple announced a new chip on Tuesday. The chip is faster than the previous model. "
            "Analysts expect strong sales this year. The company also cut prices in Europe. "
            "Shares rose two percent after the announcement. Rivals are expected to respond soon.")
    s = IntelligentSummarizer()
    out = s.summarize_article(text, "brief", "textrank")
    assert out["method"] == "textrank"
    assert len(s.pre.split_sentences(out["summary"])) == 2
    q = s.assess_summary_quality(text, out["summary"], reference="Apple unveils faster chip")
    assert "rouge1" in q and 0 < q["compression"] < 1


def test_headlines():
    h = IntelligentSummarizer().generate_headlines("Apple announced a new chip on Tuesday, which is fast. More text follows here.")
    assert set(h) == {"informative", "keyword", "social"}


def test_semantic_search_filters():
    df = pd.DataFrame({"doc_id": [f"d{i}" for i in range(len(DOCS))], "title": DOCS, "text": DOCS,
                       "category": ["sport"] * 8 + ["business"] * 8 + ["politics"] * 8 + ["tech"] * 8,
                       "language": "en"})
    eng = SemanticSearchEngine(SemanticEmbedder(use_transformer=False, lsa_components=8)).encode_documents(df)
    res = eng.semantic_search("football goal", top_k=3)
    assert (res["category"] == "sport").all()
    res = eng.semantic_search("football goal", top_k=3, filters={"category": "tech"})
    assert (res["category"] == "tech").all()
    sim = eng.find_similar_articles(DOCS[20], top_k=2)
    assert (sim["category"] == "politics").all()
