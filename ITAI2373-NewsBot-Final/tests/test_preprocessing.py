"""Unit tests for data processing."""

import pandas as pd
import pytest

from src.data_processing.data_validator import validate_corpus
from src.data_processing.feature_extractor import LSAEmbedder, StyleFeatures
from src.data_processing.text_preprocessor import TextPreprocessor


@pytest.fixture
def pre():
    return TextPreprocessor()


def test_clean_keeps_case_and_removes_urls(pre):
    out = pre.clean("Apple’s CEO spoke.  Visit https://apple.com now")
    assert "Apple's" in out and "https" not in out and "  " not in out


def test_normalize_currency_and_contractions(pre):
    out = pre.normalize("The firm won't pay £168m, it can't.")
    assert "currency_gbp" in out
    assert "not" in out          # negation kept
    assert "firm" in out and "the" not in out.split()


def test_bbc_style_stripped_apostrophes(pre):
    assert "not" in pre.normalize("they don t know").split()


def test_language_specific_stopwords():
    es = TextPreprocessor(language="es").normalize("El banco central de la economía")
    assert "banco" in es and "el" not in es.split() and "de" not in es.split()


def test_sentence_split_handles_abbreviations_and_newlines(pre):
    sents = pre.split_sentences("Mr. Smith went to Washington. He met Dr. Jones.\nNew paragraph here")
    assert sents == ["Mr. Smith went to Washington.", "He met Dr. Jones.", "New paragraph here"]


def test_empty_and_non_string_input(pre):
    assert pre.clean(None) == ""
    assert pre.normalize("") == ""


def test_validator_flags_problems():
    df = pd.DataFrame({"doc_id": ["a", "a"], "text": ["short text", "short text"],
                       "category": ["x", "y"], "language": ["en", "en"]})
    rep = validate_corpus(df, min_words=5)
    assert not rep.ok
    assert any("duplicate doc_ids" in i for i in rep.issues)


def test_validator_missing_columns():
    rep = validate_corpus(pd.DataFrame({"text": ["x"]}))
    assert not rep.ok and "missing columns" in rep.issues[0]


def test_style_features_shape():
    X = StyleFeatures().transform(["Hello World 2025?", ""])
    assert X.shape == (2, len(StyleFeatures.names))


def test_lsa_embedder_normalised():
    texts = ["football match goal team", "team wins football cup", "stock market shares fall",
             "market investors shares bank", "election vote party minister", "minister party election campaign"] * 3
    emb = LSAEmbedder(n_components=4).fit(texts)
    v = emb.transform(["football team"])
    assert abs((v ** 2).sum() - 1) < 1e-6
