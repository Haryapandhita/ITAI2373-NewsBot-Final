"""Tests for the classifier, sentiment, NER, language detection and intents."""

import pytest

from src.analysis.classifier import NewsClassifier
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.conversation.intent_classifier import IntentClassifier
from src.multilingual.language_detector import LanguageDetector

TRAIN = {
    "sport": ["The team won the football match with a late goal from the striker",
              "Tennis champion wins the final in straight sets at Wimbledon",
              "The coach praised the players after the league victory",
              "Cricket captain scores a century as England win the test"],
    "business": ["Shares rose after the company reported record quarterly profits",
                 "The central bank raised interest rates to fight inflation",
                 "Investors sold stocks as the market fell sharply on Monday",
                 "The firm announced a merger worth billions of dollars"],
    "tech": ["The new smartphone has a faster chip and better camera software",
             "Researchers released an artificial intelligence model for coding",
             "The software update fixes security bugs in the operating system",
             "Broadband and mobile internet speeds increased across the country"],
}


@pytest.fixture(scope="module")
def clf():
    texts, labels = [], []
    for lab, items in TRAIN.items():
        for t in items:
            for k in range(3):  # small augmentation so CV inside calibration works
                texts.append(t + " " + "extra" * k)
                labels.append(lab)
    return NewsClassifier().train(texts, labels)


def test_prediction_structure(clf):
    p = clf.predict_with_confidence("The striker scored twice and the team won the match")
    assert p.label == "sport"
    assert 0 <= p.confidence <= 1
    assert abs(sum(p.probabilities.values()) - 1) < 1e-3
    assert isinstance(p.needs_review, bool)


def test_explanation_has_terms(clf):
    exp = clf.explain_prediction("Shares fell as the market reacted to weak profits")
    assert exp["prediction"]["label"] == "business"
    assert exp["key_terms"] and exp["similar_training_articles"]


def test_empty_text_raises(clf):
    with pytest.raises(ValueError):
        clf.predict_with_confidence("   ")


def test_evaluate_keys(clf):
    r = clf.evaluate(["The team won the cup final"], ["sport"])
    assert {"accuracy", "macro_f1", "confusion_matrix"} <= set(r)


def test_sentiment_direction():
    sa = SentimentAnalyzer()
    pos = sa.analyze("The team celebrated a wonderful victory. Fans were delighted.")
    neg = sa.analyze("The war killed many people. The crisis is getting worse.")
    assert pos["label"] == "positive" and neg["label"] == "negative"
    assert abs(sum(pos["emotions"].values()) - 1) < 1e-3


def test_news_lexicon_fix():
    sa = SentimentAnalyzer()
    assert sa.analyze("The party held a meeting.")["label"] == "neutral"


def test_language_detector():
    texts = {"en": "The government announced new measures to support the economy and jobs",
             "es": "El gobierno anunció nuevas medidas para apoyar la economía y el empleo",
             "de": "Die Regierung kündigte neue Maßnahmen zur Unterstützung der Wirtschaft an",
             "fr": "Le gouvernement a annoncé de nouvelles mesures pour soutenir l'économie",
             "it": "Il governo ha annunciato nuove misure per sostenere l'economia e il lavoro",
             "pt": "O governo anunciou novas medidas para apoiar a economia e o emprego"}
    X, y = [], []
    for lang, t in texts.items():
        words = t.split()
        for i in range(len(words) - 3):
            X.append(" ".join(words[i:i + 4]))
            y.append(lang)
        X.append(t)
        y.append(lang)
    det = LanguageDetector(min_confidence=0.2)
    det.model.set_params(tfidf__min_df=1)
    det.train(X, y)
    assert det.detect_language("la economía y el empleo del gobierno")["language"] == "es"
    assert det.detect_language("")["language"] == "und"


def test_intents():
    ic = IntentClassifier().train()
    assert ic.classify_intent("show me sports news")["intent"] == "search"
    assert ic.classify_intent("compare Apple and Google")["intent"] == "compare"
    assert ic.classify_intent("hello")["intent"] == "greeting"
    assert ic.evaluate()["accuracy"] > 0.85
