"""
Intent classification for user queries (Module D, course Module 12).

There is no public labelled dataset for "questions to a news bot", so the
training set is generated programmatically from templates × slot values
(categories, entities, timeframes, languages) – a standard bootstrapping
technique for task-oriented dialogue. Evaluation uses a separate set of
hand-written queries with different wording (``HELD_OUT_QUERIES``).
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

INTENTS = ["search", "summarize", "sentiment", "compare", "topics", "trends", "entity",
           "classify", "translate", "insights", "export", "help", "greeting"]

SLOTS = {
    "CAT": ["tech", "technology", "sports", "sport", "business", "politics", "political", "economy",
            "football", "science", "entertainment", "finance", "market"],
    "ENT": ["Apple", "Google", "Tesla", "Trump", "Biden", "China", "the EU", "Microsoft", "OpenAI",
            "Manchester United", "the UK", "Germany", "Elon Musk", "Nvidia", "Real Madrid", "India"],
    "TIME": ["this week", "last month", "in 2024", "this year", "recently", "last year", "in 2025",
             "lately", "since January", "in the last quarter", "today"],
    "LANG": ["Spanish", "German", "French", "Italian", "Portuguese", "English"],
    "SENT": ["positive", "negative", "good", "bad", "optimistic", "pessimistic"],
}

TEMPLATES: Dict[str, List[str]] = {
    "search": ["show me {CAT} news", "find articles about {ENT}", "any news on {ENT} {TIME}?",
               "show me {SENT} {CAT} news from {TIME}", "I want to read about {ENT}", "search for {ENT} stories",
               "what's new in {CAT}?", "get me the latest {CAT} articles", "articles mentioning {ENT}",
               "look up {CAT} stories {TIME}", "news about {ENT} in {LANG}", "list {CAT} headlines",
               "find {LANG} articles about {CAT}", "{ENT} news please", "anything about {ENT}?",
               "what happened with {ENT} {TIME}", "latest on {ENT}", "show {SENT} stories about {ENT}",
               "show {LANG} coverage of {CAT}", "display {LANG} reporting on {ENT}", "browse {CAT} coverage"],
    "summarize": ["summarize {CAT} news {TIME}", "give me a summary of {ENT} coverage", "tl;dr on {ENT}",
                  "summarise the articles about {ENT}", "can you sum up {CAT} news", "brief me on {ENT}",
                  "short summary of {CAT} stories {TIME}", "what are the key points about {ENT}?",
                  "recap {CAT} news {TIME}", "give me the gist of {ENT} news", "summarize them",
                  "summarize this article", "condense the {CAT} coverage", "executive summary of {ENT}",
                  "sum up the {CAT} news", "in short, what is going on with {ENT}?", "what's going on with {ENT} in a nutshell",
                  "explain {CAT} news in two sentences"],
    "sentiment": ["what is the sentiment of {CAT} news?", "how positive is coverage of {ENT}?",
                  "is {ENT} coverage negative?", "analyze sentiment about {ENT} {TIME}",
                  "how do people feel about {ENT}", "what's the tone of {CAT} articles",
                  "is the news about {ENT} good or bad", "mood of {CAT} coverage {TIME}",
                  "how negative is {LANG} news about {ENT}", "emotional tone of {ENT} stories",
                  "sentiment analysis for {CAT}", "are {CAT} stories mostly positive?",
                  "how cheerful is {CAT} coverage", "what are the feelings towards {ENT}", "does {ENT} get good press?",
                  "attitude of the media towards {ENT}", "how happy or sad are {CAT} articles"],
    "compare": ["compare {ENT} and {ENT}", "{ENT} vs {ENT} coverage", "compare {CAT} and {CAT} sentiment",
                "how does {ENT} coverage differ from {ENT}", "compare {LANG} and {LANG} news on {ENT}",
                "difference between {CAT} and {CAT} news", "{ENT} versus {ENT}",
                "which is covered more, {ENT} or {ENT}?", "compare coverage across languages of {ENT}",
                "contrast {LANG} and English coverage of {CAT}",
                "{ENT} or {ENT}: which gets more attention?", "who is covered better, {ENT} or {ENT}?",
                "put {ENT} and {ENT} side by side"],
    "topics": ["what topics are trending in {CAT}?", "main topics {TIME}", "what are people writing about {TIME}",
               "show me the topics", "what themes appear in {CAT} news", "topic breakdown of {LANG} news",
               "discover hidden topics", "what subjects dominate {CAT}", "list the main themes",
               "which topics are emerging?", "topic model results"],
    "trends": ["how has sentiment about {ENT} changed over time?", "trend of {CAT} coverage {TIME}",
               "show the timeline for {ENT}", "is {CAT} news getting more negative?",
               "how did coverage of {ENT} evolve", "track {CAT} sentiment over time",
               "plot {ENT} mentions by month", "what changed since {TIME}", "{CAT} trend",
               "evolution of topics over time", "sentiment over the last year"],
    "entity": ["who is {ENT}?", "how are {ENT} and {ENT} connected?", "what is the relationship between {ENT} and {ENT}",
               "tell me about {ENT}", "who is connected to {ENT}", "background on {ENT}",
               "what companies are linked to {ENT}", "explain the link between {ENT} and {ENT}",
               "which people are associated with {ENT}", "entity network for {ENT}"],
    "classify": ["classify this article: {ENT} announced record profits today",
                 "what category is this: the match ended two nil", "analyze this article",
                 "which section does this story belong to", "categorize: parliament voted on the new bill",
                 "analyse this text for me", "label this news story", "is this tech or business news?",
                 "run a full analysis on this article", "what kind of news is this"],
    "translate": ["translate this to {LANG}", "translate the {LANG} articles about {ENT}",
                  "what does this {LANG} article say?", "show {LANG} news in English",
                  "translate: {ENT} gana el partido", "can you translate {LANG} coverage of {CAT}",
                  "english version of the {LANG} story", "translate into English please"],
    "insights": ["give me an overview", "what are the key insights?", "generate a report on {CAT}",
                 "overall insights {TIME}", "what should I know about the news {TIME}",
                 "dashboard summary", "any interesting patterns?", "big picture of {CAT} coverage",
                 "insights about {LANG} media", "what stands out in the data?",
                 "main conclusions from the news data", "summarize the findings of the analysis", "key findings please"],
    "export": ["export the results as csv", "download a report", "save this as pdf", "export to json",
               "create a report I can share", "export {CAT} articles", "give me a markdown report",
               "save results", "export the summary to html"],
    "help": ["help", "what can you do?", "how do I use this?", "what kind of questions can I ask?",
             "show me the commands", "I'm lost", "options", "how does this work"],
    "greeting": ["hi", "hello", "hey newsbot", "good morning", "thanks", "thank you", "bye", "cheers",
                 "hello there", "good evening", "see you"],
}

# Hand-written evaluation queries – different wording from the templates.
HELD_OUT_QUERIES: List[Tuple[str, str]] = [
    ("Any good news in technology lately?", "search"), ("Pull up stories on Nvidia", "search"),
    ("I'd like to see sport articles from last week", "search"), ("Find me something about OpenAI", "search"),
    ("Show German coverage of the economy", "search"), ("What's the latest about Tesla?", "search"),
    ("Could you condense today's politics news?", "summarize"), ("Sum up what happened with Apple", "summarize"),
    ("In a few sentences, what's going on with China?", "summarize"), ("Quick recap of business news", "summarize"),
    ("Is the press friendly towards Musk?", "sentiment"), ("How upbeat are sports stories?", "sentiment"),
    ("Is coverage of the economy mostly gloomy?", "sentiment"), ("Feelings about Biden in the news?", "sentiment"),
    ("Google or Microsoft – who gets better press?", "compare"), ("Contrast Spanish and French politics news", "compare"),
    ("Tech versus business sentiment", "compare"), ("Compare Trump to Biden", "compare"),
    ("What are the hot themes in politics?", "topics"), ("Which subjects are popular right now?", "topics"),
    ("Break the news down into topics", "topics"), ("What themes keep coming up?", "topics"),
    ("Has tech news become more positive over time?", "trends"), ("Show me how Apple coverage changed", "trends"),
    ("Monthly trend for politics", "trends"), ("Did sentiment drop in 2025?", "trends"),
    ("Who is Christine Lagarde?", "entity"), ("What links Apple and Google?", "entity"),
    ("Tell me who works with Elon Musk", "entity"), ("How is India related to China in the news?", "entity"),
    ("Which category fits this: shares fell after the earnings call", "classify"), ("Analyse the following text", "classify"),
    ("What section would this story go in?", "classify"),
    ("Translate this Spanish headline", "translate"), ("Give me the French articles in English", "translate"),
    ("What are the main takeaways from the data?", "insights"), ("Give me a big-picture report", "insights"),
    ("Download everything as a CSV file", "export"), ("Make me a PDF report", "export"),
    ("What commands do you support?", "help"), ("How can you help me?", "help"),
    ("Hey there!", "greeting"), ("Thanks a lot", "greeting"),
]


def generate_training_data(n_per_template: int = 6, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Fill every template ``n_per_template`` times with random slot values."""
    rng = random.Random(seed)
    X, y = [], []
    for intent, templates in TEMPLATES.items():
        for tpl in templates:
            for _ in range(n_per_template):
                text = tpl
                for slot, values in SLOTS.items():
                    while "{" + slot + "}" in text:
                        text = text.replace("{" + slot + "}", rng.choice(values), 1)
                # light noise: casing / punctuation variation
                if rng.random() < 0.3:
                    text = text.lower()
                if rng.random() < 0.2:
                    text = text.rstrip("?.!") + rng.choice(["", "?", " please", "!"])
                X.append(text)
                y.append(intent)
    return X, y


class IntentClassifier:
    """TF-IDF (word + char) + logistic regression intent model with confidence."""

    def __init__(self, min_confidence: float = 0.35):
        """Create the object; parameters are described in the class docstring."""
        self.min_confidence = min_confidence
        self.model = Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)),
            ])),
            ("lr", LogisticRegression(C=10, max_iter=3000, class_weight="balanced")),
        ])
        self.fitted = False

    def train(self, X: List[str] = None, y: List[str] = None) -> "IntentClassifier":
        """Fit the model on texts and labels; returns ``self`` for chaining."""
        if X is None:
            X, y = generate_training_data()
        self.model.fit(X, y)
        self.fitted = True
        return self

    def classify_intent(self, query: str) -> dict:
        """Return the intent, its probability and the runner-up."""
        if not self.fitted:
            self.train()
        probs = self.model.predict_proba([query])[0]
        classes = self.model.classes_
        order = np.argsort(-probs)
        intent = classes[order[0]] if probs[order[0]] >= self.min_confidence else "unknown"
        return {"intent": intent, "confidence": round(float(probs[order[0]]), 3),
                "runner_up": classes[order[1]], "runner_up_confidence": round(float(probs[order[1]]), 3)}

    def evaluate(self, queries: List[Tuple[str, str]] = None) -> dict:
        """Accuracy, macro-F1, per-intent report and misclassified queries on the held-out set."""
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        queries = queries or HELD_OUT_QUERIES
        X = [q for q, _ in queries]
        y = [l for _, l in queries]
        pred = list(self.model.predict(X))
        errors = [(q, t, p) for q, t, p in zip(X, y, pred) if t != p]
        return {"n": len(y), "accuracy": round(accuracy_score(y, pred), 4),
                "macro_f1": round(f1_score(y, pred, average="macro"), 4),
                "report": classification_report(y, pred, output_dict=True, zero_division=0),
                "errors": errors}

    def save(self, path) -> None:
        """Serialise this object to ``path`` with joblib (compressed)."""
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "IntentClassifier":
        """Load an object previously written with :meth:`save`."""
        return joblib.load(path)
