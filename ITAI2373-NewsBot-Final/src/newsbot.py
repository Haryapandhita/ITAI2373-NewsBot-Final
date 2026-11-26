"""
NewsBot 2.0 – system orchestrator.

``NewsBot2`` wires the four modules together:

    A  Content analysis      src.analysis.*
    B  Language understanding & generation   src.language_models.*
    C  Multilingual intelligence             src.multilingual.*
    D  Conversational interface              src.conversation.*

Typical use::

    from src.newsbot import NewsBot2
    bot = NewsBot2().build()                 # loads data, trains / loads models
    bot.analyze_article(text)                # full analysis of one article
    bot.chat("Show me negative tech news from 2025")
    bot.generate_insights_report()
"""

from __future__ import annotations

import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from config.settings import DEFAULT_CONFIG, NewsBotConfig
from src.analysis.classifier import NewsClassifier
from src.analysis.ner_extractor import EntityRelationshipMapper
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.analysis.topic_modeler import TopicModeler
from src.conversation.intent_classifier import IntentClassifier
from src.conversation.query_processor import ConversationContext, ParsedQuery, QueryProcessor
from src.conversation.response_generator import HELP_TEXT, ResponseGenerator
from src.data_processing.data_loader import load_all
from src.data_processing.text_preprocessor import TextPreprocessor
from src.language_models.embeddings import SemanticEmbedder, SemanticSearchEngine
from src.language_models.generator import ContentEnhancer, OllamaClient
from src.language_models.summarizer import IntelligentSummarizer
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer
from src.multilingual.language_detector import LANGUAGE_NAMES, LanguageDetector
from src.multilingual.translator import Translator
from src.utils.export import article_report_markdown, export_results
from src.utils.logging_utils import get_logger

log = get_logger("newsbot")


class NewsBot2:
    """
    The complete NewsBot 2.0 system.

    Parameters
    ----------
    config : NewsBotConfig
    use_transformers : bool, optional
        Override ``config.use_transformers`` (HuggingFace models for
        embeddings, summarisation and translation; classical fallbacks otherwise).
    """

    def __init__(self, config: NewsBotConfig = DEFAULT_CONFIG, use_transformers: Optional[bool] = None):
        """Create the object; parameters are described in the class docstring."""
        self.cfg = config
        self.cfg.ensure_dirs()
        self.use_transformers = config.use_transformers if use_transformers is None else use_transformers
        self.corpus: Optional[pd.DataFrame] = None
        self.classifier: Optional[NewsClassifier] = None
        self.sentiment = SentimentAnalyzer()
        self.mapper: Optional[EntityRelationshipMapper] = None
        self.topics: Optional[TopicModeler] = None
        self.summarizer: Optional[IntelligentSummarizer] = None
        self.search: Optional[SemanticSearchEngine] = None
        self.langid: Optional[LanguageDetector] = None
        self.translator: Optional[Translator] = None
        self.enhancer: Optional[ContentEnhancer] = None
        self.intents: Optional[IntentClassifier] = None
        self.query_processor: Optional[QueryProcessor] = None
        self.responder = ResponseGenerator()
        self.context = ConversationContext()
        self.timings: Dict[str, float] = {}
        self.llm = OllamaClient(config.ollama_host, config.ollama_model) if config.use_llm else None
        self.language_label_issues = 0

    # ================================================================ build
    def _step(self, name: str, fn):
        t0 = time.time()
        out = fn()
        self.timings[name] = round(time.time() - t0, 2)
        log.info("%-28s %6.1fs", name, self.timings[name])
        return out

    def build(self, data: Optional[Dict[str, pd.DataFrame]] = None, max_webz: Optional[int] = None,
              kg_docs: int = 4000, retrain: bool = False) -> "NewsBot2":
        """
        Load data and prepare every component. Heavy, re-usable artefacts
        (enriched corpus, classifier, language detector) are cached in
        ``data/processed`` and ``data/models``; ``retrain=True`` rebuilds them.
        """
        data = data or self._step("load data", lambda: load_all(self.cfg))
        bbc, webz = data["bbc"].copy(), data["webz"].copy()
        if max_webz and len(webz) > max_webz:
            webz = webz.sample(max_webz, random_state=self.cfg.random_state)
        corpus = pd.concat([bbc, webz], ignore_index=True)
        corpus["full_text"] = corpus["title"].fillna("") + ". " + corpus["text"].fillna("")

        self.classifier = self._step("classifier", lambda: self._load_or_train_classifier(bbc, webz, retrain))
        self.langid = self._step("language detector", lambda: self._load_or_train_langid(webz, retrain))
        self.corpus = self._step("enrich corpus", lambda: self._enrich(corpus, retrain))
        self.mapper = self.mapper or EntityRelationshipMapper(self.cfg.spacy_model, self.cfg.spacy_multilingual_model)
        en = self.corpus[self.corpus["language"] == "en"]
        kg_sample = en.sample(min(kg_docs, len(en)), random_state=1) if len(en) > kg_docs else en
        self._step("knowledge graph", lambda: self.mapper.build_knowledge_graph(
            kg_sample["full_text"].str[:3000], kg_sample["category"], min_cooccurrence=2))
        self.topics = self._step("topic model", lambda: self._fit_topics(en))
        self.summarizer = self._step("summarizer", lambda: IntelligentSummarizer(
            self.use_transformers, self.cfg.summarization_model))
        embedder = SemanticEmbedder(self.cfg.embedding_model, use_transformer=self.use_transformers)
        self.search = self._step("semantic index", lambda: SemanticSearchEngine(embedder).encode_documents(self.corpus))
        self.translator = Translator(use_transformer=self.use_transformers)
        self.enhancer = ContentEnhancer(self.mapper, self.search,
                                        self.corpus[["category", "sentiment"]], self.llm)
        self.intents = self._step("intent classifier", lambda: IntentClassifier().train())
        known = [n for n, d in sorted(self.mapper.graph.nodes(data=True), key=lambda kv: -kv[1]["mentions"])[:400]]
        ref_date = pd.to_datetime(self.corpus["published"], utc=True).max()
        self.query_processor = QueryProcessor(self.intents, self.mapper.nlp, ref_date, known)
        self.responder = ResponseGenerator(self.llm)
        log.info("NewsBot 2.0 ready: %d articles, %d languages", len(self.corpus), self.corpus["language"].nunique())
        return self

    def _load_or_train_classifier(self, bbc: pd.DataFrame, webz: pd.DataFrame, retrain: bool) -> NewsClassifier:
        path = self.cfg.models_dir / "newsbot2_classifier.joblib"
        if path.exists() and not retrain:
            try:
                return NewsClassifier.load(path)
            except Exception as exc:  # e.g. scikit-learn version mismatch in Colab
                log.warning("Could not load saved classifier (%s) – retraining", exc)
        train = training_frame(bbc, webz, self.cfg.random_state)
        clf = NewsClassifier(self.cfg.low_confidence, self.cfg.secondary_label_min).train(train["full"], train["category"])
        clf.save(path)
        return clf

    def _load_or_train_langid(self, webz: pd.DataFrame, retrain: bool) -> LanguageDetector:
        path = self.cfg.models_dir / "language_detector.joblib"
        if path.exists() and not retrain:
            try:
                return LanguageDetector.load(path)
            except Exception as exc:
                log.warning("Could not load language detector (%s) – retraining", exc)
        per_lang = pd.concat([g.sample(min(len(g), 300), random_state=self.cfg.random_state)
                              for _, g in webz.groupby("language")])
        X, y = LanguageDetector.make_training_snippets(per_lang["text"], per_lang["language"])
        det = LanguageDetector(self.cfg.langid_min_confidence).train(X, y)
        det.train_doc_ids = per_lang["doc_id"].tolist()  # so evaluation can exclude them
        det.save(path)
        return det

    def _enrich(self, corpus: pd.DataFrame, retrain: bool) -> pd.DataFrame:
        """Add sentiment, emotions and entity lists to every article (cached)."""
        cache = self.cfg.processed_dir / "corpus_enriched.pkl"
        if cache.exists() and not retrain:
            cached = pd.read_pickle(cache)
            if isinstance(cached, dict) and set(corpus["doc_id"]) <= set(cached["input_ids"]):
                self.language_label_issues = cached.get("language_label_issues", 0)
                df = cached["corpus"]
                return df[df["doc_id"].isin(set(corpus["doc_id"]))].reset_index(drop=True)
        input_ids = corpus["doc_id"].tolist()
        # Data quality: our language detector double-checks the Webz language
        # labels; articles whose label disagrees with a confident detection
        # (or that are in an unsupported language) are dropped.
        det = self.langid.model.predict_proba(corpus["text"].str[:600].tolist())
        pred = np.array(self.langid.model.classes_)[det.argmax(axis=1)]
        conf = det.max(axis=1)
        corpus["detected_language"] = np.where(conf >= 0.6, pred, "und")
        bad = (corpus["detected_language"] != corpus["language"])
        self.language_label_issues = int(bad.sum())
        log.info("Language check: %d articles with wrong/unsupported language label removed", int(bad.sum()))
        corpus = corpus[~bad].reset_index(drop=True)
        scores = self.sentiment.batch_scores(corpus["full_text"].str[:6000])
        corpus = pd.concat([corpus.reset_index(drop=True), scores], axis=1)
        self.mapper = EntityRelationshipMapper(self.cfg.spacy_model, self.cfg.spacy_multilingual_model)
        corpus["entities"] = self.mapper.corpus_entities(corpus["full_text"], corpus["language"])
        pd.to_pickle({"input_ids": input_ids, "corpus": corpus,
                      "language_label_issues": self.language_label_issues}, cache)
        return corpus

    def _fit_topics(self, en: pd.DataFrame) -> TopicModeler:
        tm = TopicModeler(self.cfg.n_topics, "nmf", random_state=self.cfg.random_state)
        tm.fit_transform(en["full_text"])
        self._topic_doc_ids = en["doc_id"].tolist()
        self._topic_dates = en["published"].tolist()
        return tm

    # ===================================================== single article
    def analyze_article(self, text: str, title: str = "", summary_type: str = "balanced",
                        enhance: bool = True) -> dict:
        """
        Comprehensive analysis of one article in any supported language:
        language → (translation) → classification, sentiment, entities,
        relations, topics, summary, headlines, enhancements, cultural context.
        Each step is isolated: a failure is reported under ``errors`` and the
        rest of the analysis still runs.
        """
        if not text or not text.strip():
            raise ValueError("Empty article")
        full = (title + ". " if title else "") + text
        out: Dict[str, object] = {"errors": {}}

        def safe(key, fn):
            """Run one analysis step and record (instead of raise) its error."""
            try:
                out[key] = fn()
            except Exception as exc:  # keep going, report the problem
                out["errors"][key] = f"{type(exc).__name__}: {exc}"
                log.debug(traceback.format_exc())

        safe("language", lambda: self.langid.detect_language(full))
        lang = out.get("language", {}).get("language", "en")
        english = full
        if lang not in ("en", "und"):
            safe("translation", lambda: self.translator.translate_text(full, lang, "en"))
            if out.get("translation", {}).get("translated"):
                english = out["translation"]["text"]
            safe("cultural_context", lambda: CrossLingualAnalyzer.extract_cultural_context(full, lang))
        safe("classification", lambda: self.classifier.predict_with_confidence(english).to_dict())
        # Inputs the models were not built for are never trusted blindly
        n_words = len(text.split())
        if out.get("classification") and (lang == "und" or n_words < 8):
            out["classification"]["needs_review"] = True
            out["warnings"] = [f"input too short or language undetermined ({n_words} words, language={lang}) "
                               "– results are unreliable"]
        safe("sentiment", lambda: self.sentiment.analyze(english))
        safe("entities", lambda: self.mapper.extract_entities(full if lang in ("en", "und") else full, lang))
        safe("relations", lambda: self.mapper.extract_relationships(english) if lang == "en" or english != full else [])
        if out.get("entities"):
            names = [e["canonical"] for e in out["entities"] if e["label"] in ("PERSON", "ORG", "GPE")]
            safe("entity_sentiment", lambda: self.sentiment.entity_sentiment(english, names))
        safe("topics", lambda: self.topics.get_article_topics(english))
        safe("summary", lambda: self.summarizer.summarize_article(english, summary_type))
        safe("headlines", lambda: self.summarizer.generate_headlines(english))
        if enhance:
            safe("enhancements", lambda: self.enhancer.enhance_article(
                english, out.get("entities"), out.get("classification", {}).get("label"),
                out.get("sentiment", {}).get("score")))
        return out

    def batch_analysis(self, articles: Iterable[str], enhance: bool = False, progress: bool = True) -> pd.DataFrame:
        """Analyse many articles; one bad article never stops the batch."""
        rows = []
        articles = list(articles)
        for i, text in enumerate(articles):
            try:
                a = self.analyze_article(text, enhance=enhance)
                rows.append({"idx": i, "language": a.get("language", {}).get("language"),
                             "category": a.get("classification", {}).get("label"),
                             "confidence": a.get("classification", {}).get("confidence"),
                             "needs_review": a.get("classification", {}).get("needs_review"),
                             "sentiment": a.get("sentiment", {}).get("score"),
                             "sentiment_label": a.get("sentiment", {}).get("label"),
                             "n_entities": len(a.get("entities", [])),
                             "top_topic": (a.get("topics") or [{}])[0].get("label"),
                             "summary": (a.get("summary") or {}).get("summary", ""),
                             "errors": "; ".join(a["errors"].values())})
            except Exception as exc:
                rows.append({"idx": i, "errors": str(exc)})
            if progress and (i + 1) % 25 == 0:
                log.info("analysed %d/%d", i + 1, len(articles))
        return pd.DataFrame(rows)

    # ===================================================== conversation
    def chat(self, message: str) -> str:
        """Chatbot entry point: returns the reply text (see ``process_query`` for data)."""
        return self.process_query(message)["response"]

    def process_query(self, message: str) -> dict:
        """Parse a (multi-part) message, execute each part and build the reply."""
        replies, payloads = [], []
        for pq in self.query_processor.parse(message, self.context):
            try:
                text, data, ids = self._execute(pq)
            except Exception as exc:
                log.warning("query failed: %s", exc)
                text, data, ids = f"Sorry, something went wrong while answering that ({exc}).", None, None
            text = self.responder.finalize(text)
            self.context.remember(pq, ids, text)
            replies.append(text)
            payloads.append({"intent": pq.intent, "confidence": pq.confidence, "slots": pq.slots,
                             "follow_up": pq.is_follow_up, "data": data})
        return {"response": "\n\n".join(replies), "parsed": payloads}

    def _subset(self, pq: ParsedQuery, use_sentiment_filter: bool = True) -> pd.DataFrame:
        """Articles matching the query filters (+ entity mentions)."""
        df = self.corpus
        filters = pq.filters()
        if not use_sentiment_filter:
            filters.pop("sentiment_label", None)
        mask = self.search._filter_mask(filters)
        ents = pq.slots.get("entities") or []
        if ents:
            resolved = [self.mapper.resolve(e) or e for e in ents]
            ent_mask = df["entities"].apply(lambda lst: any(e in lst for e in resolved)).values
            text_mask = df["full_text"].str.contains("|".join(map(__import__("re").escape, ents)), case=False, regex=True).values
            mask &= (ent_mask | text_mask)
        return df[mask]

    def _execute(self, pq: ParsedQuery):
        s, intent = pq.slots, pq.intent
        k = int(s.get("top_k", self.cfg.search_top_k))
        if intent in ("greeting",):
            return self.responder.greeting(pq.text), None, None
        if intent == "help":
            return HELP_TEXT, None, None
        if intent == "classify" or (intent in ("translate",) and s.get("article_text")):
            article = s.get("article_text")
            if not article:
                return "Paste the article after a colon, e.g. \"classify this: <text>\".", None, None
            if intent == "translate":
                lang = self.langid.detect_language(article)["language"]
                res = self.translator.translate_text(article, lang, "en")
                return self.responder.translation(res, lang), res, None
            analysis = self.analyze_article(article, enhance=False)
            return self.responder.classification(analysis), analysis, None
        if intent == "translate":
            sub = self._subset(pq)
            sub = sub[sub["language"] != "en"].head(3)
            if sub.empty:
                return "No non-English articles match that request.", None, None
            outs = [f"[{r.language}] {r.title}\n → " + self.translator.translate_text(r.title, r.language)["text"]
                    for r in sub.itertuples()]
            return "Translated headlines:\n" + "\n".join(outs), None, sub["doc_id"].tolist()
        if intent == "search":
            query = pq.text
            if pq.is_follow_up and self.context.last_query is not None and len(pq.text.split()) < 5:
                query = self.context.last_query.text
            ents = s.get("entities") or []
            filters = pq.filters()
            res = self.search.semantic_search(" ".join(ents) + " " + query, top_k=k, filters=filters)
            if ents:  # make sure entity searches really mention the entity
                sub = self._subset(pq)
                if not sub.empty:
                    res = self.search.semantic_search(" ".join(ents) + " " + query, top_k=k,
                                                      filters={**filters, "doc_id": sub["doc_id"].tolist()})
            return self.responder.search(res, s), res, res["doc_id"].tolist()
        if intent == "summarize":
            if pq.is_follow_up and self.context.last_result_ids and not s.get("entities"):
                sub = self.corpus[self.corpus["doc_id"].isin(self.context.last_result_ids)]
            else:
                sub = self._subset(pq)
                if len(sub) > 8:
                    q = " ".join(s.get("entities") or []) + " " + pq.text
                    sub = self.search.semantic_search(q, top_k=8, filters={"doc_id": sub["doc_id"].tolist()})
            n = {"brief": 3, "detailed": 7}.get(s.get("length"), 5)
            english = self._english_texts(sub.head(8))
            summary = self.summarizer.summarize_multiple_articles(english, n_sentences=n,
                                                                  focus_topic=" ".join(s.get("entities") or []) or None)
            if not summary.strip():  # e.g. pages without sentence punctuation
                summary = " ".join(f"{t}." for t in sub["title"].head(n))
            return self.responder.summary(summary, len(sub.head(8)), s), summary, sub["doc_id"].tolist()
        if intent == "sentiment":
            sub = self._subset(pq, use_sentiment_filter=False)
            return self.responder.sentiment(self._sentiment_stats(sub), s), None, sub["doc_id"].head(50).tolist()
        if intent == "compare":
            return self.responder.compare(self._compare(pq)), None, None
        if intent == "topics":
            sub = self._subset(pq)
            table = self.topics.topics_table()
            if s.get("category") or s.get("language") or s.get("date_from"):
                ids = set(sub["doc_id"])
                mask = np.array([d in ids for d in self._topic_doc_ids])
                if mask.sum() >= 5:
                    prev = self.topics.doc_topic_[mask].mean(axis=0)
                    table = table.assign(prevalence=prev[table["topic"].values]).sort_values("prevalence", ascending=False)
            trends = self.topics.track_topic_trends(self._topic_dates)
            return self.responder.topics(table, TopicModeler.emerging_topics(trends.dropna())), table, None
        if intent == "trends":
            sub = self._subset(pq, use_sentiment_filter=False).dropna(subset=["published"])
            subject = self.responder.describe_filters(s)
            tl = SentimentAnalyzer.track_over_time(sub, group_col=None)
            return self.responder.trends(tl, subject), tl, None
        if intent == "entity":
            ents = s.get("entities") or []
            if len(ents) >= 2:
                info = self.mapper.find_entity_connections(ents[0], ents[1])
            elif ents:
                info = self.mapper.entity_profile(ents[0])
            else:
                return "Which person, organisation or place do you mean?", None, None
            return self.responder.entity(info), info, None
        if intent == "insights":
            sub = self._subset(pq)
            ins = self.enhancer.generate_insights(sub if len(sub) else self.corpus)
            return self.responder.insights(ins, self.context.top_interests()), ins, None
        if intent == "export":
            fmt = s.get("format", "csv")
            ids = self.context.last_result_ids
            sub = self.corpus[self.corpus["doc_id"].isin(ids)] if ids else self._subset(pq).head(100)
            cols = [c for c in ["doc_id", "title", "category", "language", "published", "sentiment",
                                "sentiment_label", "site"] if c in sub.columns]
            if fmt in ("pdf", "html", "markdown"):
                path = self.cfg.results_dir / "exports" / ("chat_report." + ("md" if fmt == "markdown" else "html"))
                export_results(self.generate_insights_report(sub), path, "markdown" if fmt == "markdown" else "html")
            else:
                path = self.cfg.results_dir / "exports" / f"chat_export.{fmt}"
                export_results(sub[cols], path, fmt)
            return self.responder.export(str(path)), str(path), None
        return self.responder.unknown(), None, None

    # ---------------------------------------------------------- helpers
    def _english_texts(self, df: pd.DataFrame) -> List[str]:
        texts = []
        for r in df.itertuples():
            body = r.text[:2500]
            if r.language != "en":
                tr = self.translator.translate_text(body, r.language, "en")
                body = tr["text"] if tr["translated"] else body
            texts.append(body)
        return texts

    @staticmethod
    def _sentiment_stats(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"n": 0}
        from src.analysis.sentiment_analyzer import EMOTIONS
        emo = sub[EMOTIONS].mean().idxmax() if all(e in sub for e in EMOTIONS) else None
        return {"n": len(sub), "mean": float(sub["sentiment"].mean()),
                "pos_share": float((sub["sentiment_label"] == "positive").mean()),
                "neg_share": float((sub["sentiment_label"] == "negative").mean()), "top_emotion": emo}

    def _compare(self, pq: ParsedQuery) -> List[dict]:
        s = pq.slots
        groups = []
        if len(s.get("entities") or []) >= 2:
            groups = [("entities", [e]) for e in s["entities"][:3]]
        elif len(s.get("category") or []) >= 2:
            groups = [("category", [c]) for c in s["category"][:3]]
        elif len(s.get("language") or []) >= 2:
            groups = [("language", [l]) for l in s["language"][:3]]
        elif s.get("language") and s.get("entities"):
            groups = [("language", [s["language"][0]]), ("language", ["en"])]
        rows = []
        for key, val in groups:
            slots = {k: v for k, v in s.items() if k != key}
            slots[key] = val
            sub = self._subset(ParsedQuery(pq.text, "compare", 1.0, slots), use_sentiment_filter=False)
            stats = self._sentiment_stats(sub)
            name = LANGUAGE_NAMES.get(val[0], val[0]) if key == "language" else val[0]
            rows.append({"name": name, "n": stats.get("n", 0), "mean": stats.get("mean", 0.0),
                         "pos_share": stats.get("pos_share", 0.0),
                         "top_category": sub["category"].mode().iat[0] if len(sub) else None})
        return rows

    # ===================================================== reporting
    def generate_insights_report(self, df: Optional[pd.DataFrame] = None, report_type: str = "comprehensive") -> str:
        """Markdown report: overview, insights, sentiment by category/language, topics, entities."""
        df = self.corpus if df is None else df
        lines = ["# NewsBot 2.0 – Insights report", "",
                 f"*{len(df):,} articles · {df['language'].nunique()} languages · report type: {report_type}*", "",
                 "## Key insights", ""]
        lines += [f"- {i}" for i in self.enhancer.generate_insights(df)] + [""]
        lines += ["## Sentiment by category", "",
                  df.groupby("category")["sentiment"].agg(["mean", "count"]).round(3).to_markdown(), ""]
        if df["language"].nunique() > 1:
            lines += ["## Coverage by language", "",
                      pd.crosstab(df["language"], df["category"], normalize="index").round(2).to_markdown(), ""]
        if report_type in ("comprehensive", "trends") and self.topics is not None:
            lines += ["## Topics", "", self.topics.topics_table(8)[["label", "prevalence", "npmi"]].to_markdown(index=False), ""]
        ents = Counter(e for lst in df["entities"] for e in lst).most_common(10)
        lines += ["## Most mentioned entities", ""] + [f"- {e} ({c})" for e, c in ents]
        return "\n".join(lines)

    def export_article_report(self, analysis: dict, path: Path, fmt: str = "markdown") -> Path:
        """Write a Markdown/HTML report for one ``analyze_article`` result."""
        return export_results(article_report_markdown(analysis), path, fmt)

    # ===================================================== persistence
    def save(self, path: Optional[Path] = None) -> Path:
        """Pickle the whole system (handy for the web demo / quick restarts)."""
        path = Path(path or self.cfg.models_dir / "newsbot2_system.joblib")
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: Path) -> "NewsBot2":
        """Load an object previously written with :meth:`save`."""
        return joblib.load(path)


def training_frame(bbc: pd.DataFrame, webz: pd.DataFrame, random_state: int = 42,
                   webz_per_class: int = 600) -> pd.DataFrame:
    """
    Training data for the 2.0 classifier: BBC train split (2004–05 style) +
    a balanced sample of English Webz articles (2023–26 vocabulary) so the
    model is not stuck in 2005. Held-out test articles are excluded.
    """
    from sklearn.model_selection import train_test_split
    b = bbc.assign(full=bbc["title"] + ". " + bbc["text"])
    b_train, _ = train_test_split(b, test_size=0.2, stratify=b["category"], random_state=random_state)
    w = webz[webz["language"] == "en"].assign(full=lambda d: d["title"] + ". " + d["text"])
    w_train, _ = train_test_split(w, test_size=0.3, stratify=w["category"], random_state=random_state)
    w_train = pd.concat([g.sample(min(len(g), webz_per_class), random_state=random_state)
                         for _, g in w_train.groupby("category")])
    return pd.concat([b_train[["full", "category"]], w_train[["full", "category"]]], ignore_index=True)
