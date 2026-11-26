"""
Reproduce every number and figure used in the documentation.

    python scripts/run_experiments.py            # full run (~10-15 min on CPU)
    python scripts/run_experiments.py --fast     # smaller samples for a quick check

Outputs: ``data/results/metrics.json``, ``data/results/*.csv`` and
``data/results/figures/*.png|html``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import DEFAULT_CONFIG as CFG
from src.analysis.classifier import NewsClassifier, compare_baselines
from src.analysis.sentiment_analyzer import EMOTIONS, SentimentAnalyzer
from src.analysis.topic_modeler import TopicModeler, select_n_topics
from src.conversation.intent_classifier import IntentClassifier, generate_training_data
from src.data_processing.data_loader import load_all
from src.data_processing.data_validator import validate_corpus
from src.language_models.embeddings import retrieval_eval
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer
from src.multilingual.language_detector import LanguageDetector
from src.newsbot import NewsBot2, training_frame
from src.utils import visualization as viz
from src.utils.evaluation import per_class_table, save_json

FIG = CFG.figures_dir
RES = CFG.results_dir


def main(fast: bool = False, reuse: bool = False):
    t_start = time.time()
    M: dict = {"run": "fast" if fast else "full", "date": time.strftime("%Y-%m-%d %H:%M")}
    data = load_all(CFG)
    bbc, webz = data["bbc"], data["webz"]

    # ------------------------------------------------------------------ data
    M["data"] = {"bbc": validate_corpus(bbc).to_dict(), "webz": validate_corpus(webz).to_dict(),
                 "webz_by_language": webz["language"].value_counts().to_dict(),
                 "webz_by_language_category": pd.crosstab(webz["language"], webz["category"]).to_dict(),
                 "webz_quarters": webz["published"].dt.tz_convert(None).dt.to_period("Q").astype(str).value_counts().sort_index().to_dict()}
    viz.bar(bbc["category"].value_counts(), "BBC articles per category", path=FIG / "01_bbc_categories.png")
    viz.bar(webz["language"].value_counts(), "Webz articles per language", path=FIG / "01_webz_languages.png",
            color=viz.PALETTE[0])
    viz.system_architecture(FIG / "00_architecture.png")

    # -------------------------------------------------------- build system
    bot = NewsBot2(use_transformers=False).build(data, retrain=not reuse)
    M["build_timings_s"] = bot.timings
    M["data"]["language_label_issues_removed"] = bot.language_label_issues
    corpus = bot.corpus

    # -------------------------------------------------------- classification
    b = bbc.assign(full=bbc["title"] + ". " + bbc["text"])
    b_tr, b_te = train_test_split(b, test_size=0.2, stratify=b["category"], random_state=42)
    w = webz[webz["language"] == "en"].assign(full=lambda d: d["title"] + ". " + d["text"])
    _, w_te = train_test_split(w, test_size=0.3, stratify=w["category"], random_state=42)
    if fast:
        w_te = w_te.sample(800, random_state=0)
    base = compare_baselines(b_tr["full"], b_tr["category"], b_te["full"], b_te["category"])
    clf_bbc = NewsClassifier().train(b_tr["full"], b_tr["category"])
    ev_bbc = clf_bbc.evaluate(b_te["full"], b_te["category"])
    base = pd.concat([base, pd.DataFrame([{"model": "NewsBot 2.0 Ensemble (SVM+LR+NB, word+char)",
                                           "accuracy": ev_bbc["accuracy"], "macro_f1": ev_bbc["macro_f1"],
                                           "train_s": round(clf_bbc.train_seconds, 2)}])], ignore_index=True)
    base = pd.concat([base, pd.DataFrame([{"model": "Midterm Linear SVM (reported)", "accuracy": 0.9617,
                                           "macro_f1": 0.9582, "train_s": None}])], ignore_index=True)
    base.to_csv(RES / "classification_baselines.csv", index=False)
    viz.model_comparison(base.dropna(subset=["macro_f1"]), "BBC test set – macro-F1", path=FIG / "02_model_comparison.png")
    cv = clf_bbc.cross_validate(b_tr["full"], b_tr["category"], folds=3 if fast else 5)
    ev_cross_old = clf_bbc.evaluate(w_te["full"], w_te["category"])
    clf2 = bot.classifier
    ev_bbc_new = clf2.evaluate(b_te["full"], b_te["category"])
    ev_cross_new = clf2.evaluate(w_te["full"], w_te["category"])
    viz.confusion_matrix_plot(ev_bbc["confusion_matrix"], ev_bbc["labels"], "Ensemble – BBC test",
                              FIG / "02_confusion_bbc.png")
    viz.confusion_matrix_plot(ev_cross_old["confusion_matrix"], ev_cross_old["labels"],
                              "BBC-only model on 2023–26 news", FIG / "02_confusion_crossdomain_before.png")
    viz.confusion_matrix_plot(ev_cross_new["confusion_matrix"], ev_cross_new["labels"],
                              "NewsBot 2.0 model on 2023–26 news", FIG / "02_confusion_crossdomain_after.png")
    calib = clf_bbc.calibration_table(b_te["full"], b_te["category"])
    preds = [clf2.predict_with_confidence(t) for t in w_te["full"].head(400)]
    review_share = float(np.mean([p.needs_review for p in preds]))
    acc_conf = float(np.mean([p.label == y for p, y in zip(preds, w_te["category"].head(400)) if not p.needs_review]))
    acc_rev = float(np.mean([p.label == y for p, y in zip(preds, w_te["category"].head(400)) if p.needs_review]))
    ml = {}
    for lang in ["es", "de", "fr", "it", "pt"]:
        sub = webz[webz["language"] == lang]
        sub = sub.sample(min(len(sub), 300), random_state=0)
        ml[lang] = clf2.evaluate((sub["title"] + ". " + sub["text"]).tolist(), sub["category"].tolist())["accuracy"]
    M["classification"] = {
        "bbc_split": {"train": len(b_tr), "test": len(b_te)},
        "baselines": base.to_dict("records"),
        "ensemble_bbc_test": {k: ev_bbc[k] for k in ("accuracy", "macro_f1", "ms_per_article")},
        "ensemble_cv_macro_f1": cv,
        "per_class_bbc": per_class_table(ev_bbc["report"]).to_dict("records"),
        "calibration_bbc": calib.to_dict("records"),
        "cross_domain_webz_en": {"n": ev_cross_old["n"],
                                 "bbc_only_model": {k: ev_cross_old[k] for k in ("accuracy", "macro_f1")},
                                 "newsbot2_model": {k: ev_cross_new[k] for k in ("accuracy", "macro_f1")}},
        "newsbot2_model_on_bbc_test": {k: ev_bbc_new[k] for k in ("accuracy", "macro_f1")},
        "review_flag": {"share_flagged": round(review_share, 3), "accuracy_unflagged": round(acc_conf, 3),
                        "accuracy_flagged": round(acc_rev, 3)},
        "non_english_tfidf_accuracy": ml,
    }

    # --------------------------------------------------------------- sentiment
    sa = bot.sentiment
    en = corpus[(corpus["language"] == "en") & corpus["webz_sentiment"].isin(["positive", "negative"])]
    en_s = en.sample(min(len(en), 1500 if fast else 3000), random_state=0)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    plain = SentimentIntensityAnalyzer()
    whole = [plain.polarity_scores(t[:5000])["compound"] for t in en_s["full_text"]]
    def acc(scores):
        pred = np.where(np.array(scores) >= 0, "positive", "negative")
        return round(float((pred == en_s["webz_sentiment"].values).mean()), 4)
    sent_plain = [np.mean([plain.polarity_scores(s)["compound"] for s in sa.pre.split_sentences(t)] or [0]) for t in en_s["full_text"]]
    M["sentiment"] = {
        "eval_vs_webz_labels_n": len(en_s),
        "accuracy_whole_text_vader": acc(whole),
        "accuracy_sentence_avg_vader": acc(sent_plain),
        "accuracy_sentence_avg_news_lexicon": acc(en_s["sentiment"].values),
        "whole_text_saturation_share": round(float(np.mean(np.abs(whole) > 0.9)), 3),
        "by_category": corpus.groupby("category")["sentiment"].mean().round(4).to_dict(),
        "by_language": corpus.groupby("language")["sentiment"].mean().round(4).to_dict(),
        "emotions_by_category": corpus.groupby("category")[EMOTIONS].mean().round(4).to_dict(orient="index"),
    }
    dated = corpus.dropna(subset=["published"])
    tl = SentimentAnalyzer.track_over_time(dated, group_col="category", freq="Q")
    tl = tl[tl["n_articles"] >= 30]
    an = SentimentAnalyzer.detect_anomalies(tl, z=1.5)
    an.to_csv(RES / "sentiment_timeline.csv", index=False)
    viz.timeline(an, "period", "mean_sentiment", "category", "Average sentiment per quarter (Webz, all languages)",
                 "sentence-avg VADER", FIG / "03_sentiment_timeline.png", anomalies=an)
    M["sentiment"]["anomalies"] = an[an["anomaly"]][["period", "category", "mean_sentiment", "z_score"]].to_dict("records")
    viz.heatmap(corpus.groupby("category")[EMOTIONS].mean(), "NRC emotion profile by category",
                FIG / "03_emotions.png", fmt="{:.2f}", cmap="Blues")

    # ------------------------------------------------------------------- NER
    mapper = bot.mapper
    sample = b_te.sample(150, random_state=0)
    cased = sum(len([e for e in mapper.extract_entities(t) if e["label"] in ("PERSON", "ORG", "GPE")]) for t in sample["text"])
    lower = sum(len([e for e in mapper.extract_entities(t.lower()) if e["label"] in ("PERSON", "ORG", "GPE")]) for t in sample["text"])
    g = mapper.graph
    rel_edges = sum(1 for _, _, d in g.edges(data=True) if d.get("relations"))
    roles = Counter()
    for _, _, d in g.edges(data=True):
        roles.update(d.get("relations", {}).keys())
    examples = {}
    for a, c in [("Donald Trump", "China"), ("Elon Musk", "Tesla"), ("Apple", "Google"), ("OpenAI", "Microsoft")]:
        r = mapper.find_entity_connections(a, c)
        examples[f"{a} – {c}"] = {k: r.get(k) for k in ("found", "direct", "weight", "relations", "paths")}
    M["ner"] = {"casing_experiment_150_bbc_articles": {"entities_original_casing": cased, "entities_lowercased": lower,
                                                       "gain_pct": round(100 * (cased - lower) / max(lower, 1), 1)},
                "graph_nodes": g.number_of_nodes(), "graph_edges": g.number_of_edges(),
                "edges_with_typed_relations": rel_edges, "top_relation_types": roles.most_common(10),
                "top_entities": mapper.top_entities(15).to_dict("records"), "connection_examples": examples}
    viz.network(g, "Knowledge graph – 40 most mentioned entities (English news)", path=FIG / "04_knowledge_graph.png")

    # ---------------------------------------------------------------- topics
    en_all = corpus[corpus["language"] == "en"]
    docs = en_all["full_text"].sample(min(len(en_all), 2500 if fast else 5000), random_state=0).tolist()
    sel = pd.concat([select_n_topics(docs, (8, 12, 16) if fast else (8, 10, 12, 15, 20), "nmf"),
                     select_n_topics(docs, (8, 12) if fast else (8, 12, 16), "lda")], ignore_index=True)
    sel.to_csv(RES / "topic_model_selection.csv", index=False)
    tm = bot.topics
    table = tm.topics_table(10)
    table.to_csv(RES / "topics_final.csv", index=False)
    trends = tm.track_topic_trends(bot._topic_dates, freq="Q").dropna()
    counts = pd.Series(pd.to_datetime(pd.Series(bot._topic_dates), utc=True).dt.tz_convert(None).dt.to_period("Q").astype(str)).value_counts()
    trends = trends[trends["period"].isin(counts[counts >= 100].index)]
    viz.topic_trends(trends, "Topic share per quarter (English news, NMF)", path=FIG / "05_topic_trends.png")
    tm.visualize_topics().write_html(FIG / "05_intertopic_map.html", include_plotlyjs="cdn")
    en_cat = en_all.set_index("doc_id").loc[bot._topic_doc_ids, "category"]
    M["topics"] = {"selection": sel.to_dict("records"), "final": {"method": "nmf", "k": tm.n_topics,
                   "npmi": round(tm.coherence(), 4), "diversity": round(tm.diversity(), 4),
                   "purity_vs_category": round(tm.cluster_purity(en_cat.values), 4)},
                   "topics": table.to_dict("records"),
                   "emerging": TopicModeler.emerging_topics(trends).to_dict("records")}

    # ------------------------------------------------------ semantic search
    eng = bot.search
    M["semantic_search"] = {"backend": eng.embedder.backend,
                            "retrieval_precision_at_10": retrieval_eval(eng, k=10, n_queries=150 if fast else 300),
                            "queries": {}}
    for q in ["technology companies facing regulation", "central bank interest rates inflation",
              "football transfer window", "election campaign results"]:
        r = eng.semantic_search(q, top_k=3)
        M["semantic_search"]["queries"][q] = r[["title", "language", "category", "score"]].to_dict("records")
    M["semantic_search"]["expansion_examples"] = {q: eng.expand_query(q) for q in ["ai regulation", "economy", "election"]}

    # --------------------------------------------------------- summarization
    summ = bot.summarizer
    rows = []
    for _, r in b_te.sample(120 if fast else 300, random_state=0).iterrows():
        ents = list({e["canonical"] for e in mapper.extract_entities(r["text"][:3000]) if e["label"] in ("PERSON", "ORG", "GPE")})
        for method in ("lead", "textrank"):
            s = summ.summarize_article(r["text"], "balanced", method)["summary"]
            q = summ.assess_summary_quality(r["text"], s, reference=None, entities=ents or None)
            q["method"] = method
            rows.append(q)
        h = summ.generate_headlines(r["text"])["informative"]
        hq = summ.assess_summary_quality(r["text"], h, reference=r["title"])
        rows.append({"method": "headline_vs_title", **hq})
    sq = pd.DataFrame(rows)
    sq.to_csv(RES / "summary_quality.csv", index=False)
    M["summarization"] = sq.groupby("method").mean(numeric_only=True).round(4).to_dict(orient="index")

    # ------------------------------------------------------ language detection
    det = bot.langid
    used = set(getattr(det, 'train_doc_ids', []))
    rest = webz[~webz["doc_id"].isin(used)]
    test = pd.concat([g.sample(min(len(g), 150), random_state=99) for _, g in rest.groupby("language")])
    M["language_detection"] = det.evaluate(test["text"].tolist(), test["language"].tolist())
    mixed = "The meeting was postponed until Friday. El presidente dijo que la economía crecerá. Die Börse reagierte positiv."
    M["language_detection"]["mixed_example"] = det.detect_segments(mixed)

    # ------------------------------------------------------- cross-lingual
    cla = CrossLingualAnalyzer(corpus[corpus["source"] == "webz"])
    cov = cla.coverage_by_language()
    sent_lang = cla.sentiment_by_language()
    overlap = cla.entity_overlap(50)
    viz.stacked_share(cov, "Category share by language (Webz)", FIG / "06_coverage_by_language.png")
    viz.heatmap(sent_lang.drop(columns=["n"]), "Share of positive articles (Webz labels)", FIG / "06_sentiment_by_language.png",
                vmin=0, vmax=0.8)
    viz.heatmap(overlap, "Top-50 entity overlap between languages (Jaccard)", FIG / "06_entity_overlap.png", cmap="Purples",
                vmin=0, vmax=1)
    persp = {e: cla.perspective_on(e, min_articles=5).to_dict("records") for e in ["Donald Trump", "China", "United States", "Ukraine"]}
    links = cla.link_stories(min_shared=3, window_days=7, max_links=15)
    links.to_csv(RES / "cross_lingual_story_links.csv", index=False)
    M["cross_lingual"] = {"coverage": cov.to_dict(orient="index"), "positive_share": sent_lang.to_dict(orient="index"),
                          "entity_overlap": overlap.to_dict(orient="index"), "perspectives": persp,
                          "top_entities_by_language": {k: v[:8] for k, v in cla.entities_by_language(8).items()},
                          "story_links_found": len(links), "story_link_examples": links.head(6).to_dict("records"),
                          "country_focus": cla.country_focus().to_dict("records")}

    # ---------------------------------------------------------- conversation
    ic = IntentClassifier().train()
    ev = ic.evaluate()
    X, y = generate_training_data()
    transcript = []
    bot.context = type(bot.context)()
    for q in ["hi", "Show me positive tech news from this year", "what about last year?", "only negative ones",
              "summarize them", "Compare Apple and Google", "How are Donald Trump and China connected?",
              "Who is Elon Musk?", "What topics are trending in business?",
              "How has sentiment about China changed over time?", "Compare Spanish and German politics news",
              "Classify this article: Shares in Tesla fell 5% after the earnings call disappointed investors.",
              "give me an overview"]:
        t0 = time.time()
        res = bot.process_query(q)
        transcript.append({"user": q, "bot": res["response"], "intents": [p["intent"] for p in res["parsed"]],
                           "ms": round(1000 * (time.time() - t0))})
    save_json(transcript, RES / "chat_transcript.json")
    M["conversation"] = {"training_examples": len(X), "intents": len(set(y)),
                         "held_out_accuracy": ev["accuracy"], "held_out_macro_f1": ev["macro_f1"],
                         "held_out_n": ev["n"], "errors": ev["errors"],
                         "median_response_ms": float(np.median([t["ms"] for t in transcript]))}

    # ------------------------------------------------------------ latency
    from src.data_processing.data_loader import sample_articles
    t0 = time.time()
    for a in sample_articles():
        bot.analyze_article(a["text"], a["title"])
    M["latency_ms_per_article_full_analysis"] = round(1000 * (time.time() - t0) / 4)
    M["total_runtime_min"] = round((time.time() - t_start) / 60, 1)
    save_json(M, RES / "metrics.json")
    bot.save()
    print(f"Done in {M['total_runtime_min']} min – results in {RES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse cached corpus/models instead of retraining")
    args = ap.parse_args()
    main(args.fast, args.reuse)
