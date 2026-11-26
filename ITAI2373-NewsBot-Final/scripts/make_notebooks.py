"""Generate the seven tutorial notebooks in ``notebooks/`` (run once; then execute them)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = '''# ⚙️ Setup – works in Google Colab (clones the repo + installs) and locally
import os, sys, subprocess
from pathlib import Path
IN_COLAB = "google.colab" in sys.modules
ROOT = Path("/content/ITAI2373-NewsBot-Final/ITAI2373-NewsBot-Final") if IN_COLAB else Path.cwd().resolve().parent
if IN_COLAB and not ROOT.exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/Haryapandhita/ITAI2373-NewsBot-Final.git",
                    "/content/ITAI2373-NewsBot-Final"], check=True)
sys.path.insert(0, str(ROOT))
from scripts.colab_setup import setup
ROOT = setup()

import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 120)
# Transformer models (summarisation, translation, multilingual embeddings) are used when
# available (Colab: Runtime -> Change runtime type -> T4 GPU recommended); otherwise
# NewsBot falls back to classical methods automatically.
USE_TRANSFORMERS = os.environ.get("NEWSBOT_USE_TRANSFORMERS", "1") == "1"
print("Transformers enabled:", USE_TRANSFORMERS)'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def header(num, title, goal, modules):
    return md(f"""# NewsBot 2.0 · Notebook {num:02d} – {title}
**ITAI 2373 Final Project · Mourinho Hernanto**

**Goal:** {goal}

**Course modules:** {modules}

> Runs top-to-bottom in Google Colab (first cell installs everything). All logic lives in `src/` –
> this notebook demonstrates and evaluates it.""")


NOTEBOOKS = {}

# --------------------------------------------------------------------------- 01
NOTEBOOKS["01_Data_Exploration.ipynb"] = [
    header(1, "Data Exploration", "load both corpora, validate data quality and understand what the system works with.",
           "1 (NLP applications), 2 (preprocessing), 3 (features)"),
    code(SETUP),
    md("""## 1. Load the data
Two public datasets are downloaded on first use (not stored in the repo):
* **BBC News** – 2,225 English articles (2004–05) in 5 categories, **original casing and headlines**.
  The midterm used the lower-cased Kaggle copy, which broke NER; this version fixes that.
* **Webz.io free news** – 20 weekly snapshots, **6 languages, dated Dec 2023 – Nov 2026**, with category and sentiment labels.
  This is what makes topic/sentiment *evolution* and the *multilingual* module possible."""),
    code('''from src.data_processing.data_loader import load_all
data = load_all()
bbc, webz = data["bbc"], data["webz"]
print("BBC :", bbc.shape, "| Webz:", webz.shape)
bbc.head(3)'''),
    code('''webz[["title", "language", "category", "published", "site", "webz_sentiment"]].sample(6, random_state=1)'''),
    md("## 2. Data-quality validation"),
    code('''from src.data_processing.data_validator import validate_corpus
for name, df in data.items():
    print(f"== {name}"); print(validate_corpus(df)); print(validate_corpus(df).stats); print()'''),
    md("## 3. Composition"),
    code('''fig, axes = plt.subplots(1, 3, figsize=(16, 4))
bbc["category"].value_counts().plot.bar(ax=axes[0], title="BBC – articles per category", color="#2563eb")
webz["language"].value_counts().plot.bar(ax=axes[1], title="Webz – articles per language", color="#f97316")
pd.crosstab(webz["language"], webz["category"]).plot.bar(stacked=True, ax=axes[2], title="Webz – category by language")
for ax in axes: ax.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.show()'''),
    code('''q = webz["published"].dt.tz_convert(None).dt.to_period("Q").astype(str)
pd.crosstab(q, webz["category"]).plot.bar(stacked=True, figsize=(11, 4), title="Webz articles per quarter")
plt.xticks(rotation=45); plt.show()'''),
    code('''lengths = pd.DataFrame({"BBC": bbc["text"].str.split().str.len(), "Webz": webz["text"].str.split().str.len()})
print(lengths.describe().round(0))
lengths.clip(upper=1500).plot.hist(bins=50, alpha=0.6, figsize=(9, 3.5), title="Words per article (clipped at 1500)"); plt.show()'''),
    md("""## 4. Preprocessing – two versions of every text
* `clean()` keeps casing and punctuation → used by spaCy, VADER, summarisation.
* `normalize()` lower-cases, repairs contractions, turns money into tokens and removes language-specific stop words → used by TF-IDF and topic models."""),
    code('''from src.data_processing.text_preprocessor import TextPreprocessor
pre = TextPreprocessor()
example = bbc.loc[0, "text"][:400]
print("CLEAN     :", pre.clean(example)[:300]); print()
print("NORMALIZED:", pre.normalize(example)[:300]); print()
es = webz[webz.language == "es"].iloc[0]["text"][:300]
print("SPANISH   :", TextPreprocessor(language="es").normalize(es)[:200])'''),
    code('''from src.data_processing.feature_extractor import describe_features
describe_features(bbc)'''),
    md("""## 5. Language-label check
The Webz `language` field is produced automatically and is sometimes wrong (e.g. a Vietnamese page labelled German).
NewsBot's own language detector (notebook 05) double-checks every label during `NewsBot2.build()` and drops
articles where a confident detection disagrees."""),
    code('''from src.multilingual.language_detector import LanguageDetector
X, y = LanguageDetector.make_training_snippets(webz.groupby("language").sample(300, random_state=0)["text"],
                                               webz.groupby("language").sample(300, random_state=0)["language"])
det = LanguageDetector().train(X, y)
pred = det.detect_batch(webz["text"].str[:600])
mismatch = webz[np.array(pred) != webz["language"].values]
print(f"{len(mismatch)} of {len(webz)} Webz articles have a language label that disagrees with the detector")
mismatch[["title", "language"]].assign(detected=np.array(pred)[np.array(pred) != webz["language"].values]).head(8)'''),
    md("""## Key takeaways
* 2,092 unique BBC articles (133 exact duplicates removed) and ~13k Webz articles in en/es/de/pt/fr/it.
* Webz is English-heavy (≈65%) and politics/sport-heavy in Spanish – the multilingual analysis must normalise by language.
* Dates cluster in weekly snapshots, so trends are reported **per quarter**.
* BBC text now keeps its original casing → much better entity recognition (measured in notebook 02)."""),
]

# --------------------------------------------------------------------------- 02
NOTEBOOKS["02_Advanced_Classification.ipynb"] = [
    header(2, "Advanced Classification, Sentiment & Entities",
           "build the Module A content-analysis engine: ensemble classifier with confidence and explanations, "
           "sentence-level sentiment with emotions, NER with relation extraction and a knowledge graph.",
           "4–8 (POS/syntax, sentiment, classification, NER)"),
    code(SETUP),
    code('''from src.data_processing.data_loader import load_all
from sklearn.model_selection import train_test_split
data = load_all(); bbc, webz = data["bbc"], data["webz"]
b = bbc.assign(full=bbc.title + ". " + bbc.text)
b_tr, b_te = train_test_split(b, test_size=0.2, stratify=b.category, random_state=42)
print(len(b_tr), "train /", len(b_te), "test")'''),
    md("## 1. Baselines vs. the NewsBot 2.0 ensemble (same split, no leakage)"),
    code('''from src.analysis.classifier import NewsClassifier, compare_baselines
base = compare_baselines(b_tr.full, b_tr.category, b_te.full, b_te.category)
clf = NewsClassifier().train(b_tr.full, b_tr.category)
ev = clf.evaluate(b_te.full, b_te.category)
base.loc[len(base)] = ["NewsBot 2.0 Ensemble (SVM+LR+NB, word+char)", ev["accuracy"], ev["macro_f1"], clf.train_seconds]
base.loc[len(base)] = ["Midterm Linear SVM (reported)", 0.9617, 0.9582, None]
base.sort_values("macro_f1", ascending=False)'''),
    code('''from src.utils import visualization as viz
from src.utils.evaluation import per_class_table
viz.confusion_matrix_plot(ev["confusion_matrix"], ev["labels"], "Ensemble – BBC test"); plt.show()
per_class_table(ev["report"])'''),
    code('''print("5-fold CV macro-F1 on the training split:", clf.cross_validate(b_tr.full, b_tr.category, folds=5))'''),
    md("""## 2. Confidence you can trust
Calibrated probabilities: when the model says 90%, it should be right ~90% of the time. Articles with low confidence or a small
margin between the top two classes are flagged `needs_review` for a human editor."""),
    code('''clf.calibration_table(b_te.full, b_te.category)'''),
    code('''p = clf.predict_with_confidence("Manchester United's shares jumped after the club announced record TV revenue.")
p.to_dict()'''),
    md("## 3. Explaining a decision"),
    code('''exp = clf.explain_prediction("The new smartphone chip doubles battery life, analysts expect strong sales.")
print(exp["prediction"]["label"], exp["prediction"]["confidence"])
print("Key terms:", exp["key_terms"])
pd.DataFrame(exp["similar_training_articles"])'''),
    md("""## 4. Domain shift: does a 2004 model understand 2026 news?
We test the BBC-trained model on **English Webz articles from 2023–2026** (4 shared categories). Vocabulary such as
"AI", "crypto" or "TikTok" did not exist in 2004 – accuracy drops sharply. NewsBot 2.0 therefore trains on BBC **plus**
a balanced sample of recent Webz articles (test articles excluded)."""),
    code('''from src.newsbot import training_frame
w = webz[webz.language == "en"].assign(full=lambda d: d.title + ". " + d.text)
_, w_te = train_test_split(w, test_size=0.3, stratify=w.category, random_state=42)
old = clf.evaluate(w_te.full, w_te.category)
train2 = training_frame(bbc, webz)
clf2 = NewsClassifier().train(train2.full, train2.category)
new = clf2.evaluate(w_te.full, w_te.category)
bbc_new = clf2.evaluate(b_te.full, b_te.category)
pd.DataFrame({"BBC-only model": [old["accuracy"], old["macro_f1"], ev["accuracy"]],
              "NewsBot 2.0 model": [new["accuracy"], new["macro_f1"], bbc_new["accuracy"]]},
             index=["Webz 2023-26 accuracy", "Webz 2023-26 macro-F1", "BBC test accuracy"])'''),
    code('''fig = viz.confusion_matrix_plot(new["confusion_matrix"], new["labels"], "NewsBot 2.0 model on 2023–26 news"); plt.show()'''),
    md("""## 5. Multi-label output
A story can belong to two sections. Secondary labels are returned when their probability ≥ 0.20."""),
    code('''for t in ["Chelsea football club reported a £90m loss as wages and transfer fees rose, the club's accounts show.",
          "The government will regulate artificial intelligence companies under a new law passed by parliament."]:
    p = clf2.predict_with_confidence(t); print(p.label, p.confidence, "secondary:", p.secondary_labels, "| review:", p.needs_review)'''),
    md("""## 6. Optional – multilingual zero-shot classification with sentence embeddings
With the multilingual MiniLM encoder, a classifier trained on **English** articles can label **Spanish/German/French**
articles directly (shared embedding space). Runs when `sentence-transformers` is available (Colab)."""),
    code('''from src.language_models.embeddings import SemanticEmbedder
from src.analysis.classifier import EmbeddingClassifier
emb = SemanticEmbedder(use_transformer=USE_TRANSFORMERS)
if emb.backend == "sentence-transformers":
    ec = EmbeddingClassifier(emb).train(train2.full.tolist(), train2.category.tolist())
    rows = []
    for lang in ["es", "de", "fr", "it", "pt"]:
        s = webz[webz.language == lang].sample(200, random_state=0)
        tfidf_acc = clf2.evaluate((s.title + ". " + s.text).tolist(), s.category.tolist())["accuracy"]
        emb_acc = float(np.mean(np.array(ec.predict((s.title + ". " + s.text).tolist())) == s.category.values))
        rows.append({"language": lang, "TF-IDF model": tfidf_acc, "multilingual embeddings": round(emb_acc, 4)})
    display(pd.DataFrame(rows))
else:
    print("Sentence-transformers not available here – run in Colab to see zero-shot multilingual classification.")'''),
    md("""## 7. Sentiment: sentence-level VADER + news lexicon + emotions
The midterm found whole-article VADER saturates near ±1. NewsBot averages per sentence and corrects news words
("party", "penalty", "shares" are neutral). We evaluate against Webz's sentiment labels (English, positive vs. negative)."""),
    code('''from src.analysis.sentiment_analyzer import SentimentAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sa, plain = SentimentAnalyzer(), SentimentIntensityAnalyzer()
ev_s = webz[(webz.language == "en") & webz.webz_sentiment.isin(["positive", "negative"])].sample(1500, random_state=0)
texts = (ev_s.title + ". " + ev_s.text).tolist()
whole = np.array([plain.polarity_scores(t[:5000])["compound"] for t in texts])
sent_avg = np.array([np.mean(sa.sentence_scores(t) or [0]) for t in texts])
acc = lambda s: float((np.where(s >= 0, "positive", "negative") == ev_s.webz_sentiment.values).mean())
print(f"whole-text VADER accuracy      : {acc(whole):.3f}  (saturated |score|>0.9: {np.mean(np.abs(whole) > 0.9):.0%})")
print(f"sentence-average + news lexicon: {acc(sent_avg):.3f}")'''),
    code('''sa.analyze("The striker scored a stunning winner. Fans celebrated, but the coach was angry about the referee.")'''),
    md("## 8. Named entities, relations and the knowledge graph"),
    code('''from src.analysis.ner_extractor import EntityRelationshipMapper
mapper = EntityRelationshipMapper()
text = "Microsoft boss Bill Gates met Tony Blair in London. Tim Cook, chief executive of Apple, visited Berlin on Monday."
display(pd.DataFrame(mapper.extract_entities(text))[["text", "canonical", "label"]])
pd.DataFrame(mapper.extract_relationships(text))[["source", "relation", "target"]]'''),
    md("**Midterm limitation fixed – casing matters for NER** (150 BBC test articles, PERSON/ORG/GPE entities):"),
    code('''s = b_te.sample(150, random_state=0)["text"]
count = lambda texts: sum(len([e for e in mapper.extract_entities(t) if e["label"] in ("PERSON", "ORG", "GPE")]) for t in texts)
cased, lower = count(s), count(s.str.lower())
print(f"original casing: {cased} entities | lower-cased (midterm data): {lower} entities | +{100*(cased-lower)/lower:.0f}%")'''),
    code('''g = mapper.build_knowledge_graph(b.full.str[:3000], b.category)
display(mapper.top_entities(10))
viz.network(g, "BBC knowledge graph – 40 most mentioned entities"); plt.show()'''),
    code('''for a, c in [("Blair", "Brown"), ("Microsoft", "Google"), ("Arsenal", "Chelsea")]:
    print(a, "↔", c, "→", mapper.find_entity_connections(a, c))
mapper.entity_profile("Tony Blair")'''),
    md("""## Key takeaways
* On clean BBC data every strong model reaches 97–98% – differences of <1 point are within noise on 419 test articles.
  The jump over the midterm (96.2%) mostly comes from **better data** (original casing + headlines). We keep the ensemble
  because it gives **calibrated probabilities** (a plain SVM has none), which power the review flag and multi-label output.
* A 2004-only model loses a lot of accuracy on 2023–26 news; adding recent articles recovers it → real-world robustness.
* Sentence-averaged sentiment agrees with the external labels slightly more often than whole-article VADER (≈+1.5 points)
  and, more importantly, no longer saturates at ±1 – scores become comparable between articles.
* Original casing multiplies the number of recognised entities; relation patterns (ROLE_AT, verbs) turn co-mentions into a knowledge graph."""),
]

# --------------------------------------------------------------------------- 03
NOTEBOOKS["03_Topic_Modeling.ipynb"] = [
    header(3, "Topic Modeling & Trends", "discover hidden topics with LDA and NMF, choose the number of topics with coherence, "
           "and track how topics and sentiment evolve over time.", "9 (topic modelling), 6 (sentiment)"),
    code(SETUP),
    code('''from src.data_processing.data_loader import load_all
from src.analysis.topic_modeler import TopicModeler, select_n_topics
data = load_all(); webz, bbc = data["webz"], data["bbc"]
en = pd.concat([bbc, webz[webz.language == "en"]], ignore_index=True)
en["full"] = en.title + ". " + en.text
docs = en.full.sample(4000, random_state=0).tolist()
print(len(en), "English articles; model selection on a 4,000 sample")'''),
    md("""## 1. How many topics? LDA vs NMF
**NPMI coherence** (implemented in `topic_modeler.py`) measures whether a topic's top words really occur together (−1…1).
**Diversity** = share of unique words across topics (penalises repeated topics)."""),
    code('''sel = pd.concat([select_n_topics(docs, (8, 10, 12, 15), "nmf"), select_n_topics(docs, (8, 12, 16), "lda")])
sel.sort_values("npmi", ascending=False)'''),
    code('''fig, ax = plt.subplots(figsize=(7, 3.5))
for m, g in sel.groupby("method"): ax.plot(g.n_topics, g.npmi, marker="o", label=m.upper())
ax.set_xlabel("number of topics"); ax.set_ylabel("NPMI coherence"); ax.legend(); ax.set_title("Topic coherence"); plt.show()'''),
    md("## 2. Final model (NMF, k = 15)"),
    code('''tm = TopicModeler(15, "nmf")
W = tm.fit_transform(en.full)
print(f"NPMI {tm.coherence():.3f} | diversity {tm.diversity():.3f} | purity vs. category {tm.cluster_purity(en.category):.3f}")
tm.topics_table(10)'''),
    code('''fig = tm.visualize_topics()
fig.show()   # interactive intertopic distance map (bubble size = prevalence)'''),
    code('''lda = TopicModeler(12, "lda"); lda.fit_transform(en.full.sample(4000, random_state=1))
lda.topics_table(8)[["label", "top_words", "prevalence", "npmi"]]'''),
    md("## 3. Topics of a new article & content clustering"),
    code('''tm.get_article_topics("Nvidia shares hit a record as demand for artificial intelligence chips keeps rising, "
                      "while regulators in Brussels examine competition in the cloud market.")'''),
    code('''pd.crosstab(pd.Series(tm.labels_)[tm.dominant_topics()].values, en.category, normalize="index").round(2)'''),
    md("## 4. Topic evolution 2023 → 2026 (dated Webz articles)"),
    code('''mask = en.published.notna().values
tm_dated = TopicModeler(15, "nmf"); tm_dated.fit_transform(en.loc[mask, "full"])
trends = tm_dated.track_topic_trends(en.loc[mask, "published"], freq="Q")
counts = en.loc[mask, "published"].dt.tz_convert(None).dt.to_period("Q").astype(str).value_counts()
trends = trends[trends.period.isin(counts[counts >= 100].index)]
from src.utils import visualization as viz
viz.topic_trends(trends, "Topic share per quarter"); plt.show()
TopicModeler.emerging_topics(trends)'''),
    md("## 5. Sentiment evolution and anomaly detection"),
    code('''from src.analysis.sentiment_analyzer import SentimentAnalyzer
sa = SentimentAnalyzer()
w = webz.copy(); w["sentiment"] = sa.batch_scores((w.title + ". " + w.text).str[:6000])["sentiment"].values
tl = SentimentAnalyzer.track_over_time(w, group_col="category", freq="Q"); tl = tl[tl.n_articles >= 30]
an = SentimentAnalyzer.detect_anomalies(tl, z=1.5)
viz.timeline(an, "period", "mean_sentiment", "category", "Sentiment per quarter (all languages)", "sentence-avg VADER", anomalies=an); plt.show()
an[an.anomaly]'''),
    md("""**Caveat:** each Webz snapshot was selected by Webz with a sentiment filter (e.g. *Politics_negative*), so part of the
quarter-to-quarter swing reflects snapshot selection, not public mood. We report it as a *capability demo* and discuss this
limitation in the technical report."""),
    md("""## Key takeaways
* NMF gives roughly twice the coherence of LDA on short, noisy web news, with higher diversity.
* NMF coherence is fairly flat for k = 10–20 and the best k changes with the sample (here k = 12; on the 5,000-article
  sample in `scripts/run_experiments.py` k = 20 > 10 > 15 > 12). The system uses k = 15 as a balance of coherence,
  diversity and granularity – the choice is not sharp, which we state openly.
* Topics align with but go beyond the 5 categories (AI, markets, courts, health, US politics…).
* Topic shares and sentiment can be tracked per quarter; z-score anomalies point to unusual periods worth investigating."""),
]

# --------------------------------------------------------------------------- 04
NOTEBOOKS["04_Language_Models.ipynb"] = [
    header(4, "Language Understanding & Generation", "summarise articles (extractive + abstractive), generate headlines, "
           "search by meaning, expand queries and enrich articles with context.", "10 (neural LMs), 11 (generation), 12 (NLU)"),
    code(SETUP),
    code('''from src.data_processing.data_loader import load_all
data = load_all(); bbc, webz = data["bbc"], data["webz"]
art = bbc[bbc.category == "business"].iloc[3]
print(art.title); print(art.text[:800], "...")'''),
    md("## 1. Summarisation – Lead-3 vs TextRank vs DistilBART"),
    code('''from src.language_models.summarizer import IntelligentSummarizer
summ = IntelligentSummarizer(use_transformer=USE_TRANSFORMERS)
for method in ["lead", "textrank", "abstractive"]:
    r = summ.summarize_article(art.text, "balanced", method)
    print(f"--- {method} (used: {r['method']}, compression {r['compression']:.0%})"); print(r["summary"]); print()'''),
    code('''for kind in ["brief", "detailed"]:
    print(kind.upper(), "→", summ.summarize_article(art.text, kind)["summary"]); print()'''),
    md("## 2. Headline generation – compared with the real BBC headline (ROUGE)"),
    code('''rows = []
for _, r in bbc.sample(100, random_state=0).iterrows():
    h = summ.generate_headlines(r.text)
    q = summ.assess_summary_quality(r.text, h["informative"], reference=r.title)
    rows.append({"real": r.title, "generated": h["informative"], **{k: q[k] for k in ("rouge1", "rougeL")}})
hd = pd.DataFrame(rows); print(hd[["rouge1", "rougeL"]].mean().round(3)); hd.head(6)'''),
    md("## 3. Summary quality without references"),
    code('''from src.analysis.ner_extractor import EntityRelationshipMapper
mapper = EntityRelationshipMapper()
rows = []
for _, r in bbc.sample(80, random_state=1).iterrows():
    ents = list({e["canonical"] for e in mapper.extract_entities(r.text) if e["label"] in ("PERSON", "ORG", "GPE")})
    for m in ("lead", "textrank"):
        s = summ.summarize_article(r.text, "balanced", m)["summary"]
        rows.append({"method": m, **summ.assess_summary_quality(r.text, s, entities=ents or None)})
pd.DataFrame(rows).groupby("method").mean(numeric_only=True).round(3)'''),
    md("## 4. Multi-document summary (several articles about one story)"),
    code('''related = bbc[bbc.text.str.contains("Yukos")].text.tolist()
print(len(related), "articles mention Yukos")
print(summ.summarize_multiple_articles(related, n_sentences=5, focus_topic="Yukos oil company Russia"))'''),
    md("""## 5. Semantic search
Hybrid score = 0.6 × dense similarity + 0.4 × TF-IDF match. With `sentence-transformers` the dense part is the
**multilingual** MiniLM model, so an English query also retrieves Spanish/German articles; offline it uses LSA."""),
    code('''from src.language_models.embeddings import SemanticEmbedder, SemanticSearchEngine, retrieval_eval
corpus = pd.concat([bbc, webz.sample(6000, random_state=0)], ignore_index=True)
engine = SemanticSearchEngine(SemanticEmbedder(use_transformer=USE_TRANSFORMERS)).encode_documents(corpus)
print("backend:", engine.embedder.backend)
engine.semantic_search("technology companies facing regulation", top_k=5)[["title", "language", "category", "score"]]'''),
    code('''engine.semantic_search("central bank raises interest rates", top_k=5,
                       filters={"language": ["es", "de", "fr"]})[["title", "language", "score"]]'''),
    code('''print("Query expansion:", engine.expand_query("ai regulation economy"))
print(retrieval_eval(engine, k=10, n_queries=200))'''),
    code('''q = bbc.iloc[100]; print("QUERY ARTICLE:", q.title)
engine.find_similar_articles(q.title + ". " + q.text, top_k=5, exclude_doc_id=q.doc_id)[["title", "category", "similarity"]]'''),
    code('''engine.cluster_similar_content(n_clusters=8)'''),
    md("## 6. Content enhancement – grounded context for an article"),
    code('''from src.language_models.generator import ContentEnhancer
mapper.build_knowledge_graph(bbc.title + ". " + bbc.text.str[:2000], bbc.category)
enh = ContentEnhancer(mapper, engine)
target = bbc[bbc.title.str.contains("Blair")].iloc[0]
ents = mapper.extract_entities(target.text)
print(target.title)
enh.enhance_article(target.text, ents)'''),
    code('''enh.cross_reference_facts(target.text)'''),
    md("""## Key takeaways
* TextRank+MMR produces non-redundant summaries; DistilBART (Colab) writes fluent abstractive summaries – both behind one API.
* Generated headlines overlap meaningfully with the real ones (ROUGE), with zero training.
* Hybrid semantic search finds relevant articles by meaning, supports filters and expands queries; with multilingual
  embeddings it searches across languages.
* Enhancements are **grounded** in the system's own knowledge graph and corpus – no hallucinated facts."""),
]

# --------------------------------------------------------------------------- 05
NOTEBOOKS["05_Multilingual_Analysis.ipynb"] = [
    header(5, "Multilingual Intelligence", "detect languages with our own model, translate with MarianMT and assess quality, "
           "and compare how six languages cover the news (tone, entities, cultural context).", "11 (MT & multilingual), 8 (NER)"),
    code(SETUP),
    code('''from src.data_processing.data_loader import load_all
data = load_all(); webz = data["webz"]
webz["language"].value_counts()'''),
    md("""## 1. Language detection – our own character n-gram model
TF-IDF over 1–4 character n-grams + logistic regression, trained on sentence snippets from 300 articles per language.
Evaluated on **unseen** articles and on very short inputs; `langdetect` is the baseline."""),
    code('''from src.multilingual.language_detector import LanguageDetector
train = webz.groupby("language").sample(300, random_state=42)
X, y = LanguageDetector.make_training_snippets(train.text, train.language)
det = LanguageDetector().train(X, y)
test = webz[~webz.doc_id.isin(train.doc_id)].groupby("language").sample(150, random_state=7)
pd.DataFrame(det.evaluate(test.text.tolist(), test.language.tolist())).T'''),
    code('''for t in ["Breaking news", "Der Bundestag stimmt heute ab", "La Liga vuelve este fin de semana", "Il governo"]:
    print(f"{t!r:40} → {det.detect_language(t)}")'''),
    code('''det.detect_segments("The meeting was postponed until Friday. El presidente dijo que la economía crecerá. Die Börse reagierte positiv.")'''),
    md("## 2. Translation with MarianMT + reference-free quality estimate (round-trip chrF)"),
    code('''from src.multilingual.translator import Translator
tr = Translator(use_transformer=USE_TRANSFORMERS)
samples = webz[webz.language != "en"].groupby("language").sample(1, random_state=3)
rows = []
for _, r in samples.iterrows():
    out = tr.translate_text(r.title, r.language)
    q = tr.assess_quality(r.title, r.language, out["text"]) if out["translated"] else {}
    rows.append({"lang": r.language, "original": r.title, "english": out["text"], "engine": out["engine"],
                 "round_trip_chrF": q.get("round_trip_chrf")})
pd.DataFrame(rows)'''),
    md("## 3. Cross-lingual comparison"),
    code('''from src.analysis.ner_extractor import EntityRelationshipMapper
from src.multilingual.cross_lingual_analyzer import CrossLingualAnalyzer
mapper = EntityRelationshipMapper()
w = webz.copy()
w["entities"] = mapper.corpus_entities((w.title + ". " + w.text).tolist(), w.language.tolist())
cla = CrossLingualAnalyzer(w)
from src.utils import visualization as viz
viz.stacked_share(cla.coverage_by_language(), "Category share by language"); plt.show()'''),
    code('''viz.heatmap(cla.sentiment_by_language().drop(columns="n"), "Share of positive articles (Webz labels)", vmin=0, vmax=0.8); plt.show()'''),
    code('''pd.DataFrame({l: [f"{e} ({c})" for e, c in v] for l, v in cla.entities_by_language(10).items()})'''),
    code('''viz.heatmap(cla.entity_overlap(50), "Top-50 entity overlap (Jaccard)", cmap="Purples", vmin=0, vmax=1); plt.show()'''),
    md("## 4. Regional perspectives on the same entity"),
    code('''for e in ["Donald Trump", "China", "Ukraine"]:
    print("==", e); display(cla.perspective_on(e))'''),
    md("## 5. The same story in different languages (entity overlap + 7-day window)"),
    code('''cla.link_stories(min_shared=3, window_days=7, max_links=10)'''),
    md("## 6. Cultural context"),
    code('''display(cla.country_focus())
for lang in ["es", "de", "fr", "pt"]:
    t = webz[webz.language == lang].text
    hits = [CrossLingualAnalyzer.extract_cultural_context(x, lang) for x in t.head(300)]
    refs = pd.Series([k for h in hits for k in h["references"]]).value_counts().head(5)
    print(lang, dict(refs))'''),
    md("""## Key takeaways
* Our own language detector is near-perfect on articles and stays strong on very short snippets, matching or beating `langdetect`.
* MarianMT (Colab) translates titles/articles locally; round-trip chrF gives a quality signal without references.
* Languages differ clearly in **what** they cover (Spanish: politics-heavy, German: tech-heavy) and **how** (share of
  positive articles, entity focus). Multilingual aliases (*Estados Unidos*, *Alemania* → canonical names) make entities comparable.
* Cultural references (Bundestag, Moncloa, Ibovespa…) are detected and explained for international readers."""),
]

# --------------------------------------------------------------------------- 06
NOTEBOOKS["06_Conversational_Interface.ipynb"] = [
    header(6, "Conversational Interface", "understand natural-language questions (intent + slots), keep conversation context "
           "for follow-ups, and answer from the analysed news.", "12 (conversational AI), 7 (intent classification), 8 (entities)"),
    code(SETUP),
    md("## 1. Intent classification"),
    code('''from src.conversation.intent_classifier import IntentClassifier, generate_training_data, HELD_OUT_QUERIES
X, y = generate_training_data()
print(len(X), "generated training queries over", len(set(y)), "intents"); print(pd.Series(y).value_counts().to_dict())
ic = IntentClassifier().train(X, y)
ev = ic.evaluate()
print(f"Held-out hand-written queries: accuracy {ev['accuracy']:.3f}, macro-F1 {ev['macro_f1']:.3f} (n={ev['n']})")
ev["errors"]'''),
    md("## 2. Slot extraction (category, sentiment, time, language, entities, format …)"),
    code('''from src.conversation.query_processor import QueryProcessor
qp = QueryProcessor(ic, reference_date=pd.Timestamp("2026-11-13", tz="UTC"))
for q in ["Show me positive tech news from this week", "Compare Spanish and German coverage of the economy",
          "Top 3 stories about Elon Musk in Q3 2025", "Export the results as csv"]:
    p = qp.parse(q)[0]; print(f"{q}\\n   → {p.intent} ({p.confidence}) {p.slots}\\n")'''),
    md("## 3. The full chatbot (NewsBot2) – with context-aware follow-ups"),
    code('''from src.newsbot import NewsBot2
bot = NewsBot2(use_transformers=USE_TRANSFORMERS).build()
def ask(q):
    print("🧑", q); print("🤖", bot.chat(q)); print("-" * 100)'''),
    code('''for q in ["hi", "Show me positive tech news from this year", "what about last year?", "only negative ones", "summarize them"]:
    ask(q)'''),
    code('''for q in ["Compare Apple and Google", "How are Donald Trump and China connected?", "Who is Elon Musk?"]:
    ask(q)'''),
    code('''for q in ["What topics are trending in business?", "How has sentiment about China changed over time?",
          "Is coverage of the economy mostly negative?", "Compare Spanish and German politics news"]:
    ask(q)'''),
    code('''ask("show German business news and then summarize them")    # multi-part question
ask("Classify this article: Shares in Tesla fell 5% after the earnings call disappointed investors on Wall Street.")
ask("give me an overview"); ask("export the results as csv"); ask("help")'''),
    md("""## 4. Optional: polishing answers with a local LLM (Ollama)
Set `NEWSBOT_USE_LLM=1` (and run `ollama serve` with e.g. `llama3.2`) to let a local LLM rephrase the **grounded** draft.
The LLM receives only the facts NewsBot computed and is told not to add any; without Ollama the template answer is used."""),
    code('''from src.language_models.generator import OllamaClient
llm = OllamaClient()
print("Ollama available:", llm.available())
draft = bot.chat("Who is Elon Musk?")
print(llm.rewrite(draft) if llm.available() else draft)'''),
    md("""## Key takeaways
* Template-bootstrapped intent model reaches high accuracy on differently-worded held-out questions.
* Rule + NER slot filling handles categories, sentiment, relative/absolute dates, languages, entities and formats.
* Conversation memory enables follow-ups ("what about last year?", "only negative ones", "summarize them") and
  multi-part questions; answers are grounded in the analysed corpus."""),
]

# --------------------------------------------------------------------------- 07
NOTEBOOKS["07_System_Integration.ipynb"] = [
    header(7, "System Integration & Testing", "run the complete NewsBot 2.0 system end-to-end, test robustness and performance, "
           "and export reports.", "all modules"),
    code(SETUP),
    code('''from src.newsbot import NewsBot2
from src.data_processing.data_loader import sample_articles
bot = NewsBot2(use_transformers=USE_TRANSFORMERS).build()
pd.Series(bot.timings, name="seconds")'''),
    md("## 1. Comprehensive analysis of one article"),
    code('''art = sample_articles()[0]
res = bot.analyze_article(art["text"], art["title"])
for k in ["language", "classification", "sentiment", "topics", "summary", "headlines"]:
    print(f"== {k}"); print(res[k])
print("== entities:", sorted({e["canonical"] for e in res["entities"]}))
print("== relations:", res["relations"]); print("== errors:", res["errors"])'''),
    code('''res["enhancements"]'''),
    md("## 2. A non-English article (language → translation → analysis → cultural context)"),
    code('''es = sample_articles()[3]
r = bot.analyze_article(es["text"], es["title"])
{k: r.get(k) for k in ["language", "translation", "classification", "sentiment", "cultural_context", "errors"]}'''),
    md("## 3. Batch analysis"),
    code('''import time
batch_texts = bot.corpus.sample(40, random_state=5)["full_text"].tolist() + [a["text"] for a in sample_articles()]
t0 = time.time(); batch = bot.batch_analysis(batch_texts, progress=False)
print(f"{len(batch)} articles in {time.time()-t0:.1f}s ({1000*(time.time()-t0)/len(batch):.0f} ms/article)")
batch.head(10)'''),
    md("## 4. Robustness – edge cases"),
    code('''cases = {"empty": "", "very short": "Goal!", "numbers only": "123 456 789", "gibberish": "asdkj qwpoe zmxnc",
         "very long": " ".join([sample_articles()[1]["text"]] * 60), "mixed language": "Apple sube en bolsa. Shares rallied in New York."}
for name, text in cases.items():
    try:
        r = bot.analyze_article(text, enhance=False)
        c = r.get("classification", {})
        print(f"{name:15} → {c.get('label')} (review: {c.get('needs_review')}) | lang {r.get('language', {}).get('language')} "
              f"| warnings: {r.get('warnings', [])} | errors: {r['errors']}")
    except ValueError as e:
        print(f"{name:15} → rejected cleanly: {e}")'''),
    md("## 5. Insights report and export"),
    code('''report = bot.generate_insights_report()
print(report[:2500])'''),
    code('''from src.utils.export import export_results
out = bot.cfg.results_dir / "exports"
print(export_results(report, out / "insights_report.html"))
print(bot.export_article_report(res, out / "article_report.md"))
print(export_results(batch, out / "batch_analysis.csv"))'''),
    md("## 6. Automated tests"),
    code('''import subprocess, sys
print(subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True, cwd=ROOT).stdout[-1500:])'''),
    md("## 7. Results of the full experiment run (`scripts/run_experiments.py`)"),
    code('''from src.utils.evaluation import load_json
M = load_json(ROOT / "data/results/metrics.json")
summary = {
  "BBC test accuracy (ensemble)": M["classification"]["ensemble_bbc_test"]["accuracy"],
  "BBC test macro-F1 (ensemble)": M["classification"]["ensemble_bbc_test"]["macro_f1"],
  "2023-26 news accuracy, BBC-only → NewsBot 2.0": f'{M["classification"]["cross_domain_webz_en"]["bbc_only_model"]["accuracy"]} → {M["classification"]["cross_domain_webz_en"]["newsbot2_model"]["accuracy"]}',
  "Sentiment agreement with Webz labels": M["sentiment"]["accuracy_sentence_avg_news_lexicon"],
  "Topic NPMI coherence (NMF k=15)": M["topics"]["final"]["npmi"],
  "Language ID accuracy (first 20 chars)": M["language_detection"]["first_20_chars"]["accuracy"],
  "Intent accuracy (held-out)": M["conversation"]["held_out_accuracy"],
  "Search precision@10": M["semantic_search"]["retrieval_precision_at_10"]["precision_at_k"],
}
pd.Series(summary, name="value")'''),
    md("""## Conclusion
All four modules run together through one orchestrator (`NewsBot2`): any supported-language article is detected, translated
when needed, classified with calibrated confidence, scored for sentiment and emotions, mined for entities/relations, placed
in topics, summarised, enriched with grounded context and made explorable through a context-aware chatbot. Every step is
isolated so one failure never breaks the whole analysis, and heavy artefacts are cached for fast restarts."""),
]


def main():
    NB_DIR.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = nbf.v4.new_notebook()
        nb["cells"] = cells
        nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                          "language_info": {"name": "python"}, "colab": {"provenance": []}}
        nbf.write(nb, NB_DIR / name)
        print("wrote", name)


if __name__ == "__main__":
    main()
