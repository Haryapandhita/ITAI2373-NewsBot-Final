# NewsBot 2.0 – API Reference

*Generated from the source docstrings by `scripts/make_api_reference.py`. Private helpers (leading underscore) are omitted.*

## Modules

- [`config.settings`](#configsettings)
- [`src.newsbot`](#srcnewsbot)
- [`src.data_processing.data_loader`](#srcdata_processingdata_loader)
- [`src.data_processing.text_preprocessor`](#srcdata_processingtext_preprocessor)
- [`src.data_processing.data_validator`](#srcdata_processingdata_validator)
- [`src.data_processing.feature_extractor`](#srcdata_processingfeature_extractor)
- [`src.analysis.classifier`](#srcanalysisclassifier)
- [`src.analysis.sentiment_analyzer`](#srcanalysissentiment_analyzer)
- [`src.analysis.ner_extractor`](#srcanalysisner_extractor)
- [`src.analysis.topic_modeler`](#srcanalysistopic_modeler)
- [`src.language_models.summarizer`](#srclanguage_modelssummarizer)
- [`src.language_models.embeddings`](#srclanguage_modelsembeddings)
- [`src.language_models.generator`](#srclanguage_modelsgenerator)
- [`src.multilingual.language_detector`](#srcmultilinguallanguage_detector)
- [`src.multilingual.translator`](#srcmultilingualtranslator)
- [`src.multilingual.cross_lingual_analyzer`](#srcmultilingualcross_lingual_analyzer)
- [`src.conversation.intent_classifier`](#srcconversationintent_classifier)
- [`src.conversation.query_processor`](#srcconversationquery_processor)
- [`src.conversation.response_generator`](#srcconversationresponse_generator)
- [`src.utils.visualization`](#srcutilsvisualization)
- [`src.utils.evaluation`](#srcutilsevaluation)
- [`src.utils.export`](#srcutilsexport)

## `config.settings`

Central configuration for NewsBot 2.0.

### class `NewsBotConfig`

All tunable settings of the system in one object.

- **`ensure_dirs() -> None`** – Create every output directory if it does not exist yet.

## `src.newsbot`

NewsBot 2.0 – system orchestrator.

### class `NewsBot2`

The complete NewsBot 2.0 system.

- **`__init__(config=DEFAULT_CONFIG, use_transformers=None)`** – Create the object; parameters are described in the class docstring.
- **`build(data=None, max_webz=None, kg_docs=4000, retrain=False) -> 'NewsBot2'`** – Load data and prepare every component. Heavy, re-usable artefacts (enriched corpus, classifier, language detector) are cached in ``data/processed`` and ``data/models``; ``retrain=True`` rebuilds them.
- **`analyze_article(text, title='', summary_type='balanced', enhance=True) -> dict`** – Comprehensive analysis of one article in any supported language: language → (translation) → classification, sentiment, entities, relations, topics, summary, headlines, enhancements, cultural context. Each step is isolated: a failure is reported under ``errors`` and the rest of the analysis still runs.
- **`batch_analysis(articles, enhance=False, progress=True) -> pd.DataFrame`** – Analyse many articles; one bad article never stops the batch.
- **`chat(message) -> str`** – Chatbot entry point: returns the reply text (see ``process_query`` for data).
- **`process_query(message) -> dict`** – Parse a (multi-part) message, execute each part and build the reply.
- **`generate_insights_report(df=None, report_type='comprehensive') -> str`** – Markdown report: overview, insights, sentiment by category/language, topics, entities.
- **`export_article_report(analysis, path, fmt='markdown') -> Path`** – Write a Markdown/HTML report for one ``analyze_article`` result.
- **`save(path=None) -> Path`** – Pickle the whole system (handy for the web demo / quick restarts).
- **`load(path) -> 'NewsBot2'`** – Load an object previously written with :meth:`save`.

### function `training_frame(bbc, webz, random_state=42, webz_per_class=600) -> pd.DataFrame`

Training data for the 2.0 classifier: BBC train split (2004–05 style) + a balanced sample of English Webz articles (2023–26 vocabulary) so the model is not stuck in 2005. Held-out test articles are excluded.

## `src.data_processing.data_loader`

Dataset loading for NewsBot 2.0.

### function `load_bbc(cfg=DEFAULT_CONFIG, deduplicate=True) -> pd.DataFrame`

Load the BBC News corpus as a DataFrame.

### function `load_webz(cfg=DEFAULT_CONFIG, snapshots=None, languages=None) -> pd.DataFrame`

Load (and download if needed) the Webz.io multilingual news snapshots.

### function `load_all(cfg=DEFAULT_CONFIG, cache=True) -> dict`

Load both corpora, caching them as pickles in ``data/processed``.

### function `sample_articles() -> List[dict]`

A few short hand-written articles used in demos and unit tests.

## `src.data_processing.text_preprocessor`

Text preprocessing (enhanced from the midterm).

### class `TextPreprocessor`

Configurable text cleaning pipeline.

- **`__init__(language='en', lemmatize=False, min_token_len=2)`** – Create the object; parameters are described in the class docstring.
- **`clean(text) -> str`** – Light cleaning that keeps casing and punctuation (for spaCy/VADER).
- **`normalize(text, language=None) -> str`** – Aggressive normalisation for bag-of-words models (TF-IDF, LDA).
- **`batch_normalize(texts, languages=None) -> List[str]`** – Normalise many texts; ``languages`` can give a code per text.
- **`split_sentences(text) -> List[str]`** – Rule-based sentence splitter (fast, language-agnostic, no downloads).

## `src.data_processing.data_validator`

Data-quality checks run before any model is trained.

### class `ValidationReport`

Result of :func:`validate_corpus` – printable and JSON-serialisable.

- **`ok() -> bool`** – True when no ERROR-level issue was found.
- **`to_dict() -> dict`** – Plain ``dict`` representation (JSON-friendly).

### function `validate_corpus(df, min_words=40, max_imbalance=3.0) -> ValidationReport`

Check a corpus DataFrame for the problems that hurt NLP models.

## `src.data_processing.feature_extractor`

Feature extraction: TF-IDF (word + character), handcrafted style/sentiment features and dense document vectors (LSA, or transformer embeddings via ``src.language_models.embeddings``).

### function `build_tfidf_union(max_word_features=40000, max_char_features=40000) -> FeatureUnion`

Word 1–2 grams + character 3–5 grams. Character n-grams make the classifier robust to spelling variants, unseen inflections and OCR noise (an improvement over the word-only midterm model).

### class `StyleFeatures`

Length, punctuation and numeric-density features from the *clean* text.

- **`fit(X, y=None)`** – No-op fit (stateless transformer); returns ``self``.
- **`transform(X)`** – Transform the input texts into feature vectors.

### class `LSAEmbedder`

Dense document vectors via TF-IDF + truncated SVD (Latent Semantic Analysis). This is the offline fallback for transformer sentence embeddings.

- **`__init__(n_components=256, random_state=42)`** – Create the object; parameters are described in the class docstring.
- **`fit(texts) -> 'LSAEmbedder'`** – Learn the TF-IDF vocabulary and SVD projection from ``texts``.
- **`transform(texts) -> np.ndarray`** – Transform the input texts into feature vectors.
- **`vocabulary_neighbours(term, k=5) -> List[str]`** – Terms closest to ``term`` in the LSA space (used for query expansion).

### function `describe_features(df, text_col='text') -> pd.DataFrame`

Per-category summary of the style features (for EDA notebooks).

## `src.analysis.classifier`

Enhanced news classification (Module A).

### class `Prediction`

Result of :meth:`NewsClassifier.predict_with_confidence`.

- **`to_dict() -> dict`** – Plain ``dict`` representation (JSON-friendly).

### class `NewsClassifier`

Ensemble news classifier with confidence, multi-label and explanations.

- **`__init__(low_confidence=0.55, secondary_min=0.2, random_state=42)`** – Create the object; parameters are described in the class docstring.
- **`train(texts, labels) -> 'NewsClassifier'`** – Fit the ensemble and the explanation model on raw article texts.
- **`predict(texts) -> List[str]`** – Predicted label for each raw text.
- **`predict_proba(texts) -> np.ndarray`** – Class probabilities for each raw text (columns follow ``classes_``).
- **`predict_with_confidence(text) -> Prediction`** – Primary label, calibrated confidence, alternatives and a review flag.
- **`explain_prediction(text, top_n=8, n_similar=3) -> dict`** – Explain a decision with (1) the terms that contributed most to the predicted class in the word-level model and (2) similar training articles (nearest neighbours in TF-IDF space).
- **`evaluate(texts, labels) -> dict`** – Accuracy, macro-F1, per-class report and confusion matrix.
- **`calibration_table(texts, labels, bins=5) -> pd.DataFrame`** – Reliability table: accuracy within confidence bins (is 0.9 really 90%?).
- **`cross_validate(texts, labels, folds=5) -> dict`** – Stratified k-fold macro-F1 of the full pipeline (no leakage).
- **`save(path) -> None`** – Serialise this object to ``path`` with joblib (compressed).
- **`load(path) -> 'NewsClassifier'`** – Load an object previously written with :meth:`save`.

### function `compare_baselines(train_texts, train_labels, test_texts, test_labels) -> pd.DataFrame`

Compare single models against the ensemble on the same split.

### class `EmbeddingClassifier`

Logistic regression on multilingual sentence embeddings. Because the embedding space is shared across languages, a model trained on English BBC articles can label Spanish/German/French news (zero-shot transfer).

- **`__init__(embedder)`** – Create the object; parameters are described in the class docstring.
- **`train(texts, labels) -> 'EmbeddingClassifier'`** – Fit the model on texts and labels; returns ``self`` for chaining.
- **`predict(texts) -> List[str]`** – Predicted label for each raw text.
- **`predict_proba(texts) -> np.ndarray`** – Class probabilities for each raw text (columns follow ``classes_``).

## `src.analysis.sentiment_analyzer`

Advanced sentiment analysis (Module A).

### class `SentimentAnalyzer`

Multi-dimensional sentiment analysis for news.

- **`__init__(use_transformer=False)`** – Create the object; parameters are described in the class docstring.
- **`label(score, threshold=0.05) -> str`** – Map a compound score to positive / neutral / negative.
- **`sentence_scores(text) -> List[float]`** – VADER compound score for every sentence of ``text``.
- **`analyze(text, language='en') -> dict`** – Full sentiment profile of one article.
- **`emotions(text) -> Dict[str, float]`** – Relative frequency of the eight NRC emotions (sums to 1, or all 0).
- **`batch_scores(texts) -> pd.DataFrame`** – Fast batch scoring (sentence-average + emotions) for many articles.
- **`entity_sentiment(text, entities) -> Dict[str, dict]`** – Average sentiment of the sentences that mention each entity.
- **`track_over_time(df, date_col='published', score_col='sentiment', group_col='category', freq='Q') -> pd.DataFrame`** – Mean sentiment and article count per period (and per group).
- **`detect_anomalies(timeline, value_col='mean_sentiment', group_col='category', z=1.5) -> pd.DataFrame`** – Flag periods whose sentiment deviates more than ``z`` standard deviations from that group's own average – candidate "events".
- **`evaluate_against_labels(texts, gold) -> dict`** – Compare predicted polarity with external labels (e.g. Webz sentiment).

## `src.analysis.ner_extractor`

Named entity recognition, relation extraction and knowledge graph (Module A).

### function `normalize_entity(text, label='') -> str`

Canonical form of an entity string (aliases, titles, possessives, whitespace).

### class `EntityRelationshipMapper`

NER + relations + knowledge graph.

- **`__init__(model='en_core_web_sm', multilingual_model='xx_ent_wiki_sm')`** – Create the object; parameters are described in the class docstring.
- **`extract_entities(text, language='en') -> List[dict]`** – Entities with canonical name, type, character offsets and sentence.
- **`corpus_entities(texts, languages=None, batch_size=64, max_chars=1500, types=('PERSON', 'ORG', 'GPE', 'LOC', 'NORP', 'PRODUCT', 'EVENT')) -> list`** – Canonical entity list for every text (fast: uses the first ``max_chars`` characters, i.e. headline + lead, where news names its key actors). English uses ``en_core_web_sm``; other languages ``xx_ent_wiki_sm``.
- **`extract_relationships(text) -> List[dict]`** – Typed relations between entities found in the dependency parse.
- **`build_knowledge_graph(texts, categories=None, batch_size=32, min_cooccurrence=2) -> nx.Graph`** – Build an entity graph from many articles. Nodes carry type, mention count and category counts; edges carry co-occurrence weight and any typed relations (ROLE_AT, verbs) found between the two entities.
- **`find_entity_connections(entity1, entity2, max_paths=3) -> dict`** – Shortest paths between two entities, preferring strong edges.
- **`resolve(name) -> Optional[str]`** – Match user input to a graph node (canonical form, then case-insensitive, then substring).
- **`entity_profile(name, k=5) -> dict`** – Background for an entity: type, mentions, categories, strongest neighbours.
- **`top_entities(n=20, types=None) -> pd.DataFrame`** – Most-mentioned graph entities as a DataFrame (optionally filtered by type).

## `src.analysis.topic_modeler`

Topic modelling and content discovery (Module A, course Module 9).

### class `TopicModeler`

Parameters ---------- n_topics : int method : ``"lda"`` or ``"nmf"`` max_features : vocabulary size

- **`__init__(n_topics=10, method='lda', max_features=8000, random_state=42, language='en')`** – Create the object; parameters are described in the class docstring.
- **`fit_transform(documents, normalized=False) -> np.ndarray`** – Train the topic model; returns the document-topic matrix.
- **`get_topic_words(topic_id, n_words=10) -> List[str]`** – Top ``n_words`` terms of topic ``topic_id``.
- **`topics_table(n_words=10) -> pd.DataFrame`** – One row per topic: label, top words, prevalence and NPMI coherence.
- **`get_article_topics(article_text, top_n=3) -> List[dict]`** – Topic distribution of an unseen article (top ``top_n`` topics).
- **`coherence_per_topic(n_words=10) -> np.ndarray`** – NPMI coherence (Bouma 2009; Röder et al. 2015) of each topic's top words, using document co-occurrence in the training corpus. Range -1..1; higher = words that really appear together.
- **`coherence(n_words=10) -> float`** – Mean NPMI coherence over all topics.
- **`diversity(n_words=25) -> float`** – Share of unique words among all topics' top words (Dieng et al. 2020).
- **`dominant_topics() -> np.ndarray`** – Index of the strongest topic for every training document.
- **`cluster_purity(labels) -> float`** – How well dominant topics align with known categories (purity).
- **`track_topic_trends(dates, freq='Q', groups=None) -> pd.DataFrame`** – Average topic share per period (long format). With ``groups`` (e.g. language) the trend is computed per group as well.
- **`emerging_topics(trends, top_n=5) -> pd.DataFrame`** – Topics whose share grew most between the first and last period.
- **`visualize_topics(title='Intertopic distance map')`** – Interactive bubble chart: topics placed by similarity, sized by prevalence.

### function `select_n_topics(documents, candidates=(6, 8, 10, 12, 15), method='nmf', random_state=42) -> pd.DataFrame`

Fit one model per candidate k and report coherence and diversity.

## `src.language_models.summarizer`

Intelligent summarisation (Module B).

### class `IntelligentSummarizer`

Parameters ---------- use_transformer : bool     Load DistilBART for abstractive summaries/headlines when possible. model_name : str     HuggingFace summarisation model.

- **`__init__(use_transformer=False, model_name='sshleifer/distilbart-cnn-12-6')`** – Create the object; parameters are described in the class docstring.
- **`abstractive_available() -> bool`** – True when the DistilBART pipeline was loaded.
- **`lead(text, n=3) -> str`** – First ``n`` sentences – the classic news baseline.
- **`textrank(text, n=3, position_weight=0.15, mmr_lambda=0.7) -> str`** – Graph-based extractive summary; sentences kept in original order.
- **`abstractive(text, max_length=130, min_length=30) -> str`** – Abstractive summary with DistilBART (falls back to TextRank).
- **`summarize_article(text, summary_type='balanced', method='auto') -> dict`** – Summarise one article.
- **`summarize_multiple_articles(texts, n_sentences=5, focus_topic=None) -> str`** – Multi-document summary: pool the lead sentences of every article, optionally re-weight them by similarity to ``focus_topic``, then run TextRank+MMR across the pool so repeated facts appear only once.
- **`generate_headlines(text) -> Dict[str, str]`** – Headline variants: informative (abstractive or compressed lead), keyword (top TF-IDF noun phrases) and social (short + hook).
- **`assess_summary_quality(original, summary, reference=None, entities=None) -> dict`** – Reference-based (ROUGE-1/2/L F1 when ``reference`` is given) and reference-free metrics: compression, entity coverage, redundancy (share of repeated bigrams) and novel-word ratio (abstractiveness).

## `src.language_models.embeddings`

Semantic embeddings and semantic search (Module B, course Module 10).

### class `SemanticEmbedder`

Text -> L2-normalised vector, transformer or LSA backend.

- **`__init__(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', use_transformer=True, lsa_components=256)`** – Create the object; parameters are described in the class docstring.
- **`multilingual() -> bool`** – True when the active backend embeds all languages in one space.
- **`fit(texts) -> 'SemanticEmbedder'`** – Only needed for the LSA backend (learns the vocabulary).
- **`encode(texts, batch_size=64) -> np.ndarray`** – Embed texts as L2-normalised vectors (transformer or LSA backend).
- **`neighbours(term, k=5) -> List[str]`** – Vocabulary terms closest to ``term`` (LSA space) for query expansion.

### class `SemanticSearchEngine`

Hybrid semantic search over an article DataFrame.

- **`__init__(embedder=None, alpha=0.6)`** – Create the object; parameters are described in the class docstring.
- **`encode_documents(df) -> 'SemanticSearchEngine'`** – Index the articles (dense + sparse).
- **`expand_query(query, k=2) -> Dict[str, List[str]]`** – Related terms for each query word (synonym map + LSA neighbours).
- **`semantic_search(query, top_k=5, filters=None, expand=True) -> pd.DataFrame`** – Natural-language search ("technology companies facing regulation"). Returns the top articles with dense, sparse and combined scores.
- **`find_similar_articles(query_article, top_k=5, exclude_doc_id=None, filters=None) -> pd.DataFrame`** – "More like this": nearest neighbours of an article in embedding space.
- **`cluster_similar_content(n_clusters=8, subset=None, random_state=42) -> pd.DataFrame`** – K-means story clusters with their most central article and top terms.
- **`detect_near_duplicates(threshold=0.95, max_pairs=50) -> List[tuple]`** – Pairs of articles that are (almost) the same story – syndication detection.

### function `retrieval_eval(engine, label_col='category', k=10, n_queries=200, random_state=42) -> dict`

Precision@k using articles as queries: a retrieved article counts as relevant when it shares the query article's category.

## `src.language_models.generator`

Content enhancement and insight generation (Module B).

### class `OllamaClient`

Minimal client for a local Ollama server (optional).

- **`__init__(host='http://localhost:11434', model='llama3.2', timeout=60)`** – Create the object; parameters are described in the class docstring.
- **`available() -> bool`** – True if an Ollama server answers at ``host`` (checked once).
- **`rewrite(draft, instruction='Rewrite as a friendly, concise answer.') -> str`** – Ask the LLM to rephrase ``draft`` without adding facts; returns draft on failure.

### class `ContentEnhancer`

Adds context to articles and produces corpus-level insights.

- **`__init__(entity_mapper=None, search_engine=None, corpus_stats=None, llm=None)`** – Create the object; parameters are described in the class docstring.
- **`enhance_article(text, entities=None, category=None, sentiment=None, k_related=3) -> dict`** – Context pack for one article: background on its key entities (from the knowledge graph), related coverage, how its tone compares with the category average, and a short "why it matters" line.
- **`extract_numeric_facts(text) -> List[dict]`** – Numbers with a few words of context (amounts, percentages, scores).
- **`cross_reference_facts(text, k=5) -> List[dict]`** – Lightweight consistency check: find related articles and report numeric facts whose context words match but whose values differ – candidates for a human fact-checker (not an automatic verdict).
- **`generate_insights(df, top_n=5) -> List[str]`** – Plain-English findings from an analysed article collection. Expects columns ``category`` and ``sentiment`` (optionally ``language``, ``published``, ``entities``).
- **`detect_information_gaps(df, topic_mask, group_col='language') -> dict`** – For one topic, which groups (languages/sources) under-cover it compared with their overall share – possible blind spots.
- **`polish(draft) -> str`** – Optionally rephrase a grounded draft with the local LLM.

## `src.multilingual.language_detector`

Language identification (Module C, course Module 11).

### class `LanguageDetector`

Character n-gram language identifier with confidence scores.

- **`__init__(min_confidence=0.6, max_chars=600)`** – Create the object; parameters are described in the class docstring.
- **`make_training_snippets(texts, labels, per_text=3, seed=42) -> tuple`** – Cut articles into sentence-sized snippets so the model also learns short inputs.
- **`train(texts, labels) -> 'LanguageDetector'`** – Fit the model on texts and labels; returns ``self`` for chaining.
- **`detect_language(text) -> dict`** – Most likely language, confidence and the top alternatives.
- **`detect_batch(texts) -> List[str]`** – Most likely language for each text (no confidence filtering).
- **`detect_segments(text) -> dict`** – Per-sentence detection for code-switched / mixed-language text. Returns the segments and the language shares.
- **`evaluate(texts, labels, lengths=(20, 50, 100, 300)) -> dict`** – Accuracy overall and when only the first N characters are available.
- **`save(path) -> None`** – Serialise this object to ``path`` with joblib (compressed).
- **`load(path) -> 'LanguageDetector'`** – Load an object previously written with :meth:`save`.

## `src.multilingual.translator`

Translation integration (Module C).

### class `Translator`

Parameters ---------- use_transformer : bool     Load MarianMT models on demand (one per language pair, cached). max_sentences : int     Long articles are translated sentence by sentence up to this limit.

- **`__init__(use_transformer=True, max_sentences=40, device=None)`** – Create the object; parameters are described in the class docstring.
- **`translate_text(text, source_language, target_language='en') -> dict`** – Translate ``text``; returns ``{"text", "translated", "engine"}``. Same-language input is returned unchanged.
- **`translate_batch(texts, languages, target_language='en') -> List[dict]`** – Translate many texts, one language code per text.
- **`assess_quality(source, source_language, translation=None) -> dict`** – Reference-free quality estimate. ``round_trip_chrf`` compares the source with its back-translation (0–100, higher is better).
- **`corpus_scores(hypotheses, references) -> dict`** – BLEU and chrF for a set of translations against references.

## `src.multilingual.cross_lingual_analyzer`

Cross-lingual analysis and cultural context (Module C).

### class `CrossLingualAnalyzer`

Comparative analysis of a multilingual, analysed article DataFrame.

- **`__init__(df, entity_col='entities')`** – Create the object; parameters are described in the class docstring.
- **`coverage_by_language() -> pd.DataFrame`** – Category share within each language (rows sum to 1).
- **`sentiment_by_language(label_col='webz_sentiment') -> pd.DataFrame`** – Share of positive articles per language × category (+ overall).
- **`entities_by_language(top_n=15) -> Dict[str, List[tuple]]`** – Most frequent entities (document frequency) per language.
- **`entity_overlap(top_n=50) -> pd.DataFrame`** – Jaccard overlap of the top-N entities between every pair of languages.
- **`perspective_on(entity, label_col='webz_sentiment', min_articles=5) -> pd.DataFrame`** – How each language covers one entity: volume, positive share, top co-mentions.
- **`link_stories(min_shared=3, window_days=7, max_links=30) -> pd.DataFrame`** – Same story in different languages: articles that share at least ``min_shared`` named entities and were published within ``window_days``.
- **`country_focus() -> pd.DataFrame`** – Share of each language's articles coming from its 'home' countries.
- **`extract_cultural_context(text, source_language) -> dict`** – Culture-specific references (institutions, leagues, indices) and currencies in a text.

## `src.conversation.intent_classifier`

Intent classification for user queries (Module D, course Module 12).

### function `generate_training_data(n_per_template=6, seed=42) -> Tuple[List[str], List[str]]`

Fill every template ``n_per_template`` times with random slot values.

### class `IntentClassifier`

TF-IDF (word + char) + logistic regression intent model with confidence.

- **`__init__(min_confidence=0.35)`** – Create the object; parameters are described in the class docstring.
- **`train(X=None, y=None) -> 'IntentClassifier'`** – Fit the model on texts and labels; returns ``self`` for chaining.
- **`classify_intent(query) -> dict`** – Return the intent, its probability and the runner-up.
- **`evaluate(queries=None) -> dict`** – Accuracy, macro-F1, per-intent report and misclassified queries on the held-out set.
- **`save(path) -> None`** – Serialise this object to ``path`` with joblib (compressed).
- **`load(path) -> 'IntentClassifier'`** – Load an object previously written with :meth:`save`.

## `src.conversation.query_processor`

Natural-language query understanding and conversation state (Module D).

### class `ParsedQuery`

Structured representation of one user request.

- **`filters() -> dict`** – Search filters derived from the slots.

### class `ConversationContext`

Dialogue memory: previous queries, slots and result ids.

- **`remember(query, result_ids=None, response='') -> None`** – Store a parsed query, its result ids and the bot reply; update user interests.
- **`top_interests(n=3) -> List[str]`** – The ``n`` categories/languages/entities the user asked about most.

### class `QueryProcessor`

Parse user messages into :class:`ParsedQuery` objects.

- **`__init__(intent_classifier=None, nlp=None, reference_date=None, known_entities=None)`** – Create the object; parameters are described in the class docstring.
- **`extract_query_entities(query) -> Dict[str, object]`** – All slots found in a query (see module docstring).
- **`parse(query, context=None) -> List[ParsedQuery]`** – Parse a (possibly multi-part) message into one or more queries.

## `src.conversation.response_generator`

Response generation (Module D).

### class `ResponseGenerator`

Format engine results as chatbot replies.

- **`__init__(llm=None, seed=0)`** – Create the object; parameters are described in the class docstring.
- **`describe_filters(slots) -> str`** – Human-readable description of the active filters ("positive tech news in es ...").
- **`search(results, slots) -> str`** – Numbered list of search results with metadata.
- **`summary(summary, n_articles, slots) -> str`** – Wrap a multi-document summary in a reply.
- **`sentiment(stats, slots) -> str`** – Describe the tone of a set of articles.
- **`compare(rows) -> str`** – Side-by-side comparison of two or more groups/entities.
- **`topics(table, emerging=None) -> str`** – List the main (and fastest-growing) topics.
- **`trends(timeline, subject) -> str`** – Describe a sentiment timeline: direction, busiest period, recent values.
- **`entity(info) -> str`** – Describe an entity profile or the connection between two entities.
- **`classification(analysis) -> str`** – Compact reply for a full single-article analysis.
- **`translation(result, source_language) -> str`** – Reply with a translation (or explain why it is unavailable).
- **`insights(insights, interests=None) -> str`** – Bullet list of corpus insights plus personalised suggestion.
- **`export(path) -> str`** – Confirm where an export was written.
- **`greeting(text) -> str`** – Reply to greetings, thanks and goodbyes.
- **`unknown() -> str`** – Fallback reply with usage examples.
- **`finalize(text) -> str`** – Optional LLM polishing (grounded – the draft contains all facts).

## `src.utils.visualization`

Plotting functions (matplotlib for static report figures, Plotly for interactive notebook/web views). Every function returns the figure and can save it when ``path`` is given.

### function `bar(series, title, xlabel='', path=None, color=None, horizontal=False)`

Bar chart of a Series (category colours where possible).

### function `confusion_matrix_plot(cm, labels, title, path=None)`

Row-normalised confusion matrix with counts.

### function `model_comparison(df, title, metric='macro_f1', path=None)`

Horizontal bar chart comparing models on one metric.

### function `timeline(df, x, y, group, title, ylabel='', path=None, anomalies=None)`

Line chart over periods, one line per group, anomalies circled.

### function `heatmap(table, title, path=None, fmt='{:.2f}', cmap='RdYlGn', vmin=None, vmax=None)`

Annotated heatmap of a DataFrame.

### function `topic_trends(trends, title, top_n=6, path=None)`

Share of the ``top_n`` topics per period.

### function `network(graph, title, top_n=40, path=None, seed=7)`

Draw the most-mentioned entities and the strongest links between them.

### function `stacked_share(table, title, path=None)`

Stacked horizontal bars of shares (rows sum to 1).

### function `system_architecture(path=None)`

Static architecture diagram for documentation and slides.

## `src.utils.evaluation`

Evaluation helpers shared by notebooks, tests and the experiment script.

### function `to_jsonable(obj) -> Any`

Convert numpy / pandas objects so ``json.dump`` accepts them.

### function `save_json(data, path) -> Path`

Write ``data`` as pretty JSON (numpy/pandas values converted).

### function `load_json(path) -> Dict`

Read a JSON file into a dict.

### function `timer(store, key)`

``with timer(results, 'train_s'):`` – records elapsed seconds.

### function `per_class_table(report) -> pd.DataFrame`

sklearn ``classification_report(output_dict=True)`` -> tidy DataFrame.

## `src.utils.export`

Report generation and export (Markdown, HTML, JSON, CSV).

### function `article_report_markdown(analysis, title='Article analysis') -> str`

Readable Markdown report for one ``NewsBot2.analyze_article`` result.

### function `markdown_to_html(md, title='NewsBot report') -> str`

Very small Markdown -> HTML converter (headings, bullets, emphasis).

### function `export_results(data, path, fmt=None) -> Path`

Save ``data`` (DataFrame, dict or Markdown string) as csv / json / markdown / html, chosen from ``fmt`` or the file extension.
