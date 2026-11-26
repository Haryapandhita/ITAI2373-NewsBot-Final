# Individual Contributions

**Team:** Solo (individual project – no other members were available in the class)
**Member:** Mourinho Hernanto – responsible for **every** module, document and the presentation.

| Area | Contribution |
|---|---|
| Planning & architecture | Mapped the midterm notebook onto the required `src/` package structure; designed the four-module architecture and the `NewsBot2` orchestrator; chose the datasets |
| Data (Modules 1–3) | Replaced the lower-cased Kaggle BBC copy with the original-casing BBC archive; added the multilingual, dated Webz.io corpus; data loader, validator, 6-language preprocessor, word+char TF-IDF and LSA features; automatic language-label checking |
| Module A – content analysis | Soft-voting ensemble classifier with calibrated confidence, review flag, multi-label output and explanations; domain-shift experiment (2004 → 2026 news); sentence-level sentiment with news lexicon, NRC emotions, entity sentiment, temporal tracking and anomaly detection; NER with multilingual aliases, dependency-based relation extraction and knowledge graph; LDA/NMF topic models with own NPMI coherence, model selection and topic evolution |
| Module B – language models | TextRank+MMR extractive summariser, DistilBART integration, multi-document summaries, headline generation, summary quality metrics; hybrid semantic search (multilingual MiniLM / LSA), query expansion, clustering, near-duplicate detection; grounded content enhancement and fact cross-referencing |
| Module C – multilingual | Own character n-gram language detector with confidence and mixed-language segmentation; MarianMT translation with round-trip quality estimation; cross-lingual coverage, tone, entity overlap, regional perspectives, story linking and cultural-context detection |
| Module D – conversation | Template-bootstrapped intent dataset and classifier, held-out evaluation; slot extraction (dates, languages, sentiment, entities, formats); conversation memory for follow-ups and multi-part questions; response generation with optional local LLM polishing |
| Integration & quality | Error-isolated `analyze_article`, batch analysis, caching, persistence, export; 29 automated tests; experiment script reproducing every reported number |
| Documentation | README, technical documentation, user guide, API reference generator, deployment guide, executive summary, reflective journal, 7 tutorial notebooks, presentation with speaker notes |

## Use of AI assistance

As permitted by the course, I used an AI assistant (Claude, Anthropic) as a pair-programming and writing aid:
discussing architecture options, generating first drafts of code and documentation, debugging, and suggesting
evaluation designs. I reviewed, ran and tested all code, made the design decisions (datasets, models, what to
evaluate and how), and verified every reported number by running `scripts/run_experiments.py`. External
libraries, models and datasets are credited in the README *References* section.
