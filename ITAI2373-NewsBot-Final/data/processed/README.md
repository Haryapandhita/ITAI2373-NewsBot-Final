# Processed data (generated, not committed)

* `bbc.pkl`, `webz.pkl` – cleaned, de-duplicated corpora (created by `load_all()`).
* `corpus_enriched.pkl` – corpus with sentiment, emotions, detected language and entity lists (created by `NewsBot2.build()`).

Delete these files (or call `build(retrain=True)`) to rebuild from the raw data.
