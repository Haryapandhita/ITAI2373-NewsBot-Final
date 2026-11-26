# Raw data (downloaded automatically, not committed)

| Dataset | Source | Licence / terms | Used for |
|---|---|---|---|
| **BBC News** – 2,225 articles (2004–05), 5 categories, original casing + headlines | D. Greene & P. Cunningham (ICML 2006), mirror: https://github.com/jonathanporta/bbc-news-article-archive | content © BBC, research use | classification, NER, summarisation |
| **Webz.io free news datasets** – 20 snapshots, 6 languages, Dec 2023 – Nov 2026 | https://github.com/Webhose/free-news-datasets | Webz.io Terms of Use (non-transferable licence – do not redistribute) | multilingual analysis, topic & sentiment evolution, cross-domain test |

`src/data_processing/data_loader.py` downloads both into this folder on first use
(`load_all()` or notebook 01). Snapshot names are listed in `config/settings.py`.
