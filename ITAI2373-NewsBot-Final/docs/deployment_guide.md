# NewsBot 2.0 – Deployment Guide

Three ways to run NewsBot, from easiest to most "production-like".

## 1. Google Colab (zero install)

Open any notebook in `notebooks/` in Colab and run all cells. The first cell (`scripts/colab_setup.py`)
clones the repository, installs `requirements.txt` and the spaCy models. GPU runtime is optional but speeds up the
transformer components ~10×.

## 2. Local installation (macOS / Linux / Windows)

Requirements: Python 3.10–3.13, ~3 GB free disk (models + data), 8 GB RAM recommended.

```bash
git clone https://github.com/Haryapandhita/ITAI2373-NewsBot-Final.git
cd ITAI2373-NewsBot-Final/ITAI2373-NewsBot-Final
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm

python -m pytest -q                      # 29 tests (the end-to-end test needs the data → runs after step below)
python scripts/run_experiments.py        # downloads data, builds the system, writes data/results/ (~15 min CPU)
jupyter notebook notebooks/              # or open them in VS Code
```

*Lightweight mode* – skip the transformer packages (`transformers`, `sentence-transformers`, `torch`) or set
`NEWSBOT_USE_TRANSFORMERS=0`. Everything still works with classical fallbacks (TextRank summaries, LSA search,
no machine translation); memory use drops below 2 GB.

## 3. Using NewsBot as a Python service

```python
from src.newsbot import NewsBot2
bot = NewsBot2().build()          # first time: builds + caches artefacts
bot.save()                        # data/models/newsbot2_system.joblib
...
bot = NewsBot2.load("data/models/newsbot2_system.joblib")   # fast restart (seconds)
bot.analyze_article(text); bot.chat("show me tech news")
```

The object can be wrapped by any web framework. A minimal Flask API (not required for the course, kept here as a
deployment pattern):

```python
from flask import Flask, request, jsonify
from src.newsbot import NewsBot2
from src.utils.evaluation import to_jsonable

bot = NewsBot2.load("data/models/newsbot2_system.joblib")
app = Flask(__name__)

@app.post("/analyze")
def analyze():
    return jsonify(to_jsonable(bot.analyze_article(request.json["text"], request.json.get("title", ""))))

@app.post("/chat")
def chat():
    return jsonify(to_jsonable(bot.process_query(request.json["message"])))
```

Run with `gunicorn -w 1 -b 0.0.0.0:8000 app:app` (one worker: the bot keeps conversation state in memory; for
several users create one `ConversationContext` per session).

## 4. Configuration for deployment

All settings live in `config/settings.py`; secrets/switches come from environment variables
(template: `config/api_keys_template.txt`).

| Variable | Default | Effect |
|---|---|---|
| `NEWSBOT_USE_TRANSFORMERS` | `1` | `0` = classical fallbacks only (fast, low memory) |
| `NEWSBOT_EMBEDDING_MODEL` | multilingual MiniLM | any Sentence-Transformers model |
| `NEWSBOT_SUMMARY_MODEL` | `sshleifer/distilbart-cnn-12-6` | any HF summarisation model |
| `NEWSBOT_USE_LLM` / `OLLAMA_HOST` / `OLLAMA_MODEL` | off | local LLM to polish chatbot replies |

## 5. Operations checklist

* **Caching:** `data/processed/corpus_enriched.pkl` and `data/models/*.joblib` – delete or call `build(retrain=True)` after changing data or code.
* **Logging:** every module logs through `src/utils/logging_utils.py` (timestamp | level | module). Build timings are in `bot.timings`.
* **Error isolation:** `analyze_article` runs each step in isolation and reports failures in `result["errors"]`; `batch_analysis` never stops on one bad article.
* **Updating data:** add new Webz snapshot names to `config.settings.webz_snapshots` (or pass your own DataFrame to `build(data=...)`).
* **Model versioning:** saved models are re-trained automatically if the installed scikit-learn version cannot load them.
* **Data licensing:** the raw articles are not redistributed (BBC copyright; Webz.io terms) – deployments download them at build time.
