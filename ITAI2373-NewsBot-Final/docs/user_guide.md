# NewsBot 2.0 – User Guide

*For journalists, analysts and anyone who wants answers from a large collection of news – no programming knowledge needed beyond pressing "Run".*

---

## 1. What NewsBot does for you

| You want to… | Ask NewsBot | What you get |
|---|---|---|
| Find articles | "Show me positive tech news from this year" | A ranked list with category, language, date and tone |
| Catch up quickly | "Summarize politics news in Spanish" | A short summary of the most relevant articles (translated if needed) |
| Understand the mood | "Is coverage of the economy mostly negative?" | Average tone, % positive/negative, strongest emotion |
| Track change | "How has sentiment about China changed over time?" | A quarter-by-quarter timeline and the busiest period |
| Compare | "Compare Apple and Google" / "Compare Spanish and German politics news" | Coverage volume, tone and main section side by side |
| Understand connections | "How are Donald Trump and China connected?" / "Who is Elon Musk?" | Shared coverage, extracted relations, closest connections |
| Discover themes | "What topics are trending in business?" | Main topics and the fastest-growing ones |
| Check one article | "Classify this article: <paste text>" | Section, confidence, tone, entities, topics and a summary |
| Read foreign news | "Translate this: <text>" / "Show German news in English" | English translation (MarianMT) |
| Get the big picture | "Give me an overview" | Key insights across the whole collection |
| Take results away | "Export the results as csv" (or json / html / markdown) | A file in `data/results/exports/` |

Type **help** at any time to see examples.

## 2. Getting started (Google Colab – recommended)

1. Open any notebook from the repository on GitHub and click **Open in Colab** (or upload it to colab.research.google.com).
2. *Optional but faster:* **Runtime → Change runtime type → T4 GPU**. This enables the transformer models (better summaries, translation and cross-language search).
3. Click **Runtime → Run all**. The first cell downloads the code, installs everything and downloads the data (≈ 2–3 min).
4. The first time the full system is built it takes ~8–10 minutes on CPU (fewer with a GPU). Results are cached, so later runs start in seconds.

To chat with NewsBot open **`notebooks/06_Conversational_Interface.ipynb`** and use the `ask("...")` helper:

```python
ask("Show me negative sports news from 2025")
ask("only the German ones")      # follow-up – keeps the previous filters
ask("summarize them")            # follow-up – summarises the previous results
```

To analyse your own article open **`notebooks/07_System_Integration.ipynb`**:

```python
result = bot.analyze_article("Paste the article text here ...", title="Optional headline")
result["classification"], result["sentiment"], result["summary"]
```

## 3. Tips for good questions

* **Combine filters freely:** sentiment (*positive, negative*), section (*tech, business, politics, sport, entertainment*), language (*Spanish, German, French, Italian, Portuguese, English*), time (*today, this week, last month, this year, last year, in 2025, Q3 2025, March 2026*), and names (*Apple, Elon Musk, China*).
* **Follow-ups work:** "what about last year?", "only negative ones", "and in French?", "summarize them".
* **Several requests at once:** "show German business news **and then** summarize them".
* **Ask for the size you want:** "top 3 articles about Tesla", "brief summary", "detailed summary".
* **Relative dates** ("this week") are measured from the newest article in the collection, not from today's calendar date.

## 4. Reading the results

* **Confidence** – how sure the classifier is (0–100%). Below 55%, or when two sections are close, the article is marked *needs human review*. In our tests, articles that are *not* flagged are classified correctly far more often than flagged ones.
* **Secondary labels** – a story can belong to two sections (e.g. a football club's finances: *sport* + *business*).
* **Tone score** – average sentence sentiment from −1 (very negative) to +1 (very positive). News is usually between −0.3 and +0.3; ±0.05 counts as neutral.
* **Emotions** – the eight basic emotions (anger, anticipation, disgust, fear, joy, sadness, surprise, trust) based on the NRC word list.
* **Topics** – themes discovered automatically from word patterns; their names are the three most typical words.
* **"Mostly in" / "often with"** (entity background) – how often a person/organisation appears in the collection and with whom.

## 5. FAQ & troubleshooting

**The first run is slow.** Downloading data and building the knowledge graph and search index takes a few minutes once. Everything is cached in `data/processed` and `data/models` afterwards.

**I see "Sentence-Transformer unavailable – falling back to LSA" or "Translation models aren't available".** The transformer models could not be downloaded (no internet or not installed). NewsBot keeps working with classical methods. In Colab with internet this message should not appear.

**A search returns nothing.** The filters are too narrow (e.g. a language + section + month with no articles). Remove a filter or widen the time range. Note that BBC articles have no date, so date filters only search the 2023–2026 collection.

**"Could not load saved classifier … retraining".** The saved model was created with another scikit-learn version. NewsBot retrains automatically (≈ 40 s).

**Can I use my own articles?** Yes – `bot.analyze_article(text)` for one article or `bot.batch_analysis(list_of_texts)` for many. To add a whole new dataset, create a DataFrame with the columns described in `docs/technical_documentation.md` §4 and pass it to `NewsBot2.build(data=...)`.

**Can I make the chatbot sound more natural?** Install [Ollama](https://ollama.com), run `ollama pull llama3.2`, set `NEWSBOT_USE_LLM=1`. NewsBot then lets the local model rephrase its answers – using only the facts NewsBot computed.

**Is the analysis always right?** No. Sentiment tools misread sarcasm and some domain words; machine translation and entity recognition make mistakes, especially in short or messy web text. Treat results as decision support and check important findings in the original articles (see *Limitations* in the technical documentation).

## 6. Video walkthrough

A short screen recording of the main workflows (chat, single-article analysis, report export) accompanies the final presentation; see the link in the repository README.
