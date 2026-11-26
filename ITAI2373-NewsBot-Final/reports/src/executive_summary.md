<div class="cover">
<h1>NewsBot 2.0 – Executive Summary</h1>
<div class="sub">Multilingual news intelligence for newsrooms, communications teams and analysts<br>
Mourinho Hernanto (team: Solo) · ITAI 2373 Final Project · December 2026</div>
</div>

## The problem

Organisations that depend on news – newsrooms, PR and communications teams, policy and market analysts – face
thousands of articles a day in several languages. Today, people skim, tag and summarise this flow by hand. It is
slow, coverage in other languages is mostly ignored, and important shifts in tone or new topics are noticed late.

## The solution

**NewsBot 2.0** reads news in six languages (English, Spanish, German, French, Italian, Portuguese) and turns it into
answers a non-technical user can ask for in plain English:

* *"Show me negative tech news from this week"* → ranked articles, filtered by tone, section, language and date
* *"Summarize them"* → a short multi-article summary, translated when needed
* *"How has sentiment about China changed?"* → a quarter-by-quarter trend
* *"How are Donald Trump and China connected?"* → shared coverage and extracted relationships
* *"Compare Spanish and German politics news"* → coverage volume and tone side by side

Behind the chat interface, every article is automatically sorted into a section with a **confidence score**,
scored for tone and emotion, mined for people, organisations and places, placed in discovered topics, summarised
and enriched with background. Uncertain cases are **flagged for a human** instead of being guessed silently.

<div class="kpi"><b>97%</b><span>section accuracy on the benchmark news set</span></div><div class="kpi"><b>83%</b><span>on brand-new 2023–26 news (up from 69%)</span></div><div class="kpi"><b>~0.1 s</b><span>to fully analyse one article on a laptop</span></div><div class="kpi"><b>6 langs</b><span>detected automatically (98.6% on 100 characters)</span></div>

## Value proposition

| For | Value |
|---|---|
| **Editors & journalists** | automatic tagging and summaries; story clusters; "who is connected to whom" background in seconds |
| **Communications / PR** | tone tracking for brands and people across languages; early warning when sentiment turns |
| **Analysts & researchers** | topic discovery and trends over time; exportable data (CSV/JSON) and reports |
| **International teams** | foreign-language coverage made searchable and comparable – different countries cover the same events very differently (e.g. 52% of Spanish-language coverage is politics vs 10% in German) |

**Trust by design.** Confidence scores are calibrated (when NewsBot says ~97% sure, it was right 99% of the time in
testing); flagged articles are where the errors concentrate (38% correct vs 84% for unflagged), so human time is
spent where it matters. Generated text only uses facts the system extracted.

## ROI analysis (illustrative)

*Assumptions are deliberately conservative and should be replaced with an organisation's own figures.*

| Assumption | Value |
|---|---|
| Articles screened per day by a small monitoring team | 400 |
| Manual time per article (skim, tag section/tone, note key names) | 2.5 minutes |
| Fully loaded analyst cost | $35 / hour |
| Articles NewsBot routes to human review | ≈ 10% (measured: 9.2%), reviewed in 1.5 minutes each |
| Human spot-check of automated results | 15 minutes / day |

| | Manual | With NewsBot |
|---|---|---|
| Analyst time per day | 400 × 2.5 min = **16.7 h** | 40 × 1.5 min + 15 min = **1.25 h** |
| Cost per year (250 working days) | ≈ **$146,000** | ≈ **$10,900** |
| Running cost per year | – | ≈ $1,800 (one cloud VM or existing laptop; open-source software, no licence fees) |

**Estimated saving ≈ $133,000 per year (≈ 15 analyst-hours per day freed)** for a 400-article/day workload, before
counting the value of covering five additional languages and detecting trends earlier. Even if only 25% of the
manual time is really replaced, the saving (~$34,000/year) is ~19× the running cost. One-off setup is a few days of
configuration (data sources, categories) because all components are already built and documented.

## Use cases

1. **Morning brief for a newsroom desk** – "Give me an overview" and "What topics are trending in business?" produce a
   brief with the main themes, the most mentioned names and how tone changed since last quarter.
2. **Brand / reputation monitoring** – "Compare Apple and Google" and "How has sentiment about Tesla changed over time?";
   anomaly alerts mark quarters where tone deviates strongly from normal.
3. **International coverage desk** – the same story is linked across languages (e.g. an NBA game reported in Spanish and
   English the same day) and cultural references (Bundestag, Moncloa, Ibovespa) are explained.
4. **Research & teaching** – topic models, entity networks and exports support media-literacy and journalism studies.

## Competitive analysis

| | Generic news aggregators / RSS readers | Enterprise media-monitoring suites | Raw NLP APIs | **NewsBot 2.0** |
|---|---|---|---|---|
| Natural-language Q&A with follow-ups | ✗ | partial | ✗ (build it yourself) | ✓ |
| Sentiment, entities, topics, summaries | limited | ✓ | ✓ (separately priced calls) | ✓ integrated |
| Cross-language comparison | ✗ | ✓ (premium tiers) | translation only | ✓ (6 languages, own language ID) |
| Calibrated confidence + human-review routing | ✗ | rarely exposed | ✗ | ✓ |
| Transparent, explainable decisions | ✗ | proprietary | proprietary | ✓ key terms + similar articles |
| Runs on own hardware / data stays in-house | ✗ | ✗ (SaaS) | ✗ | ✓ |
| Cost | free–low | enterprise subscription | pay per call | open source; infrastructure only |

**Positioning:** NewsBot offers the integrated analysis of an enterprise monitoring suite with the transparency and
cost profile of open-source software – suited to small newsrooms, NGOs, universities and teams that need control over
their data.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Automated tone/category errors | calibrated confidence, review flag, explanations, human spot checks |
| Bias in sources (English/US-heavy) | per-language normalisation, coverage-gap detection, transparent source lists |
| Copyright & data terms | only public datasets; raw text is never redistributed |
| Model drift as vocabulary changes | demonstrated retraining on recent news (+14 pts); retrain quarterly |

## Next steps

1. Pilot with one desk for four weeks on its own sources; measure time saved and review rate.
2. Add a simple web front end on the existing `NewsBot2` API.
3. Fine-tune a multilingual transformer classifier and add live RSS ingestion for real-time alerts.
