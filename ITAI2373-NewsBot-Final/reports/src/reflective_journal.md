<div class="cover">
<h1>Reflective Journal – NewsBot 2.0</h1>
<div class="sub">Team: Solo · Member: Mourinho Hernanto · ITAI 2373 Final Project · December 2026</div>
</div>

## 1. Team and individual contributions

This was planned as a team project, but no teams formed in our class, so I completed it individually. As the only
member I was responsible for every part: choosing the datasets, restructuring my midterm notebook into the required
`src/` package, implementing the four modules (content analysis, language models, multilingual, conversation),
the orchestrator, tests, notebooks, documentation and the presentation. Working alone meant there was no division of
labour or peer code review, so I relied on automated tests (29), a single experiment script that regenerates every
number, and honest error analysis instead.

## 2. What we built and how it grew from the midterm

My midterm NewsBot was one Colab notebook: it classified 1,432 lower-cased BBC articles and extracted entities and
sentiment. NewsBot 2.0 is a modular package that analyses 15,242 articles in six languages from 2004–2005 and
2023–2026, adds topic modelling, summarisation, semantic search, translation, cross-language comparison and a chatbot
that understands follow-up questions.

## 3. Challenges and how they were overcome

**Old data, new world.** The classifier scored 97% on BBC news from 2004, but only 69% on 2023–26 articles. This
showed that a high test score does not guarantee that a system works on today's news. Adding a balanced sample of recent articles to the training data raised accuracy on new news to 83%.

**Fixing my own midterm limitation.** In the midterm I wrote that lower-cased text hurt entity recognition. This time
I found the original-casing BBC archive and measured the difference: 92% more entities.

**Messy multilingual web data.** Scraped pages contained cookie banners, page code and wrong language labels.
Boilerplate words formed their own "topics" until I filtered them, and my own language detector found 63 articles
with incorrect language labels.

**Honest results.** Some ideas did not help as much as I expected – for example, the ensemble classifier was not more
accurate than a simple linear SVM on clean data, and sentence-level sentiment improved agreement only slightly. I
reported these results as they were and explained why I still kept the components (calibrated confidence,
non-saturated scores).

**Limited hardware.** Transformer models could not run everywhere, so every transformer component has a classical
fallback; the full system runs on a normal laptop and improves automatically in Colab.

[[PERSONAL – question 1: which part was the hardest or most frustrating for you during the course/project, and how did you deal with it?]]

## 4. Lessons learned

* Evaluate on data that looks like the real use case, not only on the convenient test split.
* Data quality decisions (casing, duplicates, labels, boilerplate) changed results more than model choice.
* Calibrated confidence and a "needs review" flag make an NLP system more trustworthy than a slightly higher accuracy.
* Building with modules, tests and one reproducible experiment script made it possible to change things late without
  breaking everything.

[[PERSONAL – question 2: what did you learn most / what surprised you most in this class?]]

## 5. Use of AI tools

[[PERSONAL – question 3: how did you use AI tools (Claude, ChatGPT, others) during the midterm and final? Be specific and honest – this section is required.]]

## 6. Future improvements

* Fine-tune a multilingual transformer (XLM-RoBERTa) classifier on BBC plus recent multilingual news.
* Transformer-based NER and coreference resolution to improve the knowledge graph.
* A simple web interface (Streamlit or Flask) on top of the `NewsBot2` API, and live RSS feeds for real-time alerts.

[[PERSONAL – question 4: which feature would you add with more time, and why?]]

## 7. Closing reflection

[[2–3 sentences in your own words: how this project connects to what you want to do after the course.]]
