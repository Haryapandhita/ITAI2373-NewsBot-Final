"""
Response generation (Module D).

Turns structured results from the NewsBot engine into helpful, natural
answers. Responses are template-based (fast, deterministic, grounded in the
data) and vary their wording; an optional local LLM can polish the final
text (see ``src.language_models.generator.OllamaClient``).
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import pandas as pd

HELP_TEXT = (
    "I can help you explore the news. Try for example:\n"
    "  • \"Show me positive tech news from this year\"\n"
    "  • \"Summarize politics news in Spanish\"\n"
    "  • \"How has sentiment about China changed over time?\"\n"
    "  • \"Compare Apple and Google\"  /  \"How are Trump and China connected?\"\n"
    "  • \"What topics are trending in business?\"\n"
    "  • \"Classify this article: <paste text>\"  /  \"Translate this: <text>\"\n"
    "  • \"Give me an overview\"  /  \"Export the results as csv\"\n"
    "Follow-ups work too: \"what about last year?\", \"only negative ones\", \"summarize them\"."
)

OPENERS = {
    "search": ["Here's what I found", "These articles match", "Top results"],
    "summarize": ["Here's the short version", "In a nutshell", "Summary"],
    "sentiment": ["Tone check", "Here's how the coverage feels", "Sentiment overview"],
}


class ResponseGenerator:
    """Format engine results as chatbot replies."""

    def __init__(self, llm=None, seed: int = 0):
        """Create the object; parameters are described in the class docstring."""
        self.llm = llm
        self.rng = random.Random(seed)

    def _opener(self, intent: str) -> str:
        return self.rng.choice(OPENERS.get(intent, ["Result"]))

    @staticmethod
    def describe_filters(slots: Dict[str, object]) -> str:
        """Human-readable description of the active filters ("positive tech news in es ...")."""
        bits = []
        if slots.get("sentiment"):
            bits.append(str(slots["sentiment"]))
        if slots.get("category"):
            bits.append("/".join(slots["category"]))
        bits.append("news")
        if slots.get("entities"):
            bits.append("about " + " & ".join(slots["entities"]))
        if slots.get("language"):
            bits.append("in " + "/".join(slots["language"]))
        if slots.get("date_from"):
            bits.append(f"from {slots['date_from']} to {slots.get('date_to', 'now')}")
        return " ".join(bits)

    # ---------------------------------------------------------- per intent
    def search(self, results: pd.DataFrame, slots: dict) -> str:
        """Numbered list of search results with metadata."""
        if results is None or results.empty:
            return (f"I couldn't find any {self.describe_filters(slots)}. "
                    "Try a wider time range or fewer filters.")
        lines = [f"{self._opener('search')} for {self.describe_filters(slots)} ({len(results)} shown):"]
        for i, (_, r) in enumerate(results.iterrows(), 1):
            date = str(r.get("published", ""))[:10] if pd.notna(r.get("published", None)) else ""
            meta = " · ".join(x for x in [r.get("category", ""), r.get("language", ""), date,
                                          r.get("sentiment_label", "")] if isinstance(x, str) and x)
            lines.append(f"  {i}. {r['title'][:100]}  [{meta}]")
        lines.append("You can ask me to summarize them, compare them, or show only positive/negative ones.")
        return "\n".join(lines)

    def summary(self, summary: str, n_articles: int, slots: dict) -> str:
        """Wrap a multi-document summary in a reply."""
        if not summary:
            return "There was nothing to summarise – try a broader request."
        return f"{self._opener('summarize')} ({n_articles} articles on {self.describe_filters(slots)}):\n{summary}"

    def sentiment(self, stats: dict, slots: dict) -> str:
        """Describe the tone of a set of articles."""
        if not stats or stats.get("n", 0) == 0:
            return f"No articles found for {self.describe_filters(slots)}."
        mood = "mostly positive" if stats["mean"] > 0.05 else "mostly negative" if stats["mean"] < -0.05 else "fairly neutral"
        subject = self.describe_filters({k: v for k, v in slots.items() if k != "sentiment"})
        txt = (f"{self._opener('sentiment')}: coverage of {subject} is {mood} "
               f"(average score {stats['mean']:+.2f} across {stats['n']} articles; "
               f"{stats['pos_share']:.0%} positive, {stats['neg_share']:.0%} negative).")
        if stats.get("top_emotion"):
            txt += f" The strongest emotion is {stats['top_emotion']}."
        return txt

    def compare(self, rows: List[dict]) -> str:
        """Side-by-side comparison of two or more groups/entities."""
        if not rows:
            return "I need two things to compare, e.g. \"compare Apple and Google\"."
        lines = ["Comparison:"]
        for r in rows:
            lines.append(f"  • {r['name']}: {r['n']} articles, average tone {r['mean']:+.2f}, "
                         f"{r['pos_share']:.0%} positive" + (f", mostly {r['top_category']}" if r.get("top_category") else ""))
        if len(rows) >= 2 and all(r["n"] for r in rows[:2]):
            a, b = rows[0], rows[1]
            more = a if a["n"] > b["n"] else b
            nicer = a if a["mean"] > b["mean"] else b
            lines.append(f"{more['name']} gets more coverage; {nicer['name']} gets the friendlier tone.")
        return "\n".join(lines)

    def topics(self, table: pd.DataFrame, emerging: Optional[pd.DataFrame] = None) -> str:
        """List the main (and fastest-growing) topics."""
        lines = ["Main topics:"]
        for _, r in table.head(6).iterrows():
            lines.append(f"  • {r['label']} – {r['prevalence']:.0%} of articles ({r['top_words'][:60]}...)")
        if emerging is not None and not emerging.empty:
            lines.append("Growing fastest: " + ", ".join(emerging.head(3)["topic"]))
        return "\n".join(lines)

    def trends(self, timeline: pd.DataFrame, subject: str) -> str:
        """Describe a sentiment timeline: direction, busiest period, recent values."""
        if timeline is None or timeline.empty:
            return f"Not enough dated articles to show a trend for {subject}."
        first, last = timeline.iloc[0], timeline.iloc[-1]
        diff = last["mean_sentiment"] - first["mean_sentiment"]
        direction = "roughly as positive as before" if abs(diff) < 0.03 else ("more positive" if diff > 0 else "more negative")
        peak = timeline.loc[timeline["n_articles"].idxmax()]
        lines = [f"Trend for {subject}: coverage became {direction} "
                 f"({first['period']}: {first['mean_sentiment']:+.2f} → {last['period']}: {last['mean_sentiment']:+.2f}).",
                 f"Busiest period: {peak['period']} with {int(peak['n_articles'])} articles."]
        lines += [f"  {r['period']}: {r['mean_sentiment']:+.2f} ({int(r['n_articles'])} articles)"
                  for _, r in timeline.tail(8).iterrows()]
        return "\n".join(lines)

    def entity(self, info: dict) -> str:
        """Describe an entity profile or the connection between two entities."""
        if "paths" in info:
            if not info.get("found"):
                return f"I couldn't connect them: {info.get('reason', 'no link found')}."
            if info.get("direct"):
                rel = ", ".join(f"{k.lower()} ({v}x)" for k, v in info.get("relations", {}).items())
                return (f"{info['paths'][0][0]} and {info['paths'][0][1]} appear together in {info['weight']} "
                        f"articles" + (f"; extracted relations: {rel}" if rel else "") + ".")
            paths = ["  " + " → ".join(p) for p in info["paths"]]
            return "They are linked indirectly:\n" + "\n".join(paths)
        if not info.get("found"):
            return "I don't know that entity yet – it does not appear in the analysed articles."
        cats = ", ".join(f"{k} ({v})" for k, v in sorted(info["categories"].items(), key=lambda kv: -kv[1])[:3])
        conns = ", ".join(n for n, _, _ in info["top_connections"])
        return (f"{info['name']} ({info['type']}) is mentioned in {info['mentions']} articles, mostly in {cats}. "
                f"Most often mentioned with: {conns}.")

    def classification(self, analysis: dict) -> str:
        """Compact reply for a full single-article analysis."""
        c = analysis["classification"]
        s = analysis["sentiment"]
        lines = [f"Category: {c['label']} (confidence {c['confidence']:.0%})"
                 + (f", also {', '.join(c['secondary_labels'])}" if c["secondary_labels"] else "")
                 + (" – flagged for human review" if c["needs_review"] else "")]
        lines.append(f"Tone: {s['label']} ({s['score']:+.2f})")
        if analysis.get("entities"):
            lines.append("Entities: " + ", ".join(sorted({e['canonical'] for e in analysis['entities']})[:8]))
        if analysis.get("topics"):
            lines.append("Topics: " + ", ".join(t["label"] for t in analysis["topics"]))
        if analysis.get("summary"):
            lines.append("Summary: " + analysis["summary"]["summary"])
        return "\n".join(lines)

    def translation(self, result: dict, source_language: str) -> str:
        """Reply with a translation (or explain why it is unavailable)."""
        if not result.get("translated"):
            return ("Translation models aren't available in this environment, so here is the original text "
                    f"(detected language: {source_language}). Run in Colab for full translation.")
        return f"Translation ({source_language} → en, {result['engine']}):\n{result['text']}"

    def insights(self, insights: List[str], interests: Optional[List[str]] = None) -> str:
        """Bullet list of corpus insights plus personalised suggestion."""
        txt = "Key insights:\n" + "\n".join(f"  • {i}" for i in insights)
        if interests:
            txt += f"\nBased on your questions you seem interested in {', '.join(interests)} – ask me for a focused report."
        return txt

    def export(self, path: str) -> str:
        """Confirm where an export was written."""
        return f"Done – I saved the report to {path}."

    def greeting(self, text: str) -> str:
        """Reply to greetings, thanks and goodbyes."""
        low = text.lower()
        if any(w in low for w in ("thank", "cheers")):
            return "You're welcome! Anything else you'd like to explore?"
        if any(w in low for w in ("bye", "see you")):
            return "Goodbye – happy reading!"
        return "Hi! I'm NewsBot 2.0. Ask me about news topics, sentiment, entities or trends (type 'help' for ideas)."

    def unknown(self) -> str:
        """Fallback reply with usage examples."""
        return "Sorry, I didn't quite get that. " + HELP_TEXT

    def finalize(self, text: str) -> str:
        """Optional LLM polishing (grounded – the draft contains all facts)."""
        if self.llm is not None and self.llm.available():
            return self.llm.rewrite(text)
        return text
