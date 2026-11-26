"""
Natural-language query understanding and conversation state (Module D).

``QueryProcessor.parse`` turns a user message into a structured
``ParsedQuery``: intent (from :class:`IntentClassifier`) + slots extracted
with rules and spaCy NER (category, sentiment, time range, language,
entities, number of results, summary length, export format, inline article
text). Multi-part questions ("show tech news and then summarize them") are
split into several parsed queries.

``ConversationContext`` keeps the dialogue state so follow-ups work:
"What about last month?" re-uses the previous intent and filters;
"summarize them" refers to the previous result set; "only negative ones"
narrows the previous search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.conversation.intent_classifier import IntentClassifier
from src.analysis.ner_extractor import normalize_entity

CATEGORY_WORDS = {
    "tech": ["tech", "technology", "technological", "science", "ai", "software", "gadget", "digital"],
    "business": ["business", "economy", "economic", "finance", "financial", "market", "markets", "stocks", "companies"],
    "politics": ["politics", "political", "government", "election", "elections", "policy", "parliament"],
    "sport": ["sport", "sports", "football", "soccer", "tennis", "cricket", "match", "matches"],
    "entertainment": ["entertainment", "film", "films", "movie", "movies", "music", "celebrity", "tv"],
}
SENTIMENT_WORDS = {
    "positive": ["positive", "good", "upbeat", "optimistic", "happy", "uplifting", "friendly", "favourable", "favorable"],
    "negative": ["negative", "bad", "gloomy", "pessimistic", "sad", "critical", "worrying", "hostile"],
}
LANGUAGE_WORDS = {"en": ["english"], "es": ["spanish", "español", "espanol"], "fr": ["french", "français"],
                  "de": ["german", "deutsch"], "it": ["italian", "italiano"], "pt": ["portuguese", "português"]}
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10, "a few": 3, "a couple": 2}
FOLLOW_UP = re.compile(r"^(and |also |what about|how about|only |just |now |then |same |those|them|these|"
                       r"what if|ok |okay )|\b(them|those|these|it|that one|the same)\b", re.IGNORECASE)
SPLIT_MULTI = re.compile(r"\s*(?:,?\s+and then\s+|;\s*|\s+then\s+|,\s*and also\s+|\.\s+also,?\s+)\s*", re.IGNORECASE)
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
STOP_ENTITIES = {"News", "Show", "Find", "Compare", "Summarize", "Summarise", "Translate", "Tell", "Who", "What",
                 "How", "Which", "Is", "Are", "Give", "Any", "English", "Spanish", "German", "French", "Italian",
                 "Portuguese", "NewsBot", "Thanks", "Hey", "Hi", "Hello", "Monthly", "Quick", "Make", "Download",
                 "Analyse", "Analyze", "Classify", "Did", "Has", "Could", "Can", "Pull", "Top", "Export", "Latest",
                 "Please", "Search", "List", "Explain", "Save", "Get", "Only", "Sum", "Brief", "Recap", "Create"}


@dataclass
class ParsedQuery:
    """Structured representation of one user request."""
    text: str
    intent: str
    confidence: float
    slots: Dict[str, object] = field(default_factory=dict)
    is_follow_up: bool = False

    def filters(self) -> dict:
        """Search filters derived from the slots."""
        f = {}
        if self.slots.get("category"):
            f["category"] = self.slots["category"]
        if self.slots.get("language"):
            f["language"] = self.slots["language"]
        if self.slots.get("sentiment"):
            f["sentiment_label"] = self.slots["sentiment"]
        if self.slots.get("date_from"):
            f["date_from"] = self.slots["date_from"]
        if self.slots.get("date_to"):
            f["date_to"] = self.slots["date_to"]
        return f


@dataclass
class ConversationContext:
    """Dialogue memory: previous queries, slots and result ids."""
    history: List[Tuple[str, str]] = field(default_factory=list)  # (role, text)
    last_query: Optional[ParsedQuery] = None
    last_result_ids: List[str] = field(default_factory=list)
    user_interests: Dict[str, int] = field(default_factory=dict)

    def remember(self, query: ParsedQuery, result_ids: Optional[List[str]] = None, response: str = "") -> None:
        """Store a parsed query, its result ids and the bot reply; update user interests."""
        self.history.append(("user", query.text))
        if response:
            self.history.append(("bot", response))
        self.last_query = query
        if result_ids is not None:
            self.last_result_ids = list(result_ids)
        for key in ("category", "language"):
            if query.slots.get(key):
                for v in query.slots[key]:
                    self.user_interests[v] = self.user_interests.get(v, 0) + 1
        for e in query.slots.get("entities", []):
            self.user_interests[e] = self.user_interests.get(e, 0) + 1

    def top_interests(self, n: int = 3) -> List[str]:
        """The ``n`` categories/languages/entities the user asked about most."""
        return [k for k, _ in sorted(self.user_interests.items(), key=lambda kv: -kv[1])[:n]]


class QueryProcessor:
    """Parse user messages into :class:`ParsedQuery` objects."""

    def __init__(self, intent_classifier: Optional[IntentClassifier] = None, nlp=None,
                 reference_date: Optional[pd.Timestamp] = None, known_entities: Optional[List[str]] = None):
        """Create the object; parameters are described in the class docstring."""
        self.intents = intent_classifier or IntentClassifier().train()
        self.nlp = nlp
        self.reference_date = pd.Timestamp(reference_date or pd.Timestamp.now(tz="UTC"))
        if self.reference_date.tzinfo is None:
            self.reference_date = self.reference_date.tz_localize("UTC")
        self.known_entities = {e.lower(): e for e in (known_entities or [])}

    # ------------------------------------------------------------- slots
    def extract_query_entities(self, query: str) -> Dict[str, object]:
        """All slots found in a query (see module docstring)."""
        q = query.lower()
        slots: Dict[str, object] = {}
        cats = [c for c, words in CATEGORY_WORDS.items() if any(re.search(rf"\b{w}\b", q) for w in words)]
        if cats:
            slots["category"] = cats
        for lab, words in SENTIMENT_WORDS.items():
            if any(re.search(rf"\b{w}\b", q) for w in words):
                slots["sentiment"] = lab
        langs = [c for c, words in LANGUAGE_WORDS.items() if any(re.search(rf"\b{w}\b", q) for w in words)]
        if langs:
            slots["language"] = langs
        slots.update(self._time_range(q))
        m = re.search(r"\b(?:top|first|last)?\s*(\d{1,2})\s+(?:articles|stories|results|headlines|news)\b", q)
        if m:
            slots["top_k"] = int(m.group(1))
        else:
            for w, n in NUMBER_WORDS.items():
                if re.search(rf"\b{w}\s+(articles|stories|results|headlines)\b", q):
                    slots["top_k"] = n
        if re.search(r"\b(brief|short|quick|tl;?dr|few sentences)\b", q):
            slots["length"] = "brief"
        elif re.search(r"\b(detailed|long|in depth|full)\b", q):
            slots["length"] = "detailed"
        fmt = re.search(r"\b(csv|json|pdf|markdown|md|html)\b", q)
        if fmt:
            slots["format"] = {"md": "markdown"}.get(fmt.group(1), fmt.group(1))
        inline = re.search(r"(?:article|text|this|story|headline)\s*:\s*(.+)$", query, re.IGNORECASE | re.DOTALL)
        if inline and len(inline.group(1).split()) >= 3:
            slots["article_text"] = inline.group(1).strip()
        ents = self._entities(query.split(":")[0] if "article_text" in slots else query)
        if ents:
            slots["entities"] = ents
        return slots

    def _entities(self, query: str) -> List[str]:
        found: List[str] = []
        low = query.lower()
        for key, canon in self.known_entities.items():  # longest known names first
            if len(key) > 2 and re.search(rf"(?<!\w){re.escape(key)}(?!\w)", low) and canon not in found:
                found.append(canon)
        if self.nlp is not None:
            for ent in self.nlp(query).ents:
                if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "NORP", "PRODUCT", "EVENT"):
                    name = normalize_entity(ent.text, ent.label_)
                    words = name.split()
                    while words and words[0] in STOP_ENTITIES:
                        words = words[1:]
                    name = self._known(" ".join(words))
                    if name and name not in found and name not in STOP_ENTITIES and \
                            name.lower() not in {w for ws in CATEGORY_WORDS.values() for w in ws}:
                        found.append(name)
        # capitalised words the NER missed (lower-case queries are common)
        for m in re.finditer(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", query):
            cand = m.group(1)
            if m.start() == 0 and cand.split()[0] in STOP_ENTITIES:
                cand = " ".join(cand.split()[1:])
            cand = self._known(normalize_entity(cand))
            if cand and cand not in STOP_ENTITIES and cand not in found and \
                    not any(cand in f for f in found) and cand.lower() not in {w for ws in CATEGORY_WORDS.values() for w in ws}:
                found.append(cand)
        # drop pieces contained in longer names
        return [f for f in found if not any(f != g and f in g for g in found)]

    def _known(self, name: str) -> str:
        """Map a surname or alias to a known full entity name ("Trump" -> "Donald Trump")."""
        low = name.lower()
        if low in self.known_entities:
            return self.known_entities[low]
        matches = [v for k, v in self.known_entities.items() if k.split()[-1] == low and " " in k]
        return matches[0] if len(matches) == 1 else name

    def _time_range(self, q: str) -> Dict[str, str]:
        ref = self.reference_date
        day = pd.Timedelta(days=1)
        rng = None
        if re.search(r"\btoday\b", q):
            rng = (ref - day, ref)
        elif re.search(r"\b(this|past|last) week\b", q):
            rng = (ref - 7 * day, ref)
        elif re.search(r"\bthis month\b", q):
            rng = (ref.replace(day=1), ref)
        elif re.search(r"\b(last|past) month\b", q):
            rng = (ref - 31 * day, ref)
        elif re.search(r"\b(last|past) (quarter|3 months|three months)\b", q):
            rng = (ref - 92 * day, ref)
        elif re.search(r"\bthis year\b", q):
            rng = (ref.replace(month=1, day=1), ref)
        elif re.search(r"\blast year\b", q):
            rng = (pd.Timestamp(ref.year - 1, 1, 1, tz="UTC"), pd.Timestamp(ref.year - 1, 12, 31, tz="UTC"))
        elif re.search(r"\bpast year\b|\blast 12 months\b", q):
            rng = (ref - 365 * day, ref)
        elif re.search(r"\b(recent|recently|lately|latest|newest)\b", q):
            rng = (ref - 92 * day, ref)
        m = re.search(r"\b(?:in|during|from)?\s*(q[1-4])\s*(20\d\d)\b", q)
        if m:
            start = pd.Period(f"{m.group(2)}{m.group(1).upper()}", "Q").start_time.tz_localize("UTC")
            rng = (start, start + pd.offsets.QuarterEnd(0))
        else:
            m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(20\d\d)\b", q)
            if m:
                start = pd.Timestamp(int(m.group(2)), MONTHS[m.group(1)], 1, tz="UTC")
                rng = (start, start + pd.offsets.MonthEnd(0))
            else:
                m = re.search(r"\b(?:in|during|from|since)\s+(20\d\d)\b", q) or re.search(r"\b(20[12]\d)\b", q)
                if m:
                    year = int(m.group(1))
                    rng = (pd.Timestamp(year, 1, 1, tz="UTC"),
                           ref if "since" in q else pd.Timestamp(year, 12, 31, tz="UTC"))
        if not rng:
            return {}
        return {"date_from": str(rng[0].date()), "date_to": str(pd.Timestamp(rng[1]).date())}

    # ------------------------------------------------------------- parse
    def parse(self, query: str, context: Optional[ConversationContext] = None) -> List[ParsedQuery]:
        """Parse a (possibly multi-part) message into one or more queries."""
        parts = [p for p in SPLIT_MULTI.split(query.strip()) if p and len(p.split()) >= 1]
        if "article_text" in self.extract_query_entities(query):  # don't split pasted articles
            parts = [query.strip()]
        parsed: List[ParsedQuery] = []
        prev = context.last_query if context else None
        for i, part in enumerate(parts):
            res = self.intents.classify_intent(part)
            slots = self.extract_query_entities(part)
            follow = bool(FOLLOW_UP.search(part)) and (prev is not None or i > 0)
            intent, conf = res["intent"], res["confidence"]
            if follow:
                base = parsed[-1] if i > 0 else prev
                if base is not None:
                    # inherit missing slots from the previous turn
                    merged = {k: v for k, v in base.slots.items() if k not in ("article_text",)}
                    merged.update(slots)
                    slots = merged
                    # "what about last month?" -> keep the previous intent
                    if (re.match(r"^(and |what about|how about|only |just )", part, re.IGNORECASE)
                            or conf < 0.5 or intent in ("unknown", "greeting")):
                        intent = base.intent
            parsed.append(ParsedQuery(part, intent, conf, slots, follow))
        return parsed
