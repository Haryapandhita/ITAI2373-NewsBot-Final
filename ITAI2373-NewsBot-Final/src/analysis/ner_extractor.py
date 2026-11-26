"""
Named entity recognition, relation extraction and knowledge graph (Module A).

* spaCy ``en_core_web_sm`` for English, ``xx_ent_wiki_sm`` for other
  languages (multilingual NER);
* entity normalisation / linking of aliases (``UK`` = ``Britain`` =
  ``United Kingdom``; ``Mr Blair`` = ``Tony Blair``);
* relation extraction from the dependency parse: subject–verb–object
  triples, appositions ("Tim Cook, chief executive of Apple" -> ROLE_AT) and
  prepositional patterns ("X of Y", "X in Y");
* a ``networkx`` knowledge graph with co-occurrence and typed relation edges
  plus path search between two entities ("How are Apple and Google connected?").
"""

from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

from src.utils.logging_utils import get_logger

log = get_logger(__name__)

FOCUS_TYPES = {"PERSON", "ORG", "GPE", "LOC", "NORP", "EVENT", "PRODUCT", "MONEY", "DATE", "PER", "MISC"}
GRAPH_TYPES = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "PER"}

ALIASES = {
    "uk": "United Kingdom", "u.k.": "United Kingdom", "u.k": "United Kingdom", "u.s": "United States", "britain": "United Kingdom",
    "great britain": "United Kingdom", "the united kingdom": "United Kingdom",
    "us": "United States", "u.s.": "United States", "usa": "United States", "america": "United States",
    "the united states": "United States", "the us": "United States",
    "eu": "European Union", "the european union": "European Union",
    "un": "United Nations", "the un": "United Nations",
    # multilingual aliases so entities can be compared across languages
    "estados unidos": "United States", "ee.uu.": "United States", "eeuu": "United States", "ee. uu.": "United States",
    "états-unis": "United States", "etats-unis": "United States", "vereinigte staaten": "United States",
    "usa.": "United States", "stati uniti": "United States", "eua": "United States",
    "reino unido": "United Kingdom", "royaume-uni": "United Kingdom", "großbritannien": "United Kingdom",
    "regno unito": "United Kingdom", "alemania": "Germany", "deutschland": "Germany", "allemagne": "Germany",
    "germania": "Germany", "alemanha": "Germany", "francia": "France", "frankreich": "France", "frança": "France",
    "españa": "Spain", "spanien": "Spain", "espagne": "Spain", "spagna": "Spain", "espanha": "Spain",
    "italia": "Italy", "italien": "Italy", "italie": "Italy", "itália": "Italy",
    "china": "China", "chine": "China", "cina": "China", "rusia": "Russia", "russland": "Russia",
    "russie": "Russia", "rússia": "Russia", "ucrania": "Ukraine", "ukraine": "Ukraine", "ucraina": "Ukraine",
    "ucrânia": "Ukraine", "unión europea": "European Union", "union européenne": "European Union",
    "europäische union": "European Union", "unione europea": "European Union", "ue": "European Union",
    "união europeia": "European Union", "brasil": "Brazil", "brésil": "Brazil", "brasilien": "Brazil",
    "méxico": "Mexico", "mexique": "Mexico", "japón": "Japan", "japon": "Japan", "giappone": "Japan",
    "india": "India", "inde": "India", "indien": "India", "israel": "Israel", "israël": "Israel",
    "otan": "NATO", "nato": "NATO", "onu": "United Nations", "uno": "United Nations",
    "donald trump": "Donald Trump", "trump": "Donald Trump", "joe biden": "Joe Biden", "biden": "Joe Biden",
    "elon musk": "Elon Musk", "musk": "Elon Musk", "putin": "Vladimir Putin", "vladimir putin": "Vladimir Putin",
    "wladimir putin": "Vladimir Putin", "poutine": "Vladimir Putin", "xi jinping": "Xi Jinping",
}
LABEL_MAP = {"PER": "PERSON", "LOC": "GPE", "MISC": "MISC", "ORG": "ORG"}
TITLES = re.compile(r"^(mr|mrs|ms|dr|prof|sir|lord|president|prime minister|chancellor|minister)\.?\s+",
                    re.IGNORECASE)
ROLE_WORDS = {"chief", "executive", "ceo", "chairman", "president", "minister", "secretary", "boss",
              "head", "director", "founder", "leader", "manager", "coach", "spokesman", "spokeswoman",
              "chancellor", "governor", "captain", "editor", "analyst", "economist"}


def normalize_entity(text: str, label: str = "") -> str:
    """Canonical form of an entity string (aliases, titles, possessives, whitespace)."""
    t = re.sub(r"\s+", " ", text).strip().strip(".,;:\"'()")
    t = re.sub(r"'s$|’s$", "", t)
    t = TITLES.sub("", t) if label in ("PERSON", "PER", "") else t
    key = t.lower()
    if key in ALIASES:
        return ALIASES[key]
    if key.startswith("the ") and label in ("ORG", "GPE"):
        t = t[4:]
    return t


class EntityRelationshipMapper:
    """NER + relations + knowledge graph."""

    def __init__(self, model: str = "en_core_web_sm", multilingual_model: str = "xx_ent_wiki_sm"):
        """Create the object; parameters are described in the class docstring."""
        import spacy
        self.nlp = spacy.load(model)
        self._multi_name = multilingual_model
        self._multi = None
        self.graph = nx.Graph()
        self._surname_map: Dict[str, str] = {}

    def _multilingual(self):
        if self._multi is None:
            import spacy
            try:
                self._multi = spacy.load(self._multi_name)
            except OSError:
                log.warning("%s not installed – multilingual NER disabled", self._multi_name)
                self._multi = False
        return self._multi

    # ------------------------------------------------------------ entities
    def extract_entities(self, text: str, language: str = "en") -> List[dict]:
        """Entities with canonical name, type, character offsets and sentence."""
        nlp = self.nlp if language == "en" else self._multilingual()
        if not nlp:
            return []
        doc = nlp(text[:100000])
        return self._entities_from_doc(doc)

    def _entities_from_doc(self, doc) -> List[dict]:
        out = []
        self._surname_map = {}  # surname linking is scoped to one document
        for ent in doc.ents:
            if ent.label_ not in FOCUS_TYPES:
                continue
            label = LABEL_MAP.get(ent.label_, ent.label_)
            canon = self._link_person(normalize_entity(ent.text, label), label)
            if len(canon) < 2:
                continue
            out.append({"text": ent.text, "canonical": canon, "label": label,
                        "start": ent.start_char, "end": ent.end_char,
                        "sentence": ent.sent.text.strip() if doc.has_annotation("SENT_START") else ""})
        return out

    def corpus_entities(self, texts, languages=None, batch_size: int = 64, max_chars: int = 1500,
                        types=("PERSON", "ORG", "GPE", "LOC", "NORP", "PRODUCT", "EVENT")) -> list:
        """
        Canonical entity list for every text (fast: uses the first ``max_chars``
        characters, i.e. headline + lead, where news names its key actors).
        English uses ``en_core_web_sm``; other languages ``xx_ent_wiki_sm``.
        """
        texts = [t[:max_chars] for t in texts]
        langs = list(languages) if languages is not None else ["en"] * len(texts)
        out = [None] * len(texts)
        for is_en in (True, False):
            idx = [i for i, l in enumerate(langs) if (l == "en") == is_en]
            nlp = self.nlp if is_en else self._multilingual()
            if not idx or not nlp:
                for i in idx:
                    out[i] = []
                continue
            for i, doc in zip(idx, nlp.pipe((texts[i] for i in idx), batch_size=batch_size)):
                ents = []
                for e in self._entities_from_doc(doc):
                    if e["label"] in types and e["canonical"] not in ents:
                        ents.append(e["canonical"])
                out[i] = ents
        return out

    def _link_person(self, name: str, label: str) -> str:
        """Within-document linking: a bare surname maps to the full name used earlier."""
        if label not in ("PERSON", "PER"):
            return name
        parts = name.split()
        if len(parts) >= 2:
            self._surname_map[parts[-1].lower()] = name
            return name
        return self._surname_map.get(name.lower(), name)

    # ----------------------------------------------------------- relations
    def extract_relationships(self, text: str) -> List[dict]:
        """Typed relations between entities found in the dependency parse."""
        doc = self.nlp(text[:100000])
        return self._relations_from_doc(doc)

    def _relations_from_doc(self, doc) -> List[dict]:
        rels = []
        ent_of = {}
        for ent in doc.ents:
            if ent.label_ in GRAPH_TYPES:
                canon = normalize_entity(ent.text, ent.label_)
                for tok in ent:
                    ent_of[tok.i] = (canon, ent.label_)

        def entity_for(token) -> Optional[Tuple[str, str]]:
            """Entity (canonical name, label) covering ``token`` or its subtree, if any."""
            if token.i in ent_of:
                return ent_of[token.i]
            for child in token.subtree:
                if child.i in ent_of:
                    return ent_of[child.i]
            return None

        for sent in doc.sents:
            for tok in sent:
                # 1) SVO: entity subject -- verb -- entity object
                if tok.pos_ == "VERB":
                    subj = [c for c in tok.children if c.dep_ in ("nsubj", "nsubjpass")]
                    objs = [c for c in tok.children if c.dep_ in ("dobj", "attr", "pobj")]
                    objs += [g for c in tok.children if c.dep_ == "prep" for g in c.children if g.dep_ == "pobj"]
                    for s in subj:
                        es = entity_for(s)
                        for o in objs:
                            eo = entity_for(o)
                            if es and eo and es[0] != eo[0]:
                                rels.append({"source": es[0], "relation": tok.lemma_.upper(),
                                             "target": eo[0], "sentence": sent.text.strip()})
                # 2) apposition with a role word: "Tim Cook, chief executive of Apple"
                if tok.dep_ == "appos" and tok.lemma_.lower() in ROLE_WORDS | {"executive", "officer"}:
                    head_ent = entity_for(tok.head)
                    org = None
                    for prep in [c for c in tok.subtree if c.dep_ == "prep" and c.lower_ in ("of", "at")]:
                        for pobj in prep.children:
                            org = org or entity_for(pobj)
                    if head_ent and org and head_ent[1] in ("PERSON", "PER"):
                        rels.append({"source": head_ent[0], "relation": "ROLE_AT", "target": org[0],
                                     "role": tok.lemma_.lower(), "sentence": sent.text.strip()})
                # 3) compound role before a name: "Microsoft boss Bill Gates"
                if tok.dep_ == "compound" and tok.i in ent_of and ent_of[tok.i][1] == "ORG":
                    head = tok.head
                    if head.lemma_.lower() in ROLE_WORDS:
                        person = entity_for(head.head) if head.head.i in ent_of else None
                        if person and person[1] in ("PERSON", "PER"):
                            rels.append({"source": person[0], "relation": "ROLE_AT",
                                         "target": ent_of[tok.i][0], "role": head.lemma_.lower(),
                                         "sentence": sent.text.strip()})
        # deduplicate
        seen, uniq = set(), []
        for r in rels:
            key = (r["source"], r["relation"], r["target"])
            if key not in seen:
                seen.add(key)
                uniq.append(r)
        return uniq

    # ---------------------------------------------------- knowledge graph
    def build_knowledge_graph(self, texts: Iterable[str], categories: Optional[Iterable[str]] = None,
                              batch_size: int = 32, min_cooccurrence: int = 2) -> nx.Graph:
        """
        Build an entity graph from many articles. Nodes carry type, mention
        count and category counts; edges carry co-occurrence weight and any
        typed relations (ROLE_AT, verbs) found between the two entities.
        """
        texts = list(texts)
        cats = list(categories) if categories is not None else [None] * len(texts)
        node_count, node_type, node_cats = Counter(), defaultdict(Counter), defaultdict(Counter)
        edge_w, edge_rel = Counter(), defaultdict(Counter)
        for doc, cat in zip(self.nlp.pipe((t[:20000] for t in texts), batch_size=batch_size), cats):
            ents = {e["canonical"]: e["label"] for e in self._entities_from_doc(doc)
                    if e["label"] in GRAPH_TYPES}
            for name, lab in ents.items():
                node_count[name] += 1
                node_type[name][lab] += 1  # majority vote over all mentions
                if cat:
                    node_cats[name][cat] += 1
            for a, b in itertools.combinations(sorted(ents), 2):
                edge_w[(a, b)] += 1
            for r in self._relations_from_doc(doc):
                a, b = sorted((r["source"], r["target"]))
                edge_rel[(a, b)][r["relation"]] += 1
                edge_w[(a, b)] += 1
        g = nx.Graph()
        for (a, b), w in edge_w.items():
            if w >= min_cooccurrence or (a, b) in edge_rel:
                for n in (a, b):
                    if n not in g:
                        types = node_type.get(n)
                        g.add_node(n, type=types.most_common(1)[0][0] if types else "?", mentions=node_count[n],
                                   categories=dict(node_cats[n]))
                g.add_edge(a, b, weight=w, relations=dict(edge_rel.get((a, b), {})))
        self.graph = g
        log.info("Knowledge graph: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())
        return g

    def find_entity_connections(self, entity1: str, entity2: str, max_paths: int = 3) -> dict:
        """Shortest paths between two entities, preferring strong edges."""
        g = self.graph
        a, b = self.resolve(entity1), self.resolve(entity2)
        if a is None or b is None:
            return {"found": False, "reason": f"unknown entity: {entity1 if a is None else entity2}"}
        if g.has_edge(a, b):
            data = g.edges[a, b]
            return {"found": True, "direct": True, "paths": [[a, b]], "weight": data["weight"],
                    "relations": data.get("relations", {})}
        try:
            # cost = 1/weight so frequent co-occurrence is a "shorter" link
            for u, v, d in g.edges(data=True):
                d["cost"] = 1.0 / d["weight"]
            paths = list(itertools.islice(nx.shortest_simple_paths(g, a, b, weight="cost"), max_paths))
            return {"found": True, "direct": False, "paths": paths}
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {"found": False, "reason": "no connection in the graph"}

    def resolve(self, name: str) -> Optional[str]:
        """Match user input to a graph node (canonical form, then case-insensitive, then substring)."""
        canon = normalize_entity(name)
        if canon in self.graph:
            return canon
        low = {n.lower(): n for n in self.graph.nodes}
        if canon.lower() in low:
            return low[canon.lower()]
        candidates = [n for n in self.graph.nodes if canon.lower() in n.lower()]
        return max(candidates, key=lambda n: self.graph.nodes[n]["mentions"]) if candidates else None

    def entity_profile(self, name: str, k: int = 5) -> dict:
        """Background for an entity: type, mentions, categories, strongest neighbours."""
        node = self.resolve(name)
        if node is None:
            return {"found": False}
        data = self.graph.nodes[node]
        nbrs = sorted(self.graph[node].items(), key=lambda kv: -kv[1]["weight"])[:k]
        return {"found": True, "name": node, "type": data["type"], "mentions": data["mentions"],
                "categories": data["categories"],
                "top_connections": [(n, d["weight"], d.get("relations", {})) for n, d in nbrs]}

    def top_entities(self, n: int = 20, types: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Most-mentioned graph entities as a DataFrame (optionally filtered by type)."""
        rows = [{"entity": k, **{x: v for x, v in d.items() if x != "categories"},
                 "degree": self.graph.degree(k)} for k, d in self.graph.nodes(data=True)
                if not types or d["type"] in set(types)]
        return pd.DataFrame(rows).sort_values("mentions", ascending=False).head(n).reset_index(drop=True)
