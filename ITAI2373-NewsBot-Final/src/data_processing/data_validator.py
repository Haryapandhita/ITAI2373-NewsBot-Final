"""Data-quality checks run before any model is trained."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


@dataclass
class ValidationReport:
    """Result of :func:`validate_corpus` – printable and JSON-serialisable."""
    n_rows: int
    issues: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no ERROR-level issue was found."""
        return not any(i.startswith("ERROR") for i in self.issues)

    def to_dict(self) -> dict:
        """Plain ``dict`` representation (JSON-friendly)."""
        return {"n_rows": self.n_rows, "ok": self.ok, "issues": self.issues, "stats": self.stats}

    def __str__(self) -> str:
        lines = [f"Rows: {self.n_rows}  |  status: {'OK' if self.ok else 'PROBLEMS'}"]
        lines += [f"  - {i}" for i in self.issues] or ["  - no issues found"]
        return "\n".join(lines)


REQUIRED = ["doc_id", "text", "category", "language"]


def validate_corpus(df: pd.DataFrame, min_words: int = 40, max_imbalance: float = 3.0) -> ValidationReport:
    """
    Check a corpus DataFrame for the problems that hurt NLP models.

    Checks: required columns, empty/short texts, duplicate ids and texts,
    class imbalance ratio, languages present, lower-cased text (which breaks
    NER – the main midterm limitation) and missing dates.
    """
    rep = ValidationReport(n_rows=len(df))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        rep.issues.append(f"ERROR missing columns: {missing}")
        return rep

    words = df["text"].fillna("").str.split().str.len()
    rep.stats["median_words"] = int(words.median()) if len(df) else 0
    rep.stats["min_words"] = int(words.min()) if len(df) else 0
    short = int((words < min_words).sum())
    if short:
        rep.issues.append(f"WARNING {short} texts shorter than {min_words} words")
    if df["text"].isna().any() or (df["text"].str.strip() == "").any():
        rep.issues.append("ERROR empty texts present")
    if df["doc_id"].duplicated().any():
        rep.issues.append(f"ERROR {int(df['doc_id'].duplicated().sum())} duplicate doc_ids")
    dup_text = int(df["text"].duplicated().sum())
    if dup_text:
        rep.issues.append(f"WARNING {dup_text} duplicate texts")

    counts = df["category"].value_counts()
    rep.stats["category_counts"] = counts.to_dict()
    ratio = counts.max() / max(counts.min(), 1)
    rep.stats["imbalance_ratio"] = round(float(ratio), 2)
    if ratio > max_imbalance:
        rep.issues.append(f"WARNING class imbalance ratio {ratio:.2f} > {max_imbalance}")

    rep.stats["languages"] = df["language"].value_counts().to_dict()
    sample = df["text"].head(200).fillna("")
    lower_share = float((sample == sample.str.lower()).mean()) if len(sample) else 0.0
    rep.stats["lowercased_share"] = round(lower_share, 3)
    if lower_share > 0.9:
        rep.issues.append("WARNING text is lower-cased – NER quality will suffer")

    if "published" in df.columns:
        rep.stats["date_range"] = [str(df["published"].min())[:10], str(df["published"].max())[:10]]
        if df["published"].isna().any():
            rep.issues.append(f"WARNING {int(df['published'].isna().sum())} rows without a date")
    return rep
