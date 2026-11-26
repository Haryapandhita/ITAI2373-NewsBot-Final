"""
Dataset loading for NewsBot 2.0.

Two public sources are used (neither is redistributed in this repository –
they are downloaded on first use into ``data/raw/``):

1. **BBC News** (Greene & Cunningham, 2006) – 2,225 English articles, five
   labelled categories, original casing and headlines. Used to train and
   evaluate the classifier, NER and summariser.
2. **Webz.io free news datasets** – multilingual, *dated* news (Dec 2023 –
   Nov 2026) with category and sentiment metadata. Used for topic/sentiment
   evolution over time, the multilingual module and cross-domain evaluation.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from config.settings import DEFAULT_CONFIG, NewsBotConfig
from src.utils.logging_utils import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- BBC
def _download_bbc(cfg: NewsBotConfig) -> Path:
    """Fetch the BBC archive with git (preferred) or the GitHub zip endpoint."""
    target = cfg.raw_dir / "bbc"
    if (target / "business").exists():
        return target
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg.raw_dir / "_bbc_clone"
    try:
        if tmp.exists():
            shutil.rmtree(tmp)
        subprocess.run(["git", "clone", "--depth", "1", "-q", cfg.bbc_git_url, str(tmp)],
                       check=True, timeout=300)
        target.mkdir(exist_ok=True)
        for cat in cfg.categories:
            shutil.copytree(tmp / cat, target / cat, dirs_exist_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("BBC dataset cloned into %s", target)
    except Exception as exc:  # git missing or blocked -> try the zip endpoint
        log.warning("git clone failed (%s); trying zip download", exc)
        with urllib.request.urlopen(cfg.bbc_zip_url, timeout=300) as resp:
            archive = zipfile.ZipFile(io.BytesIO(resp.read()))
        target.mkdir(exist_ok=True)
        for member in archive.namelist():
            parts = member.split("/")
            if len(parts) >= 3 and parts[1] in cfg.categories and parts[2].endswith(".txt"):
                (target / parts[1]).mkdir(exist_ok=True)
                (target / parts[1] / parts[2]).write_bytes(archive.read(member))
    return target


def load_bbc(cfg: NewsBotConfig = DEFAULT_CONFIG, deduplicate: bool = True) -> pd.DataFrame:
    """
    Load the BBC News corpus as a DataFrame.

    Returns columns: ``doc_id, title, text, category, source, language``.
    The first line of every file is the headline, the rest is the body.
    """
    folder = _download_bbc(cfg)
    rows = []
    for cat in cfg.categories:
        for path in sorted((folder / cat).glob("*.txt")):
            raw = path.read_bytes().decode("utf-8", errors="replace").strip()
            title, _, body = raw.partition("\n")
            rows.append({
                "doc_id": f"bbc_{cat}_{path.stem}",
                "title": title.strip(),
                "text": re.sub(r"\n\s*\n", "\n", body).strip(),
                "category": cat,
                "source": "bbc",
                "language": "en",
            })
    df = pd.DataFrame(rows)
    if deduplicate:
        before = len(df)
        df = df.drop_duplicates(subset="text").reset_index(drop=True)
        log.info("BBC: %d articles (%d exact duplicates removed)", len(df), before - len(df))
    return df


# -------------------------------------------------------------------------- Webz
def _download_webz_snapshot(name: str, cfg: NewsBotConfig) -> Path:
    folder = cfg.raw_dir / "webz" / name.replace(".zip", "")
    if folder.exists() and any(folder.rglob("*.json")):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    url = cfg.webz_raw_base + urllib.parse.quote(name)
    log.info("Downloading Webz snapshot %s", name)
    with urllib.request.urlopen(url, timeout=300) as resp:
        zipfile.ZipFile(io.BytesIO(resp.read())).extractall(folder)
    return folder


def _clean_scraped(text: str) -> str:
    text = re.sub(r"https?://\S+", " ", text or "")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def load_webz(cfg: NewsBotConfig = DEFAULT_CONFIG,
              snapshots: Optional[Iterable[str]] = None,
              languages: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Load (and download if needed) the Webz.io multilingual news snapshots.

    Returns columns: ``doc_id, title, text, category, source, language,
    published, month, site, country, webz_sentiment, snapshot``.
    Only languages listed in ``cfg.languages`` are kept (ISO codes).
    """
    snapshots = list(snapshots or cfg.webz_snapshots)
    keep = set(languages or cfg.languages.keys())
    name_to_iso = {v: k for k, v in cfg.languages.items()}
    rows = []
    for snap in snapshots:
        folder = _download_webz_snapshot(snap, cfg)
        category = next((v for k, v in cfg.webz_category_map.items() if snap.startswith(k)), "other")
        for path in folder.rglob("*.json"):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            iso = name_to_iso.get((d.get("language") or "").lower())
            if iso is None or iso not in keep:
                continue
            text = _clean_scraped(d.get("text") or "")[: cfg.max_chars]
            if len(text.split()) < cfg.min_words:
                continue
            thread = d.get("thread") or {}
            published = pd.to_datetime(d.get("published"), errors="coerce", utc=True)
            rows.append({
                "doc_id": "webz_" + (d.get("uuid") or hashlib.md5(text.encode()).hexdigest())[:16],
                "title": (d.get("title") or "").strip(),
                "text": text,
                "category": category,
                "source": "webz",
                "language": iso,
                "published": published,
                "site": thread.get("site"),
                "country": thread.get("country"),
                "webz_sentiment": d.get("sentiment"),
                "snapshot": snap.replace(".zip", ""),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    before = len(df)
    df = df.drop_duplicates(subset="text").drop_duplicates(subset=["title", "site"])
    df = df.dropna(subset=["published"]).reset_index(drop=True)
    df["month"] = df["published"].dt.tz_convert(None).dt.to_period("M").astype(str)
    log.info("Webz: %d articles kept (%d duplicates/undated removed) in %d languages",
             len(df), before - len(df), df["language"].nunique())
    return df


def load_all(cfg: NewsBotConfig = DEFAULT_CONFIG, cache: bool = True) -> dict:
    """Load both corpora, caching them as pickles in ``data/processed``."""
    cfg.ensure_dirs()
    out = {}
    for name, loader in (("bbc", load_bbc), ("webz", load_webz)):
        cache_path = cfg.processed_dir / f"{name}.pkl"
        if cache and cache_path.exists():
            out[name] = pd.read_pickle(cache_path)
        else:
            out[name] = loader(cfg)
            out[name].to_pickle(cache_path)
    return out


def sample_articles() -> List[dict]:
    """A few short hand-written articles used in demos and unit tests."""
    return [
        {"title": "Tech giant unveils new AI chip",
         "text": "Apple announced a new artificial intelligence chip on Tuesday in Cupertino. "
                 "Chief executive Tim Cook said the processor would make iPhones faster and "
                 "more energy efficient. Analysts at Morgan Stanley expect strong sales."},
        {"title": "Government faces vote on budget",
         "text": "The Prime Minister faces a crucial vote in Parliament on Wednesday after "
                 "opposition parties criticised the budget. The Chancellor defended tax rises, "
                 "saying they were needed to fund the health service."},
        {"title": "United win dramatic derby",
         "text": "Manchester United beat Manchester City 2-1 at Old Trafford on Sunday. "
                 "Marcus Rashford scored a late winner after a tense second half, lifting "
                 "the team to fourth place in the Premier League table."},
        {"title": "El banco central sube los tipos",
         "text": "El Banco Central Europeo subió los tipos de interés el jueves para frenar la "
                 "inflación. La presidenta Christine Lagarde dijo que la economía sigue fuerte "
                 "pero que los precios de la energía preocupan a los mercados."},
    ]
