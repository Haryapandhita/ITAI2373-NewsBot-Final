"""Evaluation helpers shared by notebooks, tests and the experiment script."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def to_jsonable(obj: Any) -> Any:
    """Convert numpy / pandas objects so ``json.dump`` accepts them."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict(orient="records"))
    if isinstance(obj, pd.Series):
        return to_jsonable(obj.to_dict())
    if isinstance(obj, (pd.Timestamp, pd.Period)):
        return str(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def save_json(data: Dict, path: Path) -> Path:
    """Write ``data`` as pretty JSON (numpy/pandas values converted)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(path: Path) -> Dict:
    """Read a JSON file into a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


@contextmanager
def timer(store: Dict, key: str):
    """``with timer(results, 'train_s'):`` – records elapsed seconds."""
    start = time.time()
    yield
    store[key] = round(time.time() - start, 2)


def per_class_table(report: Dict) -> pd.DataFrame:
    """sklearn ``classification_report(output_dict=True)`` -> tidy DataFrame."""
    rows = [{"class": k, **{m: round(v, 4) for m, v in d.items()}} for k, d in report.items()
            if isinstance(d, dict) and k not in ("macro avg", "weighted avg")]
    return pd.DataFrame(rows)
