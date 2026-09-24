"""Result/figure saving helpers and shared project paths."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_json(name: str, payload: Dict[str, Any], subdir: str = "") -> Path:
    ensure_dirs()
    out = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_serializable(payload), fh, indent=2)
    return path


def save_csv(name: str, rows: Iterable[Dict[str, Any]], subdir: str = "") -> Path:
    import pandas as pd

    ensure_dirs()
    out = RESULTS_DIR / subdir if subdir else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def save_fig(fig, name: str, subdir: str = "", formats=("png",)) -> Path:
    ensure_dirs()
    out = FIGURES_DIR / subdir if subdir else FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    first: Path | None = None
    for ext in formats:
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        if first is None:
            first = path
    return first


def manifest(experiment: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "experiment": experiment,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        out.update(extra)
    return out
