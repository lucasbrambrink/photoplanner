"""
1-Tap Crowd Ground-Truth Field Reports (`planner/field_reports.py`).

Implements Goal 4's Built-In Growth & Ground-Truth Loop:
  - Lets photographers tap "Epic / Partial / Bust" after golden hour right inside the app,
    recording what actually happened alongside PhotoPlanner's calibrated P50 light odds and
    fog-sea lid prediction.
  - Builds a real mountain-ridge & coastal-headland ground-truth dataset in places where
    no NOAA SURFRAD pyrheliometer exists, and powers live crowd verification statistics.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import sqlite3
import uuid
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FIELD_REPORTS_DB = ROOT_DIR / "data" / "field_reports.sqlite3"

VALID_OUTCOMES = ("epic", "partial", "bust")
VALID_PHENOMENA = ("cloud_sea", "alpenglow", "cirrus_burner", "mirror_lake", "astro", "fogged_in", "other")


def _init_db(db_path: Path | None = None) -> sqlite3.Connection:
    resolved = db_path or DEFAULT_FIELD_REPORTS_DB
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS field_reports (
            id TEXT PRIMARY KEY,
            spot_name TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            date TEXT NOT NULL,
            phase TEXT NOT NULL DEFAULT 'sunset',
            outcome TEXT NOT NULL,
            phenomenon TEXT NOT NULL DEFAULT 'other',
            predicted_p50_pct REAL,
            predicted_alpenglow_score REAL,
            predicted_fog_top_ft REAL,
            note TEXT,
            created_at_utc TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def submit_field_report(
    *,
    spot_name: str,
    lat: float,
    lon: float,
    date: str,
    phase: str = "sunset",
    outcome: str,
    phenomenon: str = "other",
    predicted_p50_pct: float | None = None,
    predicted_alpenglow_score: float | None = None,
    predicted_fog_top_ft: float | None = None,
    note: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Persists a 1-tap post-shoot photographer ground-truth report."""
    out_l = str(outcome or "").strip().lower()
    if out_l not in VALID_OUTCOMES:
        raise ValueError(f"`outcome` must be one of {VALID_OUTCOMES}.")
    ph_l = str(phase or "sunset").strip().lower()
    if ph_l not in ("sunrise", "sunset", "astro"):
        ph_l = "sunset"
    phen_l = str(phenomenon or "other").strip().lower()
    if phen_l not in VALID_PHENOMENA:
        phen_l = "other"
    if not str(spot_name or "").strip():
        raise ValueError("`spot_name` must be non-empty.")

    report_id = f"fr_{uuid.uuid4().hex[:10]}"
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    clean_note = str(note).strip()[:280] if note and str(note).strip() else None

    with _init_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO field_reports (
                id, spot_name, lat, lon, date, phase, outcome, phenomenon,
                predicted_p50_pct, predicted_alpenglow_score, predicted_fog_top_ft,
                note, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                spot_name.strip()[:120],
                float(lat),
                float(lon),
                date.strip(),
                ph_l,
                out_l,
                phen_l,
                float(predicted_p50_pct) if predicted_p50_pct is not None else None,
                float(predicted_alpenglow_score) if predicted_alpenglow_score is not None else None,
                float(predicted_fog_top_ft) if predicted_fog_top_ft is not None else None,
                clean_note,
                now_iso,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM field_reports WHERE id = ?", (report_id,)).fetchone()
    return dict(row)


def list_field_reports(
    *,
    spot_name: str | None = None,
    limit: int = 25,
    db_path: Path | None = None,
) -> list[dict[str, Any]]:
    lim = max(1, min(200, int(limit)))
    with _init_db(db_path) as conn:
        if spot_name and spot_name.strip():
            rows = conn.execute(
                "SELECT * FROM field_reports WHERE lower(spot_name) = lower(?) ORDER BY created_at_utc DESC LIMIT ?",
                (spot_name.strip(), lim),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM field_reports ORDER BY created_at_utc DESC LIMIT ?",
                (lim,),
            ).fetchall()
    return [dict(r) for r in rows]


def summarize_field_reports(
    *,
    spot_name: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregates crowd-verified outcomes & hit-rate vs predicted P50 light odds."""
    rows = list_field_reports(spot_name=spot_name, limit=200, db_path=db_path)
    counts = {"epic": 0, "partial": 0, "bust": 0}
    pred_by_outcome: dict[str, list[float]] = {"epic": [], "partial": [], "bust": []}
    for r in rows:
        o = r["outcome"]
        if o in counts:
            counts[o] += 1
            if r["predicted_p50_pct"] is not None:
                pred_by_outcome[o].append(float(r["predicted_p50_pct"]))

    total = len(rows)
    hit_rate_pct = round(100.0 * (counts["epic"] + counts["partial"]) / total, 1) if total > 0 else None
    mean_pred = {
        k: (round(sum(vals) / len(vals), 1) if vals else None)
        for k, vals in pred_by_outcome.items()
    }
    return {
        "spot_name": spot_name,
        "total_reports": total,
        "counts": counts,
        "positive_rate_pct": hit_rate_pct,
        "mean_predicted_p50_by_outcome": mean_pred,
        "recent": rows[:10],
    }
