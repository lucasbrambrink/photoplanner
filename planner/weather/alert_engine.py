"""
Proactive Calibrated Light-Alert & Watchlist Engine (`planner/weather/alert_engine.py`).

Implements Roadmap Phase 4 (Task 4.3):
  - Persists user saved spots & regional watchlists in lightweight SQLite (`data/watchlists.sqlite3`).
  - Gates 24h / 3h proactive alerts on **out-of-sample calibrated probabilities** rather than
    uncalibrated raw scores — preventing photographer alert fatigue:
      * `TRIGGER`: Calibrated `P50` transmittance, fog-breakout probability, lake Mirror Index,
        and phenomenon score ALL clear user thresholds (high-confidence GO).
      * `WATCH`: Calibrated `P90` upside clears threshold while `P50` is still uncertain
        (useful at 24h lead to pack gear & re-check at 3h nowcast).
      * `SUPPRESS`: Calibrated probability falls below threshold — returns the exact plain-English
        **Causal Bust Attribution** explaining *why* the alert was withheld.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .corridor_solver import _find_sunrise_sunset_golden_hours, solve_weather_and_corridor
from .nwp_corridor import local_tz_for_lon
from .regional_hunter import REGIONAL_SPOT_CATALOG, classify_spot_phenomena_and_attribution

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WATCHLIST_DB = ROOT_DIR / "data" / "watchlists.sqlite3"


def _init_db(db_path: Path = DEFAULT_WATCHLIST_DB) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlists (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            spot_id TEXT,
            region TEXT,
            lat REAL,
            lon REAL,
            elev_m REAL,
            phase TEXT NOT NULL DEFAULT 'sunset',
            target_phenomena_json TEXT NOT NULL DEFAULT '[]',
            min_calibrated_p50_pct REAL NOT NULL DEFAULT 50.0,
            min_above_fog_prob_pct REAL NOT NULL DEFAULT 0.0,
            min_mirror_index REAL NOT NULL DEFAULT 0.0,
            min_opportunity_score REAL NOT NULL DEFAULT 65.0,
            lead_window TEXT NOT NULL DEFAULT '24h',
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at_utc TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def create_watchlist_rule(
    *,
    name: str,
    spot_id: str | None = None,
    region: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    elev_m: float | None = None,
    phase: str = "sunset",
    target_phenomena: list[str] | None = None,
    min_calibrated_p50_pct: float = 50.0,
    min_above_fog_prob_pct: float = 0.0,
    min_mirror_index: float = 0.0,
    min_opportunity_score: float = 65.0,
    lead_window: str = "24h",
    db_path: Path = DEFAULT_WATCHLIST_DB,
) -> dict[str, Any]:
    """Creates and persists a calibrated alert watchlist rule."""
    if not name.strip():
        raise ValueError("Watchlist `name` must be non-empty.")
    if spot_id is None and region is None and (lat is None or lon is None):
        raise ValueError("Must specify `spot_id`, `region`, or (`lat`, `lon`).")

    if spot_id is not None and (lat is None or lon is None):
        match = next((s for s in REGIONAL_SPOT_CATALOG if s["id"] == spot_id), None)
        if match is not None:
            lat = float(match["lat"])
            lon = float(match["lon"])
            elev_m = float(match["elev_m"]) if elev_m is None else float(elev_m)
            region = region or str(match["region"])

    rule_id = f"wl_{uuid.uuid4().hex[:10]}"
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    ph = phase.strip().lower()
    if ph not in ("sunrise", "sunset", "both"):
        ph = "sunset"
    lw = lead_window.strip().lower()
    if lw not in ("3h", "24h", "72h"):
        lw = "24h"
    phen_list = [str(p).strip() for p in (target_phenomena or []) if str(p).strip()]

    with _init_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO watchlists (
                id, name, spot_id, region, lat, lon, elev_m, phase,
                target_phenomena_json, min_calibrated_p50_pct, min_above_fog_prob_pct,
                min_mirror_index, min_opportunity_score, lead_window, enabled, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """,
            (
                rule_id,
                name.strip(),
                spot_id,
                region,
                lat,
                lon,
                elev_m,
                ph,
                json.dumps(phen_list),
                float(min_calibrated_p50_pct),
                float(min_above_fog_prob_pct),
                float(min_mirror_index),
                float(min_opportunity_score),
                lw,
                now_iso,
            ),
        )
        conn.commit()
    return get_watchlist_rule(rule_id, db_path=db_path)  # type: ignore[return-value]


def _row_to_rule(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "spot_id": row["spot_id"],
        "region": row["region"],
        "lat": row["lat"],
        "lon": row["lon"],
        "elev_m": row["elev_m"],
        "phase": row["phase"],
        "target_phenomena": json.loads(row["target_phenomena_json"] or "[]"),
        "min_calibrated_p50_pct": float(row["min_calibrated_p50_pct"]),
        "min_above_fog_prob_pct": float(row["min_above_fog_prob_pct"]),
        "min_mirror_index": float(row["min_mirror_index"]),
        "min_opportunity_score": float(row["min_opportunity_score"]),
        "lead_window": row["lead_window"],
        "enabled": bool(row["enabled"]),
        "created_at_utc": row["created_at_utc"],
    }


def get_watchlist_rule(rule_id: str, db_path: Path = DEFAULT_WATCHLIST_DB) -> dict[str, Any] | None:
    with _init_db(db_path) as conn:
        row = conn.execute("SELECT * FROM watchlists WHERE id = ?", (rule_id,)).fetchone()
        return _row_to_rule(row) if row is not None else None


def list_watchlist_rules(db_path: Path = DEFAULT_WATCHLIST_DB) -> list[dict[str, Any]]:
    with _init_db(db_path) as conn:
        rows = conn.execute("SELECT * FROM watchlists ORDER BY created_at_utc DESC").fetchall()
        return [_row_to_rule(r) for r in rows]


def delete_watchlist_rule(rule_id: str, db_path: Path = DEFAULT_WATCHLIST_DB) -> bool:
    with _init_db(db_path) as conn:
        cur = conn.execute("DELETE FROM watchlists WHERE id = ?", (rule_id,))
        conn.commit()
        return cur.rowcount > 0


def evaluate_alert_gate(rule: dict[str, Any], scored_spot: dict[str, Any]) -> dict[str, Any]:
    """
    Pure probabilistic decision gate deciding `TRIGGER`, `WATCH`, or `SUPPRESS`
    for a scored spot against a user's calibrated watchlist thresholds.
    """
    m = scored_spot.get("metrics") or {}
    cal_p10 = float(m.get("calibrated_p10_pct", 0.0))
    cal_p50 = float(m.get("calibrated_p50_pct", 0.0))
    cal_p90 = float(m.get("calibrated_p90_pct", 0.0))
    above_fog = float(m.get("summit_above_fog_prob_pct", 100.0))
    mirror_idx = m.get("mirror_index")
    opp_score = float(scored_spot.get("opportunity_score", 0.0))
    phen_tags = [str(p.get("tag", "")) for p in (scored_spot.get("phenomena") or [])]

    min_p50 = float(rule.get("min_calibrated_p50_pct", 50.0))
    min_fog = float(rule.get("min_above_fog_prob_pct", 0.0))
    min_mir = float(rule.get("min_mirror_index", 0.0))
    min_opp = float(rule.get("min_opportunity_score", 65.0))
    req_phen = [str(t).strip().lower() for t in (rule.get("target_phenomena") or []) if str(t).strip()]

    failed_checks: list[str] = []
    if cal_p50 < min_p50:
        failed_checks.append(f"calibrated P50 beam {cal_p50:.0f}% < {min_p50:.0f}% threshold")
    if min_fog > 0.0 and above_fog < min_fog:
        failed_checks.append(f"above-fog odds {above_fog:.0f}% < {min_fog:.0f}% threshold")
    if min_mir > 0.0 and (mirror_idx is None or float(mirror_idx) < min_mir):
        mv = 0.0 if mirror_idx is None else float(mirror_idx)
        failed_checks.append(f"Mirror Index {mv:.0f} < {min_mir:.0f} threshold")
    if opp_score < min_opp:
        failed_checks.append(f"opportunity score {opp_score:.0f} < {min_opp:.0f} threshold")
    if req_phen:
        lower_tags = {t.lower() for t in phen_tags}
        if not any(rp in lower_tags or any(rp in lt for lt in lower_tags) for rp in req_phen):
            failed_checks.append(f"requested phenomenon {rule.get('target_phenomena')} not active")

    if not failed_checks:
        decision = "TRIGGER"
        headline = (
            f"ALERT TRIGGERED — {scored_spot['name']}: {scored_spot['primary_phenomenon']} "
            f"(Score {opp_score:.0f}/100, Calibrated Beam {cal_p50:.0f}%, P10–P90: {cal_p10:.0f}–{cal_p90:.0f}%)"
        )
    elif cal_p90 >= min_p50 and opp_score >= (min_opp - 15.0) and (min_fog <= 0.0 or above_fog >= min_fog - 20.0):
        decision = "WATCH"
        headline = (
            f"WATCH — {scored_spot['name']}: Upside potential (P90 {cal_p90:.0f}%) but median still uncertain "
            f"({'; '.join(failed_checks)}). Re-evaluate at 3h GOES nowcast."
        )
    else:
        decision = "SUPPRESS"
        headline = (
            f"SUPPRESSED — {scored_spot['name']}: {'; '.join(failed_checks)}. "
            f"Cause: {scored_spot.get('causal_attribution', '')}"
        )

    return {
        "rule_id": rule.get("id"),
        "rule_name": rule.get("name"),
        "spot_id": scored_spot.get("spot_id"),
        "spot_name": scored_spot.get("name"),
        "decision": decision,
        "headline": headline,
        "failed_checks": failed_checks,
        "opportunity_score": opp_score,
        "primary_phenomenon": scored_spot.get("primary_phenomenon"),
        "calibrated_p10_p50_p90": [cal_p10, cal_p50, cal_p90],
        "causal_attribution": scored_spot.get("causal_attribution"),
    }


def evaluate_watchlist_rule_live(
    rule: dict[str, Any],
    date_str: str | None = None,
    replay_mode: str = "auto",
) -> list[dict[str, Any]]:
    """
    Runs live/historical 3D corridor verification for all spots covered by `rule`
    and returns calibrated `TRIGGER` / `WATCH` / `SUPPRESS` alert evaluations.
    """
    spots: list[dict[str, Any]] = []
    if rule.get("spot_id"):
        match = next((s for s in REGIONAL_SPOT_CATALOG if s["id"] == rule["spot_id"]), None)
        if match is not None:
            spots.append(match)
    if not spots and rule.get("lat") is not None and rule.get("lon") is not None:
        spots.append(
            {
                "id": rule.get("spot_id") or rule["id"],
                "name": rule["name"],
                "region": rule.get("region") or "custom",
                "state": "US",
                "lat": float(rule["lat"]),
                "lon": float(rule["lon"]),
                "elev_m": float(rule.get("elev_m") or 500.0),
                "archetype": "alpine_lake" if float(rule.get("min_mirror_index", 0.0)) > 0 else "coastal_ridge",
                "is_lake": float(rule.get("min_mirror_index", 0.0)) > 0,
                "upwind_shelter_deg": 4.0,
            }
        )
    elif not spots and rule.get("region"):
        reg_l = str(rule["region"]).strip().lower()
        spots = [s for s in REGIONAL_SPOT_CATALOG if s["region"].lower() == reg_l][:6]

    phases = ("sunrise", "sunset") if rule.get("phase") == "both" else (str(rule.get("phase", "sunset")),)
    results: list[dict[str, Any]] = []

    for sp in spots:
        tz_obj, _ = local_tz_for_lon(float(sp["lon"]))
        d_use = date_str or dt.datetime.now(tz_obj).date().isoformat()
        req_d = dt.date.fromisoformat(d_use)
        noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz_obj)
        tz_off = (noon_loc.utcoffset() or dt.timedelta(0)).total_seconds() / 3600.0
        sr, ss = _find_sunrise_sunset_golden_hours(float(sp["lat"]), float(sp["lon"]), req_d, tz_off)

        for ph in phases:
            _, az, el = sr if ph == "sunrise" else ss
            raw = solve_weather_and_corridor(
                float(sp["lat"]),
                float(sp["lon"]),
                d_use,
                phase=ph,
                sun_az_deg=az,
                sun_el_deg=el,
                target_elev_m=float(sp["elev_m"]),
                model_engine="ensemble",
                replay_mode=replay_mode,
                include_lut=bool(sp.get("is_lake", False)),
            )
            doc = json.loads(raw)
            scored = classify_spot_phenomena_and_attribution(sp, doc)
            ev = evaluate_alert_gate(rule, scored)
            ev["date"] = d_use
            ev["phase"] = ph
            results.append(ev)

    return results
