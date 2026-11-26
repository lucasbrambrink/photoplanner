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
from .light_archetypes import ARCHETYPE_CATALOG, ARCHETYPE_ORDER, VIABLE_THRESHOLD_PCT
from .nwp_corridor import local_tz_for_lon
from .regional_hunter import REGIONAL_SPOT_CATALOG, classify_spot_phenomena_and_attribution

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WATCHLIST_DB = ROOT_DIR / "data" / "watchlists.sqlite3"
VALID_ARCHETYPE_IDS = frozenset(ARCHETYPE_ORDER)


def _normalize_target_archetypes(target_archetypes: list[str] | None) -> list[str]:
    if not target_archetypes:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in target_archetypes:
        aid = str(raw).strip().lower()
        if not aid:
            continue
        if aid not in VALID_ARCHETYPE_IDS:
            valid = ", ".join(ARCHETYPE_ORDER)
            raise ValueError(f"Unknown light archetype '{aid}'. Valid archetypes: {valid}.")
        if aid not in seen:
            seen.add(aid)
            out.append(aid)
    return out


def _init_db(db_path: Path | None = None) -> sqlite3.Connection:
    resolved = db_path or DEFAULT_WATCHLIST_DB
    resolved.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(resolved))
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
            target_archetypes_json TEXT NOT NULL DEFAULT '[]',
            min_archetype_prob_pct REAL NOT NULL DEFAULT 0.0,
            min_calibrated_p50_pct REAL NOT NULL DEFAULT 50.0,
            min_above_fog_prob_pct REAL NOT NULL DEFAULT 0.0,
            min_mirror_index REAL NOT NULL DEFAULT 0.0,
            min_opportunity_score REAL NOT NULL DEFAULT 65.0,
            lead_window TEXT NOT NULL DEFAULT '24h',
            enabled INTEGER NOT NULL DEFAULT 1,
            notify_email TEXT,
            notify_webhook_url TEXT,
            last_notified_date TEXT,
            last_notified_phase TEXT,
            created_at_utc TEXT NOT NULL
        )
        """
    )
    # Safe additive migration for existing SQLite watchlists created before delivery / archetype fields existed
    existing_cols = {r["name"] for r in conn.execute("PRAGMA table_info(watchlists)").fetchall()}
    for col_name, col_type in (
        ("notify_email", "TEXT"),
        ("notify_webhook_url", "TEXT"),
        ("last_notified_date", "TEXT"),
        ("last_notified_phase", "TEXT"),
        ("target_archetypes_json", "TEXT NOT NULL DEFAULT '[]'"),
        ("min_archetype_prob_pct", "REAL NOT NULL DEFAULT 0.0"),
    ):
        if col_name not in existing_cols:
            conn.execute(f"ALTER TABLE watchlists ADD COLUMN {col_name} {col_type}")
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
    target_archetypes: list[str] | None = None,
    min_archetype_prob_pct: float | None = None,
    min_calibrated_p50_pct: float = 50.0,
    min_above_fog_prob_pct: float = 0.0,
    min_mirror_index: float = 0.0,
    min_opportunity_score: float = 65.0,
    lead_window: str = "24h",
    notify_email: str | None = None,
    notify_webhook_url: str | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Creates and persists a calibrated LightFinder alert watchlist rule."""
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
    if ph not in ("sunrise", "sunset", "both", "any"):
        ph = "sunset"
    lw = lead_window.strip().lower()
    if lw not in ("3h", "24h", "72h"):
        lw = "24h"
    phen_list = [str(p).strip() for p in (target_phenomena or []) if str(p).strip()]
    arch_list = _normalize_target_archetypes(target_archetypes)
    if min_archetype_prob_pct is None:
        resolved_min_arch = 50.0 if arch_list else 0.0
    else:
        resolved_min_arch = max(0.0, min(100.0, float(min_archetype_prob_pct)))
    clean_email = notify_email.strip() if notify_email and notify_email.strip() else None
    clean_webhook = notify_webhook_url.strip() if notify_webhook_url and notify_webhook_url.strip() else None
    if clean_email and "@" not in clean_email:
        raise ValueError("`notify_email` must be a valid email address.")
    if clean_webhook and not clean_webhook.startswith(("http://", "https://")):
        raise ValueError("`notify_webhook_url` must start with http:// or https://.")

    with _init_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO watchlists (
                id, name, spot_id, region, lat, lon, elev_m, phase,
                target_phenomena_json, target_archetypes_json, min_archetype_prob_pct,
                min_calibrated_p50_pct, min_above_fog_prob_pct,
                min_mirror_index, min_opportunity_score, lead_window, enabled,
                notify_email, notify_webhook_url, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
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
                json.dumps(arch_list),
                resolved_min_arch,
                float(min_calibrated_p50_pct),
                float(min_above_fog_prob_pct),
                float(min_mirror_index),
                float(min_opportunity_score),
                lw,
                clean_email,
                clean_webhook,
                now_iso,
            ),
        )
        conn.commit()
    return get_watchlist_rule(rule_id, db_path=db_path)  # type: ignore[return-value]


def _row_to_rule(row: sqlite3.Row) -> dict[str, Any]:
    cols = set(row.keys())
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
        "target_archetypes": json.loads(row["target_archetypes_json"] or "[]") if "target_archetypes_json" in cols else [],
        "min_archetype_prob_pct": float(row["min_archetype_prob_pct"]) if "min_archetype_prob_pct" in cols else 0.0,
        "min_calibrated_p50_pct": float(row["min_calibrated_p50_pct"]),
        "min_above_fog_prob_pct": float(row["min_above_fog_prob_pct"]),
        "min_mirror_index": float(row["min_mirror_index"]),
        "min_opportunity_score": float(row["min_opportunity_score"]),
        "lead_window": row["lead_window"],
        "enabled": bool(row["enabled"]),
        "notify_email": row["notify_email"] if "notify_email" in cols else None,
        "notify_webhook_url": row["notify_webhook_url"] if "notify_webhook_url" in cols else None,
        "last_notified_date": row["last_notified_date"] if "last_notified_date" in cols else None,
        "last_notified_phase": row["last_notified_phase"] if "last_notified_phase" in cols else None,
        "created_at_utc": row["created_at_utc"],
    }


def get_watchlist_rule(rule_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    with _init_db(db_path) as conn:
        row = conn.execute("SELECT * FROM watchlists WHERE id = ?", (rule_id,)).fetchone()
        return _row_to_rule(row) if row is not None else None


def list_watchlist_rules(db_path: Path | None = None) -> list[dict[str, Any]]:
    with _init_db(db_path) as conn:
        rows = conn.execute("SELECT * FROM watchlists ORDER BY created_at_utc DESC").fetchall()
        return [_row_to_rule(r) for r in rows]


def delete_watchlist_rule(rule_id: str, db_path: Path | None = None) -> bool:
    with _init_db(db_path) as conn:
        cur = conn.execute("DELETE FROM watchlists WHERE id = ?", (rule_id,))
        conn.commit()
        return cur.rowcount > 0


def mark_watchlist_notified(
    rule_id: str,
    date_str: str,
    phase: str,
    db_path: Path | None = None,
) -> None:
    """Marks a watchlist rule as delivered for `(date_str, phase)` to prevent duplicate alerts."""
    with _init_db(db_path) as conn:
        conn.execute(
            "UPDATE watchlists SET last_notified_date = ?, last_notified_phase = ? WHERE id = ?",
            (date_str, phase.strip().lower(), rule_id),
        )
        conn.commit()


def _format_minutes_hhmm(minutes_local: int | float | None) -> str | None:
    if minutes_local is None:
        return None
    m = int(round(float(minutes_local))) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def _select_best_archetype_window(
    rule: dict[str, Any],
    scored_spot: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Selects the best matching 7-archetype probability distribution across the primary phase
    snapshot and eligible `hourly_timeline` entries for `rule`.

    Because the 7 LightFinder archetypes are mutually exclusive and sum strictly to 100%,
    the probability that ANY of a user's selected `target_archetypes` occurs in a given hour
    is their exact probability union: `sum(P(a) for a in target_archetypes)`.
    """
    target_archetypes = [
        str(a).strip().lower()
        for a in (rule.get("target_archetypes") or [])
        if str(a).strip().lower() in VALID_ARCHETYPE_IDS
    ]
    eval_phase = str(scored_spot.get("eval_phase") or rule.get("phase") or "sunset").strip().lower()
    phase_mins = scored_spot.get("phase_minutes_local")

    candidates: list[tuple[bool, str | None, int | None, dict[str, Any]]] = []
    primary_dist = scored_spot.get("light_archetypes")
    if isinstance(primary_dist, dict) and isinstance(primary_dist.get("probabilities"), dict):
        primary_time = _format_minutes_hhmm(phase_mins)
        primary_hr = int(phase_mins) // 60 if phase_mins is not None else None
        candidates.append((True, primary_time, primary_hr, primary_dist))

    for ch in scored_spot.get("light_chapters") or []:
        if not isinstance(ch, dict):
            continue
        c_dist = ch.get("light_archetypes")
        if not isinstance(c_dist, dict) or not isinstance(c_dist.get("probabilities"), dict):
            continue
        c_id = str(ch.get("id") or "").lower()
        c_mins = ch.get("peak_minutes")
        if eval_phase == "sunrise" and c_id not in ("dawn_blue", "morning_gold"):
            continue
        if eval_phase == "sunset" and c_id not in ("evening_gold", "dusk_blue"):
            continue
        c_hr = int(c_mins) // 60 if c_mins is not None else None
        c_lbl = str(ch.get("peak_local") or _format_minutes_hhmm(c_mins) or "") or None
        candidates.append((False, c_lbl, c_hr, c_dist))

    timeline = scored_spot.get("hourly_timeline") or []
    for entry in timeline:
        if not isinstance(entry, dict):
            continue
        h_dist = entry.get("light_archetypes")
        if not isinstance(h_dist, dict) or not isinstance(h_dist.get("probabilities"), dict):
            continue
        hr = entry.get("hour")
        h_mins = entry.get("minutes_local")
        if h_mins is None and hr is not None:
            h_mins = int(hr) * 60 + 30
        if eval_phase != "any":
            if phase_mins is not None and h_mins is not None:
                if abs(int(h_mins) - int(phase_mins)) > 135:
                    continue
            elif hr is not None:
                if eval_phase == "sunrise" and int(hr) > 11:
                    continue
                if eval_phase == "sunset" and int(hr) < 12:
                    continue
        t_lbl = str(entry.get("time_local") or (f"{int(hr):02d}:00" if hr is not None else "")) or None
        candidates.append((False, t_lbl, int(hr) if hr is not None else None, h_dist))

    if not candidates:
        return None

    def _score_candidate(item: tuple[bool, str | None, int | None, dict[str, Any]]) -> tuple[float, float, int]:
        is_primary, _, _, dist = item
        probs = dist.get("probabilities") or {}
        if target_archetypes:
            combined = float(sum(float(probs.get(aid, 0.0)) for aid in target_archetypes))
            best_single = max((float(probs.get(aid, 0.0)) for aid in target_archetypes), default=0.0)
            return (combined, best_single, 1 if is_primary else 0)
        dom = dist.get("dominant") or {}
        dom_pct = float(dom.get("probability_pct", max((float(v) for v in probs.values()), default=0.0)))
        return (dom_pct, dom_pct, 1 if is_primary else 0)

    _, best_time_local, best_hour, best_dist = max(candidates, key=_score_candidate)
    probs = {aid: int(round(float((best_dist.get("probabilities") or {}).get(aid, 0)))) for aid in ARCHETYPE_ORDER}

    dom_id = str((best_dist.get("dominant") or {}).get("id") or max(ARCHETYPE_ORDER, key=lambda a: probs[a]))
    dom_prob = probs.get(dom_id, 0)
    dom_meta = ARCHETYPE_CATALOG.get(dom_id, {})

    if target_archetypes:
        matched_id = max(target_archetypes, key=lambda a: probs.get(a, 0))
        matched_prob = probs.get(matched_id, 0)
        combined_prob = min(100, sum(probs.get(a, 0) for a in target_archetypes))
        target_breakdown = {a: probs.get(a, 0) for a in target_archetypes}
    else:
        matched_id = dom_id
        matched_prob = dom_prob
        combined_prob = dom_prob
        target_breakdown = {}

    matched_meta = ARCHETYPE_CATALOG.get(matched_id, {})
    field_ref = dict(matched_meta.get("field_reference") or {})
    return {
        "matched_archetype_id": matched_id,
        "matched_archetype_name": str(matched_meta.get("name") or matched_id),
        "matched_archetype_short": str(matched_meta.get("short_name") or matched_meta.get("name") or matched_id),
        "matched_archetype_prob_pct": float(matched_prob),
        "combined_target_prob_pct": float(combined_prob),
        "target_archetypes": target_archetypes,
        "target_breakdown": target_breakdown,
        "best_time_local": best_time_local,
        "best_hour": best_hour,
        "dominant_archetype_id": dom_id,
        "dominant_archetype_name": str(dom_meta.get("name") or dom_id),
        "dominant_archetype_prob_pct": float(dom_prob),
        "tactical_lever": str(field_ref.get("tactical_lever") or ""),
        "field_reference": field_ref,
        "light_archetypes": {
            "probabilities": probs,
            "dominant": {
                "id": dom_id,
                "name": str(dom_meta.get("name") or dom_id),
                "probability_pct": dom_prob,
            },
            "viable": [
                {
                    "id": aid,
                    "name": str(ARCHETYPE_CATALOG[aid]["name"]),
                    "probability_pct": probs[aid],
                }
                for aid in sorted(ARCHETYPE_ORDER, key=lambda a: -probs[a])
                if probs[aid] >= VIABLE_THRESHOLD_PCT
            ]
            or [
                {
                    "id": dom_id,
                    "name": str(dom_meta.get("name") or dom_id),
                    "probability_pct": dom_prob,
                }
            ],
        },
    }


def evaluate_alert_gate(rule: dict[str, Any], scored_spot: dict[str, Any]) -> dict[str, Any]:
    """
    Pure probabilistic decision gate deciding `TRIGGER`, `WATCH`, or `SUPPRESS`
    for a scored spot against a user's calibrated watchlist thresholds.

    Supports both:
      - **7-Archetype LightFinder alerts** (`target_archetypes` set and/or `min_archetype_prob_pct > 0`):
        gates on exact mutually-exclusive union probability `sum(P(a) for a in target_archetypes)`
        across the spot's phase window (or full 24h scan when `phase == 'any'`).
      - **Legacy beam/opportunity alerts** (`target_archetypes` empty): preserves calibrated
        `P50` transmittance, above-fog odds, Mirror Index, opportunity score, and phenomenon checks.
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

    target_archetypes = [
        str(a).strip().lower()
        for a in (rule.get("target_archetypes") or [])
        if str(a).strip().lower() in VALID_ARCHETYPE_IDS
    ]
    min_arch_raw = float(rule.get("min_archetype_prob_pct") or 0.0)
    is_archetype_rule = bool(target_archetypes) or min_arch_raw > 0.0
    arch_match = _select_best_archetype_window(rule, scored_spot)

    failed_checks: list[str] = []

    if is_archetype_rule:
        min_arch = max(float(VIABLE_THRESHOLD_PCT), min_arch_raw if min_arch_raw > 0.0 else 50.0)
        target_names = [ARCHETYPE_CATALOG[a]["name"] for a in target_archetypes] or [
            (arch_match or {}).get("dominant_archetype_name", "Target Light")
        ]
        target_label = " / ".join(target_names)

        if arch_match is None:
            combined_prob = 0.0
            failed_checks.append(f"{target_label} probability unavailable (< {min_arch:.0f}% threshold)")
        else:
            combined_prob = float(arch_match["combined_target_prob_pct"])
            if combined_prob < min_arch:
                dom_name = arch_match["dominant_archetype_name"]
                dom_pct = arch_match["dominant_archetype_prob_pct"]
                dom_clause = (
                    f" — {dom_name} ({dom_pct:.0f}%) is dominant instead"
                    if arch_match["dominant_archetype_id"] not in target_archetypes
                    else ""
                )
                failed_checks.append(
                    f"{target_label} odds {combined_prob:.0f}% < {min_arch:.0f}% threshold{dom_clause}"
                )

        if min_fog > 0.0 and above_fog < min_fog:
            failed_checks.append(f"above-fog odds {above_fog:.0f}% < {min_fog:.0f}% threshold")
        if min_mir > 0.0 and (mirror_idx is None or float(mirror_idx) < min_mir):
            mv = 0.0 if mirror_idx is None else float(mirror_idx)
            failed_checks.append(f"Mirror Index {mv:.0f} < {min_mir:.0f} threshold")
        if req_phen:
            lower_tags = {t.lower() for t in phen_tags}
            if not any(rp in lower_tags or any(rp in lt for lt in lower_tags) for rp in req_phen):
                failed_checks.append(f"requested phenomenon {rule.get('target_phenomena')} not active")

        best_time = (arch_match or {}).get("best_time_local")
        time_clause = f" around {best_time}" if best_time else ""
        matched_name = (arch_match or {}).get("matched_archetype_name") or target_label
        lever = (arch_match or {}).get("tactical_lever") or ""
        watch_floor = max(float(VIABLE_THRESHOLD_PCT), min_arch - 15.0)

        if not failed_checks:
            decision = "TRIGGER"
            headline = (
                f"ALERT TRIGGERED — {scored_spot['name']}: {combined_prob:.0f}% {matched_name}{time_clause}"
                + (f" · Lever: {lever}" if lever else "")
            )
            reason = (
                f"{target_label} hits {combined_prob:.0f}% probability{time_clause} "
                f"(clears ≥{min_arch:.0f}% alert threshold). "
                + (f"Tactical lever: {lever}." if lever else str(scored_spot.get("causal_attribution") or ""))
            )
        elif combined_prob >= watch_floor and (min_fog <= 0.0 or above_fog >= min_fog - 20.0):
            decision = "WATCH"
            headline = (
                f"WATCH — {scored_spot['name']}: {target_label} viable at {combined_prob:.0f}%{time_clause} "
                f"({'; '.join(failed_checks)}). Re-evaluate at 3h GOES nowcast."
            )
            reason = (
                f"{target_label} is viable at {combined_prob:.0f}%{time_clause}, just under your "
                f"≥{min_arch:.0f}% trigger ({'; '.join(failed_checks)})."
            )
        else:
            decision = "SUPPRESS"
            headline = (
                f"SUPPRESSED — {scored_spot['name']}: {'; '.join(failed_checks)}."
            )
            reason = (
                f"{'; '.join(failed_checks)}."
                + (f" {scored_spot.get('causal_attribution', '')}" if scored_spot.get("causal_attribution") else "")
            )
    else:
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
            reason = str(scored_spot.get("causal_attribution") or headline)
        elif cal_p90 >= min_p50 and opp_score >= (min_opp - 15.0) and (min_fog <= 0.0 or above_fog >= min_fog - 20.0):
            decision = "WATCH"
            headline = (
                f"WATCH — {scored_spot['name']}: Upside potential (P90 {cal_p90:.0f}%) but median still uncertain "
                f"({'; '.join(failed_checks)}). Re-evaluate at 3h GOES nowcast."
            )
            reason = headline
        else:
            decision = "SUPPRESS"
            headline = (
                f"SUPPRESSED — {scored_spot['name']}: {'; '.join(failed_checks)}. "
                f"Cause: {scored_spot.get('causal_attribution', '')}"
            )
            reason = headline

    result: dict[str, Any] = {
        "rule_id": rule.get("id"),
        "rule_name": rule.get("name"),
        "spot_id": scored_spot.get("spot_id"),
        "spot_name": scored_spot.get("name"),
        "decision": decision,
        "headline": headline,
        "reason": reason,
        "failed_checks": failed_checks,
        "opportunity_score": opp_score,
        "primary_phenomenon": scored_spot.get("primary_phenomenon"),
        "calibrated_p10_p50_p90": [cal_p10, cal_p50, cal_p90],
        "causal_attribution": scored_spot.get("causal_attribution"),
        "target_archetypes": target_archetypes,
    }
    if arch_match is not None:
        result.update(
            {
                "matched_archetype_id": arch_match["matched_archetype_id"],
                "matched_archetype_name": arch_match["matched_archetype_name"],
                "matched_archetype_short": arch_match["matched_archetype_short"],
                "matched_archetype_prob_pct": arch_match["matched_archetype_prob_pct"],
                "combined_target_prob_pct": arch_match["combined_target_prob_pct"],
                "target_breakdown": arch_match["target_breakdown"],
                "best_time_local": arch_match["best_time_local"],
                "best_hour": arch_match["best_hour"],
                "dominant_archetype_id": arch_match["dominant_archetype_id"],
                "dominant_archetype_name": arch_match["dominant_archetype_name"],
                "dominant_archetype_prob_pct": arch_match["dominant_archetype_prob_pct"],
                "tactical_lever": arch_match["tactical_lever"],
                "field_reference": arch_match["field_reference"],
                "light_archetypes": arch_match["light_archetypes"],
            }
        )
    return result


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

    rule_phase = str(rule.get("phase", "sunset")).strip().lower()
    if rule_phase == "both":
        phases = ("sunrise", "sunset")
    elif rule_phase == "any":
        phases = ("any",)
    else:
        phases = (rule_phase if rule_phase in ("sunrise", "sunset") else "sunset",)
    results: list[dict[str, Any]] = []

    for sp in spots:
        tz_obj, _ = local_tz_for_lon(float(sp["lon"]))
        d_use = date_str or dt.datetime.now(tz_obj).date().isoformat()
        req_d = dt.date.fromisoformat(d_use)
        noon_loc = dt.datetime(req_d.year, req_d.month, req_d.day, 12, 0, tzinfo=tz_obj)
        tz_off = (noon_loc.utcoffset() or dt.timedelta(0)).total_seconds() / 3600.0
        sr, ss = _find_sunrise_sunset_golden_hours(float(sp["lat"]), float(sp["lon"]), req_d, tz_off)

        for ph in phases:
            solver_phase = "sunrise" if ph == "sunrise" else "sunset"
            phase_mins, az, el = sr if solver_phase == "sunrise" else ss
            raw = solve_weather_and_corridor(
                float(sp["lat"]),
                float(sp["lon"]),
                d_use,
                phase=solver_phase,
                sun_az_deg=az,
                sun_el_deg=el,
                target_elev_m=float(sp["elev_m"]),
                model_engine="ensemble",
                replay_mode=replay_mode,
                include_lut=bool(sp.get("is_lake", False)),
            )
            doc = json.loads(raw)
            scored = classify_spot_phenomena_and_attribution(sp, doc)
            scored["light_archetypes"] = doc.get("light_archetypes")
            scored["light_chapters"] = doc.get("light_chapters") or []
            scored["hourly_timeline"] = doc.get("hourly_timeline") or []
            scored["phase_minutes_local"] = phase_mins
            scored["eval_phase"] = ph
            ev = evaluate_alert_gate(rule, scored)
            ev["date"] = d_use
            ev["phase"] = ph
            results.append(ev)

    return results
