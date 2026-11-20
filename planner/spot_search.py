"""
Nationwide Photography Spot & Coordinate Search (`planner/spot_search.py`).

Unlocks "Any Peak / Lake / Trailhead in the US" from the main PhotoPlanner search bar:
  1. Direct GPS coordinate parsing (`"37.9235, -122.5965"`, `"46.7865 -121.7353"`, `"37.74N 119.53W"`).
  2. Zero-latency curated local search across all 40 CONUS destinations in `REGIONAL_SPOT_CATALOG`
     plus Yosemite/SF built-in landmarks (matching name, state, region, or archetype).
  3. Commercial-safe Open-Meteo Geocoding API fallback (routed via `customer-geocoding-api.open-meteo.com`
     when `OPEN_METEO_API_KEY` is set) with bounded in-memory TTL caching.
"""

from __future__ import annotations

from collections import OrderedDict
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from planner.seo_guides import REGION_LABELS, slugify_spot
from planner.weather.regional_hunter import REGIONAL_SPOT_CATALOG

USER_AGENT = "PhotoPlanner/2.0 (spot-search; github.com/lucasbrambrink/photoplanner)"

# Additional classic Yosemite/SF preset landmarks whose exact names match `/api/shoot-plan`
PRESET_LANDMARK_EXTRAS: list[dict[str, Any]] = [
    {"id": "upper_yosemite_fall", "name": "Upper Yosemite Fall", "region": "sierra", "state": "CA", "lat": 37.7568, "lon": -119.5969, "elev_m": 2005.0, "archetype": "alpine_granite", "preset_landmark": "Upper Yosemite Fall"},
    {"id": "sentinel_dome", "name": "Sentinel Dome", "region": "sierra", "state": "CA", "lat": 37.7229, "lon": -119.5844, "elev_m": 2476.0, "archetype": "alpine_granite", "preset_landmark": "Sentinel Dome"},
    {"id": "bridalveil_fall", "name": "Bridalveil Fall", "region": "sierra", "state": "CA", "lat": 37.7167, "lon": -119.6468, "elev_m": 1400.0, "archetype": "alpine_granite", "preset_landmark": "Bridalveil Fall"},
    {"id": "north_dome", "name": "North Dome", "region": "sierra", "state": "CA", "lat": 37.7681, "lon": -119.5553, "elev_m": 2299.0, "archetype": "alpine_granite", "preset_landmark": "North Dome"},
]

# Maps curated catalog IDs that also have dedicated 1m/trail preset landmark names in `server.py`
CATALOG_PRESET_LANDMARK_MAP: dict[str, str] = {
    "pedro_point": "Pedro Point Headlands",
    "montara_mtn": "Montara Mountain",
    "half_dome": "Half Dome",
    "clouds_rest": "Clouds Rest",
    "tenaya_lake": "Tenaya Lake",
    "cathedral_peak": "Cathedral Peak",
}

_COORD_RE = re.compile(
    r"^\s*([+-]?\d{1,2}(?:\.\d+)?)\s*°?\s*([NSns])?\s*[,\s]+\s*([+-]?\d{1,3}(?:\.\d+)?)\s*°?\s*([EWew])?\s*$"
)

_GEO_CACHE_LOCK = threading.Lock()
_GEO_CACHE: OrderedDict[str, tuple[float, list[dict[str, Any]]]] = OrderedDict()
_GEO_CACHE_MAX = 256
_GEO_CACHE_TTL_S = 3600.0


def parse_gps_query(query: str) -> dict[str, Any] | None:
    """Parses pasted latitude/longitude strings like `37.9235, -122.5965` or `46.78N 121.73W`."""
    m = _COORD_RE.match(query or "")
    if not m:
        return None
    lat = float(m.group(1))
    ns = (m.group(2) or "").upper()
    lon = float(m.group(3))
    ew = (m.group(4) or "").upper()
    if ns == "S" and lat > 0:
        lat = -lat
    if ew == "W" and lon > 0:
        lon = -lon
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return {
        "id": f"gps_{lat:.4f}_{lon:.4f}",
        "name": f"GPS ({lat:.4f}, {lon:.4f})",
        "subtitle": "Custom GPS Coordinate",
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "elev_m": None,
        "elev_ft": None,
        "state": "",
        "region": "custom",
        "region_label": "Custom GPS",
        "archetype": "custom",
        "source": "gps",
        "preset_landmark": None,
        "guide_slug": None,
    }


def _build_local_index() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for spot in REGIONAL_SPOT_CATALOG:
        sid = spot["id"]
        seen_ids.add(sid)
        elev_ft = int(round(float(spot["elev_m"]) * 3.28084))
        reg_label = REGION_LABELS.get(spot["region"], spot["region"].replace("_", " ").title())
        items.append(
            {
                "id": sid,
                "name": spot["name"],
                "subtitle": f"{spot['state']} · {elev_ft:,} ft · {spot['archetype'].replace('_', ' ').title()}",
                "lat": float(spot["lat"]),
                "lon": float(spot["lon"]),
                "elev_m": float(spot["elev_m"]),
                "elev_ft": elev_ft,
                "state": spot["state"],
                "region": spot["region"],
                "region_label": reg_label,
                "archetype": spot["archetype"],
                "source": "curated",
                "preset_landmark": CATALOG_PRESET_LANDMARK_MAP.get(sid),
                "guide_slug": slugify_spot(spot),
            }
        )
    for extra in PRESET_LANDMARK_EXTRAS:
        if extra["id"] in seen_ids:
            continue
        elev_ft = int(round(float(extra["elev_m"]) * 3.28084))
        items.append(
            {
                "id": extra["id"],
                "name": extra["name"],
                "subtitle": f"{extra['state']} · {elev_ft:,} ft · Yosemite Preset",
                "lat": float(extra["lat"]),
                "lon": float(extra["lon"]),
                "elev_m": float(extra["elev_m"]),
                "elev_ft": elev_ft,
                "state": extra["state"],
                "region": extra["region"],
                "region_label": REGION_LABELS.get(extra["region"], "Sierra Nevada"),
                "archetype": extra["archetype"],
                "source": "curated",
                "preset_landmark": extra["preset_landmark"],
                "guide_slug": None,
            }
        )
    return items


def _score_local_match(item: dict[str, Any], q_lower: str, tokens: list[str]) -> float:
    name_l = item["name"].lower()
    region_l = f"{item['region']} {item['region_label']} {item['state']} {item['archetype']}".lower()
    haystack = f"{name_l} {region_l}"
    if not all(tok in haystack for tok in tokens):
        return 0.0
    score = 10.0
    if name_l == q_lower:
        score += 100.0
    elif name_l.startswith(q_lower):
        score += 60.0
    elif q_lower in name_l:
        score += 35.0
    if item.get("preset_landmark"):
        score += 4.0
    return score


def _geocode_open_meteo(query: str, count: int = 6, country_code: str = "US") -> list[dict[str, Any]]:
    cache_key = f"{query.strip().lower()}|{count}|{country_code.upper()}"
    now = time.monotonic()
    with _GEO_CACHE_LOCK:
        hit = _GEO_CACHE.get(cache_key)
        if hit is not None:
            exp, vals = hit
            if now < exp:
                _GEO_CACHE.move_to_end(cache_key)
                return list(vals)
            _GEO_CACHE.pop(cache_key, None)

    api_key = os.environ.get("OPEN_METEO_API_KEY", "").strip()
    host = "customer-geocoding-api.open-meteo.com" if api_key else "geocoding-api.open-meteo.com"
    params: dict[str, str] = {
        "name": query.strip(),
        "count": str(max(1, min(15, count))),
        "language": "en",
        "format": "json",
    }
    if country_code:
        params["countryCode"] = country_code.upper()
    if api_key:
        params["apikey"] = api_key
    url = f"https://{host}/v1/search?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=4.5) as resp:
            doc = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return []

    results: list[dict[str, Any]] = []
    for row in doc.get("results") or []:
        lat = row.get("latitude")
        lon = row.get("longitude")
        if lat is None or lon is None:
            continue
        elev_m = row.get("elevation")
        elev_ft = int(round(float(elev_m) * 3.28084)) if elev_m is not None else None
        admin1 = str(row.get("admin1") or "").strip()
        country = str(row.get("country_code") or "").strip()
        feat_code = str(row.get("feature_code") or "").strip()
        parts = [p for p in (admin1, country) if p]
        if elev_ft is not None:
            parts.append(f"{elev_ft:,} ft")
        display_name = str(row.get("name") or "").strip()
        full_name = f"{display_name}, {admin1}" if admin1 and admin1.lower() not in display_name.lower() else display_name
        results.append(
            {
                "id": f"geo_{row.get('id', f'{lat:.3f}_{lon:.3f}')}",
                "name": full_name,
                "subtitle": " · ".join(parts) if parts else "US Location",
                "lat": round(float(lat), 5),
                "lon": round(float(lon), 5),
                "elev_m": float(elev_m) if elev_m is not None else None,
                "elev_ft": elev_ft,
                "state": admin1,
                "region": "geocoded",
                "region_label": admin1 or "United States",
                "archetype": feat_code.lower() or "place",
                "source": "geocoder",
                "preset_landmark": None,
                "guide_slug": None,
            }
        )

    with _GEO_CACHE_LOCK:
        _GEO_CACHE[cache_key] = (now + _GEO_CACHE_TTL_S, list(results))
        while len(_GEO_CACHE) > _GEO_CACHE_MAX:
            _GEO_CACHE.popitem(last=False)
    return results


def search_photography_spots(
    query: str = "",
    *,
    limit: int = 12,
    allow_remote: bool = True,
    country_code: str = "US",
) -> dict[str, Any]:
    """
    Unified search across GPS coordinates, all 44 curated US destinations, and (optionally)
    Open-Meteo Geocoding for any arbitrary peak, lake, park, or town.
    """
    q = (query or "").strip()
    lim = max(1, min(50, int(limit)))
    local_index = _build_local_index()

    # 1. Empty query -> return curated featured spots spanning all US regions
    if not q:
        return {
            "query": "",
            "count": min(lim, len(local_index)),
            "results": local_index[:lim],
            "sources": ["curated"],
        }

    # 2. Direct GPS coordinate paste
    gps_hit = parse_gps_query(q)
    if gps_hit is not None:
        return {
            "query": q,
            "count": 1,
            "results": [gps_hit],
            "sources": ["gps"],
        }

    # 3. Curated local fuzzy/token search
    q_lower = q.lower()
    tokens = [t for t in re.split(r"[\s,/]+", q_lower) if t]
    scored = []
    for item in local_index:
        sc = _score_local_match(item, q_lower, tokens)
        if sc > 0:
            scored.append((sc, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["name"]))
    out = [item for _, item in scored[:lim]]
    sources = ["curated"] if out else []

    # 4. Commercial-safe Open-Meteo Geocoding fallback when query >= 3 chars and room remains
    if allow_remote and len(q) >= 3 and len(out) < lim:
        geo_hits = _geocode_open_meteo(q, count=lim - len(out) + 2, country_code=country_code)
        if geo_hits:
            sources.append("geocoder")
            for gh in geo_hits:
                # Deduplicate within ~0.04 deg (~4 km) of an already-returned curated spot
                dup = any(
                    abs(existing["lat"] - gh["lat"]) < 0.04 and abs(existing["lon"] - gh["lon"]) < 0.04
                    for existing in out
                )
                if not dup:
                    out.append(gh)
                if len(out) >= lim:
                    break

    return {
        "query": q,
        "count": len(out),
        "results": out[:lim],
        "sources": sources,
    }
