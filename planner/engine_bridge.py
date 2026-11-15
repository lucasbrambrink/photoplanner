"""
Python bridge connecting the PhotoPlanner Rust Light-Notch & Ray-Casting Engine
(`raycaster/target/release/photoplanner`) with the TrailBinder (`recreation`) stack.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RUST_BIN = ROOT_DIR / "raycaster" / "target" / "release" / "photoplanner"
DEFAULT_RECREATION_DB = Path("/Users/lucasbrambrink/Coding/recreation/recreation/db.sqlite3")
DEFAULT_ELEVATION_DIR = Path("/Users/lucasbrambrink/Coding/recreation/recreation/data/elevation")


class PhotoPlannerEngine:
    """High-level Python wrapper around the Rust `photoplanner` ray-caster binary."""

    def __init__(
        self,
        binary_path: Path = DEFAULT_RUST_BIN,
        recreation_db: Path = DEFAULT_RECREATION_DB,
        elevation_dir: Path = DEFAULT_ELEVATION_DIR,
    ) -> None:
        self.binary_path = Path(binary_path)
        self.recreation_db = Path(recreation_db)
        self.elevation_dir = Path(elevation_dir)

    def scan_light_notches(
        self,
        lat: float,
        lon: float,
        near_grid_m: int = 3000,
        az_step_deg: float = 0.1,
        year: int = 2026,
    ) -> Dict[str, Any]:
        """
        Runs the 1m Hierarchical Max-Mipmap + SRTM horizon sweep around (lat, lon)
        and returns detected Light Notches along with exact solar alignment dates/times.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cmd = [
                str(self.binary_path),
                "--elevation-dir",
                str(self.elevation_dir),
                "--recreation-db",
                str(self.recreation_db),
                "notch-scan",
                "--lat",
                str(lat),
                "--lon",
                str(lon),
                "--near-grid-m",
                str(near_grid_m),
                "--az-step-deg",
                str(az_step_deg),
                "--year",
                str(year),
                "--json-out",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return json.loads(tmp_path.read_text())
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    def plan_trail_itineraries(
        self,
        landmark: str = "Sentinel Dome",
        max_miles: float = 12.0,
        year: int = 2026,
    ) -> List[Dict[str, Any]]:
        """
        Queries `recreation/db.sqlite3` for routes matching `landmark`, samples vantage
        points along the trail geometry, detects Light Notches in parallel, and computes
        time-reversed trailhead departure schedules via Tobler's Hiking Function.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cmd = [
                str(self.binary_path),
                "--elevation-dir",
                str(self.elevation_dir),
                "--recreation-db",
                str(self.recreation_db),
                "trail-plan",
                "--landmark",
                landmark,
                "--max-miles",
                str(max_miles),
                "--year",
                str(year),
                "--json-out",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return json.loads(tmp_path.read_text())
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

    def shoot_landmark(
        self,
        landmark: str = "Half Dome",
        date: str = "2026-10-15",
        max_miles: float = 16.0,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Solves: 'I want to take a photo of <landmark> on <date>' — returns the Top K
        distinct trail viewpoints (VP1, VP2, VP3...), hikes from trailheads, sightline
        geometry, and golden-hour departure/arrival windows.
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            cmd = [
                str(self.binary_path),
                "--elevation-dir",
                str(self.elevation_dir),
                "--recreation-db",
                str(self.recreation_db),
                "shoot-landmark",
                "--landmark",
                landmark,
                "--date",
                date,
                "--max-miles",
                str(max_miles),
                "--top-k",
                str(top_k),
                "--json-out",
                str(tmp_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return json.loads(tmp_path.read_text())
        finally:
            if tmp_path.exists():
                os.remove(tmp_path)

