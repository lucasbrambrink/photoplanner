#!/usr/bin/env python3
"""
Scheduled Calibrated Alert Dispatcher CLI (`scripts/run_alert_dispatch.py`).

Evaluates saved SQLite watchlists and delivers TRIGGER alerts via Resend, SMTP, or
Webhook/ntfy/Discord with strict `(rule_id, date, phase)` deduplication.

Usage:
  .venv/bin/python scripts/run_alert_dispatch.py --dry-run
  .venv/bin/python scripts/run_alert_dispatch.py --base-url https://photoplanner.app
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planner.weather.alert_delivery import dispatch_due_alerts


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch due calibrated PhotoPlanner light alerts")
    parser.add_argument("--date", type=str, default=None, help="Target date YYYY-MM-DD (default: today Pacific)")
    parser.add_argument("--base-url", type=str, default="https://photoplanner.app", help="Public app URL for deep links")
    parser.add_argument("--include-watch", action="store_true", help="Also dispatch WATCH-tier alerts")
    parser.add_argument("--dry-run", action="store_true", help="Preview alert messages without sending")
    args = parser.parse_args()

    resolved_date = args.date or dt.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    report = dispatch_due_alerts(
        date_str=resolved_date,
        include_watch=bool(args.include_watch),
        dry_run=bool(args.dry_run),
        base_url=args.base_url,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
