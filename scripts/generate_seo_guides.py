#!/usr/bin/env python3
"""
Pre-builds daily static SEO Light Guides (`guides/index.html`, `guides/<slug>/index.html`,
`sitemap.xml`, `robots.txt`) for zero-cost CDN / Cloudflare Pages / Vercel deployment.

Usage:
  .venv/bin/python scripts/generate_seo_guides.py --out-dir dist/seo --base-url https://photoplanner.app
  .venv/bin/python scripts/generate_seo_guides.py --out-dir dist/seo --region bay_area --live
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planner.seo_guides import (
    build_spot_guide_payload,
    list_seo_guides,
    render_guides_directory_html,
    render_robots_txt,
    render_sitemap_xml,
    render_spot_guide_html,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate static SEO Light Guide HTML + sitemap.xml + robots.txt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist" / "seo", help="Output directory")
    parser.add_argument("--base-url", type=str, default="https://photoplanner.app", help="Canonical site base URL")
    parser.add_argument("--date", type=str, default=None, help="Forecast date YYYY-MM-DD (defaults to today local)")
    parser.add_argument("--region", type=str, default=None, help="Optional single region filter")
    parser.add_argument("--live", action="store_true", help="Hydrate live NWP/corridor forecasts for each spot")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    guides_dir = out_dir / "guides"
    guides_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "sitemap.xml").write_text(
        render_sitemap_xml(base_url=args.base_url, date_str=args.date),
        encoding="utf-8",
    )
    (out_dir / "robots.txt").write_text(
        render_robots_txt(base_url=args.base_url),
        encoding="utf-8",
    )
    (guides_dir / "index.html").write_text(
        render_guides_directory_html(region=args.region, date_str=args.date, base_url=args.base_url),
        encoding="utf-8",
    )

    spots = list_seo_guides(region=args.region)
    for idx, item in enumerate(spots, 1):
        payload = build_spot_guide_payload(
            item["slug"],
            date_str=args.date,
            include_live_forecast=bool(args.live),
            base_url=args.base_url,
        )
        spot_dir = guides_dir / item["slug"]
        spot_dir.mkdir(parents=True, exist_ok=True)
        (spot_dir / "index.html").write_text(render_spot_guide_html(payload), encoding="utf-8")
        print(f"[{idx:02d}/{len(spots):02d}] wrote {spot_dir / 'index.html'} (hydrated={payload['forecast_hydrated']})")

    print(f"Done: {len(spots)} Light Guides + sitemap.xml + robots.txt written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
