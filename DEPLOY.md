# PhotoPlanner Production Deployment Guide

PhotoPlanner ships as a single self-contained container compiling the persistent Rust raycaster (`photoplanner`) alongside the FastAPI/Uvicorn ASGI server (`api_server.py`) with GZip compression, IP rate limiting, `/api/health`, `/api/ready`, dynamic SEO Light Guides, and SQLite watchlists/field reports.

---

## Option 1: Fly.io (~5 Minutes, ~\$12/mo)

```bash
# 1. Authenticate & initialize app (uses included fly.toml)
fly auth login
fly apps create photoplanner-3d

# 2. Provision persistent 10 GB volume for SRTM tiles, watchlists & field reports
fly volumes create pp_data --size 10 --region sjc

# 3. Set production secrets (all optional for initial preview; set OPEN_METEO_API_KEY before paid launch)
fly secrets set \
  OPEN_METEO_API_KEY="your_open_meteo_commercial_key" \
  RESEND_API_KEY="re_your_resend_key" \
  PHOTOPLANNER_FROM_EMAIL="alerts@yourdomain.com" \
  PHOTOPLANNER_ADMIN_TOKEN="$(openssl rand -hex 24)" \
  PHOTOPLANNER_B2B_API_KEYS="partner_key_1"

# 4. Build & deploy
fly deploy

# 5. Verify live health & readiness
curl -fsS https://photoplanner-3d.fly.dev/api/health | jq .
```

Attach your custom domain with:
```bash
fly certs create photoplanner.app
```

---

## Option 2: Hetzner / DigitalOcean VPS via Docker Compose (~\$7–\$10/mo)

```bash
git clone https://github.com/lucasbrambrink/photoplanner.git
cd photoplanner

cat <<'EOF' > .env
PORT=8420
PHOTOPLANNER_WORKERS=2
PHOTOPLANNER_RATE_LIMIT_RPM=120
OPEN_METEO_API_KEY=
RESEND_API_KEY=
PHOTOPLANNER_FROM_EMAIL=alerts@yourdomain.com
PHOTOPLANNER_ADMIN_TOKEN=change_me
EOF

docker compose up -d --build
curl -fsS http://127.0.0.1:8420/api/ready
```

Put **Caddy** in front for automatic HTTPS:
```caddyfile
photoplanner.app {
    reverse_proxy 127.0.0.1:8420
}
```

---

## Scheduled Alert Dispatch (2:00 PM & 7:00 PM Local)

Trigger calibrated watchlist alerts via cron or GitHub Actions hitting `POST /api/alerts/dispatch`:
```bash
curl -X POST "https://photoplanner.app/api/alerts/dispatch" \
  -H "X-Admin-Token: $PHOTOPLANNER_ADMIN_TOKEN"
```
Or run directly inside the container:
```bash
docker exec photoplanner python scripts/run_alert_dispatch.py --base-url https://photoplanner.app
```

---

## Environment Variables Reference

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `OPEN_METEO_API_KEY` | *(unset)* | Routes forecast & geocoding to commercial `customer-*.open-meteo.com` |
| `RESEND_API_KEY` | *(unset)* | Sends formatted HTML/text email alerts via Resend API |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASS` | *(unset)* | Alternative SMTP email delivery for watchlist alerts |
| `PHOTOPLANNER_FROM_EMAIL` | `alerts@photoplanner.app` | Sender address for email alerts |
| `PHOTOPLANNER_RATE_LIMIT_RPM` | `120` | Per-IP requests/minute cap on `/api/*` |
| `PHOTOPLANNER_B2B_API_KEYS` | *(unset)* | Comma-separated `X-API-Key` values gating `/api/v1/guides*` |
| `PHOTOPLANNER_ADMIN_TOKEN` | *(unset)* | Protects `POST /api/alerts/dispatch` via `X-Admin-Token` header |
| `PHOTOPLANNER_SRTM_MAX_TILES` | `20` | Max resident `i16` SRTM1 tiles in Rust worker LRU (~495 MB at 20 tiles) |
| `PHOTOPLANNER_ENABLE_ALERTCA_PROXY` | `0` | Commercial-safe default disables ALERTCalifornia proxying (`403`) |
