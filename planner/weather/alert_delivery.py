"""
Calibrated Alert Delivery Engine (`planner/weather/alert_delivery.py`).

Delivers triggered PhotoPlanner Watchlist Alerts via:
  1. **Resend Email API** (`RESEND_API_KEY` + `PHOTOPLANNER_FROM_EMAIL`)
  2. **SMTP Email** (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`)
  3. **Webhook / ntfy / Discord / Slack** (`notify_webhook_url` on the rule)
  4. **Dry-Run Mode** (`dry_run=True` renders exact subject/body/URLs without sending)

Enforces strict `(rule_id, date, phase)` deduplication via `mark_watchlist_notified`
so photographers never receive duplicate alerts for the same golden hour.
"""

from __future__ import annotations

from email.message import EmailMessage
import html
import json
import os
from pathlib import Path
import smtplib
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .alert_engine import (
    DEFAULT_WATCHLIST_DB,
    evaluate_watchlist_rule_live,
    list_watchlist_rules,
    mark_watchlist_notified,
)


def build_alert_message(
    rule: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    date_str: str,
    base_url: str = "https://photoplanner.app",
) -> dict[str, str]:
    """Formats a crisp LightFinder photographer-ready subject, plain text, HTML email, and Viewfinder deep link."""
    spot_name = str(evaluation.get("spot_name") or rule.get("name") or "Target Spot")
    phase = str(evaluation.get("phase") or rule.get("phase") or "sunset").lower()
    best_hour = evaluation.get("best_hour")
    best_time = evaluation.get("best_time_local")
    if phase == "sunrise":
        phase_label = "Sunrise"
        link_phase = "sunrise"
    elif phase == "any":
        phase_label = f"24h Window ({best_time})" if best_time else "24h Window"
        link_phase = "sunrise" if (isinstance(best_hour, int) and best_hour < 12) else "sunset"
    else:
        phase_label = "Sunset"
        link_phase = "sunset"

    decision = str(evaluation.get("decision") or "TRIGGER").upper()
    p10, p50, p90 = evaluation.get("calibrated_p10_p50_p90") or [0.0, 0.0, 0.0]
    opp = int(round(float(evaluation.get("opportunity_score") or 0)))
    reason = str(
        evaluation.get("reason")
        or evaluation.get("headline")
        or evaluation.get("causal_attribution")
        or ""
    )
    phen = str(evaluation.get("primary_phenomenon") or "Golden Hour Light")

    target_archetypes = evaluation.get("target_archetypes") or rule.get("target_archetypes") or []
    matched_arch_id = evaluation.get("matched_archetype_id")
    matched_arch_name = str(evaluation.get("matched_archetype_name") or "")
    combined_arch_pct = evaluation.get("combined_target_prob_pct")
    if combined_arch_pct is None:
        combined_arch_pct = evaluation.get("matched_archetype_prob_pct")
    field_ref = evaluation.get("field_reference") or {}
    tactical_lever = str(evaluation.get("tactical_lever") or field_ref.get("tactical_lever") or "")
    viable_list = (evaluation.get("light_archetypes") or {}).get("viable") or []
    is_archetype_alert = bool(target_archetypes and matched_arch_id and combined_arch_pct is not None)

    params: dict[str, str] = {"name": spot_name, "date": date_str, "phase": link_phase}
    if rule.get("lat") is not None and rule.get("lon") is not None:
        params["lat"] = f"{float(rule['lat']):.5f}"
        params["lon"] = f"{float(rule['lon']):.5f}"
    if matched_arch_id:
        params["archetype"] = str(matched_arch_id)
    if best_hour is not None:
        params["hour"] = str(int(best_hour))
    deep_link = f"{base_url.rstrip('/')}/?{urllib.parse.urlencode(params)}"

    if is_archetype_alert:
        comb_val = float(combined_arch_pct)
        subject = f"[{decision}] {spot_name} {phase_label} ({date_str}): {comb_val:.0f}% {matched_arch_name}"
        viable_summary = ", ".join(
            f"{v.get('name')} {int(round(float(v.get('probability_pct', 0))))}%"
            for v in viable_list[:3]
            if v.get("name")
        )
        text_lines = [
            f"LightFinder Calibrated Light Alert ({decision})",
            f"Spot: {spot_name} ({date_str} · {phase_label})",
            f"Matched Light Archetype: {matched_arch_name} ({comb_val:.0f}% target odds)",
        ]
        if viable_summary:
            text_lines.append(f"Viable Scenarios (≥20%): {viable_summary}")
        if tactical_lever:
            text_lines.append(f"Tactical Lever: {tactical_lever}")
        if field_ref:
            text_lines.append(
                f"Field Reference: {field_ref.get('shadow_edge', '')} · "
                f"{field_ref.get('contrast', '')} · {field_ref.get('color_cast', '')}"
            )
        text_lines.extend(
            [
                f"Calibrated Corridor Beam: P50 {p50:.0f}% (P10 {p10:.0f}% – P90 {p90:.0f}%)",
                f"Why: {reason}",
                "",
                f"Open in 3D Lens Viewfinder: {deep_link}",
                "",
            ]
        )
        text_body = "\n".join(text_lines)

        ref_html = ""
        if field_ref or tactical_lever:
            ref_html = f"""
    <div style="background:#0f172a;border:1px solid #24324a;border-radius:10px;padding:12px 14px;margin-bottom:14px;font-size:12px;color:#cbd5e1;">
      <div style="font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#94a3b8;margin-bottom:6px;">Field Reference · {html.escape(matched_arch_name)}</div>
      <div><strong>Shadow Edge:</strong> {html.escape(str(field_ref.get('shadow_edge', '—')))} · <strong>Contrast:</strong> {html.escape(str(field_ref.get('contrast', '—')))}</div>
      <div style="margin-top:3px;"><strong>Color Cast:</strong> {html.escape(str(field_ref.get('color_cast', '—')))}</div>
      <div style="margin-top:6px;color:#fde68a;"><strong>Lever:</strong> {html.escape(tactical_lever)}</div>
    </div>"""
        viable_html = (
            f'<div style="font-size:12px;color:#cbd5e1;margin-top:5px;">Viable (≥20%): {html.escape(viable_summary)}</div>'
            if viable_summary
            else ""
        )
        html_body = f"""<!DOCTYPE html>
<html lang="en">
<body style="margin:0;padding:24px;background:#0a0f1a;color:#f1f5f9;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;">
  <div style="max-width:560px;margin:0 auto;background:#111827;border:1px solid #24324a;border-radius:14px;padding:22px;">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;color:#f59e0b;">
      LightFinder · Calibrated {html.escape(decision)} Alert
    </div>
    <h1 style="margin:6px 0 8px;font-size:20px;color:#f8fafc;">{html.escape(spot_name)} — {html.escape(phase_label)}</h1>
    <p style="margin:0 0 14px;font-size:13px;color:#94a3b8;">{html.escape(date_str)} · {html.escape(matched_arch_name)}</p>
    <div style="background:#162033;border:1px solid #24324a;border-radius:10px;padding:14px;margin-bottom:14px;">
      <div style="font-size:24px;font-weight:800;color:#fde68a;">{comb_val:.0f}% {html.escape(matched_arch_name)}</div>
      <div style="font-size:12px;color:#94a3b8;margin-top:4px;">Corridor Beam: P50 {p50:.0f}% (P10 {p10:.0f}% – P90 {p90:.0f}%)</div>
      {viable_html}
    </div>{ref_html}
    <p style="margin:0 0 18px;font-size:14px;line-height:1.5;color:#e2e8f0;">{html.escape(reason)}</p>
    <a href="{html.escape(deep_link)}" style="display:inline-block;padding:10px 16px;border-radius:8px;background:#f59e0b;color:#0a0f1a;font-weight:700;font-size:13px;text-decoration:none;">
      Open 3D Lens Viewfinder
    </a>
  </div>
</body>
</html>"""
    else:
        subject = f"[{decision}] {spot_name} {phase_label} ({date_str}): {p50:.0f}% Light Odds — {phen}"
        arch_line = (
            f"Dominant Archetype: {matched_arch_name} ({float(combined_arch_pct):.0f}%)\n"
            if matched_arch_name and combined_arch_pct is not None
            else ""
        )
        lever_line = f"Tactical Lever: {tactical_lever}\n" if tactical_lever else ""
        text_body = (
            f"LightFinder Calibrated Light Alert ({decision})\n"
            f"Spot: {spot_name} ({date_str} · {phase_label})\n"
            f"Phenomenon: {phen} (Opportunity {opp}/100)\n"
            f"{arch_line}{lever_line}"
            f"Calibrated Beam Odds: P50 {p50:.0f}% (80% band: P10 {p10:.0f}% – P90 {p90:.0f}%)\n"
            f"Why: {reason}\n\n"
            f"Open in 3D Lens Viewfinder: {deep_link}\n"
        )
        html_body = f"""<!DOCTYPE html>
<html lang="en">
<body style="margin:0;padding:24px;background:#0a0f1a;color:#f1f5f9;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;">
  <div style="max-width:560px;margin:0 auto;background:#111827;border:1px solid #24324a;border-radius:14px;padding:22px;">
    <div style="font-size:11px;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;color:#f59e0b;">
      LightFinder · Calibrated {html.escape(decision)} Alert
    </div>
    <h1 style="margin:6px 0 8px;font-size:20px;color:#f8fafc;">{html.escape(spot_name)} — {html.escape(phase_label)}</h1>
    <p style="margin:0 0 14px;font-size:13px;color:#94a3b8;">{html.escape(date_str)} · {html.escape(phen)}</p>
    <div style="background:#162033;border:1px solid #24324a;border-radius:10px;padding:14px;margin-bottom:14px;">
      <div style="font-size:24px;font-weight:800;color:#fde68a;">P50 {p50:.0f}% <span style="font-size:13px;font-weight:600;color:#94a3b8;">(P10 {p10:.0f}% – P90 {p90:.0f}%)</span></div>
      <div style="font-size:12px;color:#cbd5e1;margin-top:4px;">Opportunity Score: <strong>{opp}/100</strong></div>
    </div>
    <p style="margin:0 0 18px;font-size:14px;line-height:1.5;color:#e2e8f0;">{html.escape(reason)}</p>
    <a href="{html.escape(deep_link)}" style="display:inline-block;padding:10px 16px;border-radius:8px;background:#f59e0b;color:#0a0f1a;font-weight:700;font-size:13px;text-decoration:none;">
      Open 3D Lens Viewfinder
    </a>
  </div>
</body>
</html>"""
    return {
        "subject": subject,
        "text": text_body,
        "html": html_body,
        "deep_link": deep_link,
    }


def _send_email_resend(to_email: str, msg: dict[str, str]) -> dict[str, Any]:
    api_key = os.environ.get("RESEND_API_KEY", "").strip()
    from_addr = (
        os.environ.get("LIGHTFINDER_FROM_EMAIL", "").strip()
        or os.environ.get("PHOTOPLANNER_FROM_EMAIL", "alerts@photoplanner.app").strip()
    )
    payload = json.dumps(
        {
            "from": from_addr,
            "to": [to_email],
            "subject": msg["subject"],
            "text": msg["text"],
            "html": msg["html"],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "LightFinder/2.0",
        },
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return {"channel": "resend", "status": resp.status, "body": resp.read().decode("utf-8", errors="ignore")[:240]}


def _send_email_smtp(to_email: str, msg: dict[str, str]) -> dict[str, Any]:
    host = os.environ.get("SMTP_HOST", "").strip()
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASS", "").strip()
    from_addr = (
        os.environ.get("LIGHTFINDER_FROM_EMAIL", "").strip()
        or os.environ.get("PHOTOPLANNER_FROM_EMAIL", user or "alerts@photoplanner.app").strip()
    )

    email = EmailMessage()
    email["Subject"] = msg["subject"]
    email["From"] = from_addr
    email["To"] = to_email
    email.set_content(msg["text"])
    email.add_alternative(msg["html"], subtype="html")

    with smtplib.SMTP(host, port, timeout=10.0) as srv:
        srv.starttls()
        if user and password:
            srv.login(user, password)
        srv.send_message(email)
    return {"channel": "smtp", "status": 250}


def _send_webhook(webhook_url: str, rule: dict[str, Any], evaluation: dict[str, Any], msg: dict[str, str]) -> dict[str, Any]:
    is_ntfy = "ntfy.sh" in webhook_url.lower()
    if is_ntfy:
        req = urllib.request.Request(
            webhook_url,
            data=msg["text"].encode("utf-8"),
            method="POST",
            headers={
                "Title": msg["subject"][:120],
                "Click": msg["deep_link"],
                "Tags": "camera,sunrise",
                "User-Agent": "LightFinder/2.0",
            },
        )
    else:
        # Compatible with Discord (`content`), Slack (`text`), and generic JSON webhooks
        reason_txt = str(evaluation.get("reason") or evaluation.get("headline") or "")
        body = json.dumps(
            {
                "content": f"**{msg['subject']}**\n{reason_txt}\n{msg['deep_link']}",
                "text": f"*{msg['subject']}*\n{reason_txt}\n{msg['deep_link']}",
                "rule_id": rule["id"],
                "decision": evaluation.get("decision"),
                "spot_name": evaluation.get("spot_name"),
                "matched_archetype_id": evaluation.get("matched_archetype_id"),
                "matched_archetype_name": evaluation.get("matched_archetype_name"),
                "combined_target_prob_pct": evaluation.get("combined_target_prob_pct"),
                "tactical_lever": evaluation.get("tactical_lever"),
                "light_archetypes": evaluation.get("light_archetypes"),
                "calibrated_p10_p50_p90": evaluation.get("calibrated_p10_p50_p90"),
                "deep_link": msg["deep_link"],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "LightFinder/2.0"},
        )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return {"channel": "webhook", "status": resp.status}


def deliver_alert_for_evaluation(
    rule: dict[str, Any],
    evaluation: dict[str, Any],
    *,
    date_str: str,
    base_url: str = "https://photoplanner.app",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Delivers a single triggered alert over configured email and/or webhook channels."""
    msg = build_alert_message(rule, evaluation, date_str=date_str, base_url=base_url)
    deliveries: list[dict[str, Any]] = []
    errors: list[str] = []

    to_email = rule.get("notify_email")
    webhook_url = rule.get("notify_webhook_url")

    if dry_run:
        channels = []
        if to_email:
            channels.append(f"email:{to_email}")
        if webhook_url:
            channels.append("webhook")
        return {
            "delivered": bool(channels),
            "dry_run": True,
            "channels": channels,
            "subject": msg["subject"],
            "deep_link": msg["deep_link"],
            "errors": [],
        }

    if to_email:
        try:
            if os.environ.get("RESEND_API_KEY", "").strip():
                deliveries.append(_send_email_resend(to_email, msg))
            elif os.environ.get("SMTP_HOST", "").strip():
                deliveries.append(_send_email_smtp(to_email, msg))
            else:
                errors.append("Email configured on rule, but neither RESEND_API_KEY nor SMTP_HOST is set.")
        except Exception as exc:
            errors.append(f"Email delivery failed: {exc}")

    if webhook_url:
        try:
            deliveries.append(_send_webhook(webhook_url, rule, evaluation, msg))
        except Exception as exc:
            errors.append(f"Webhook delivery failed: {exc}")

    return {
        "delivered": len(deliveries) > 0,
        "dry_run": False,
        "deliveries": deliveries,
        "subject": msg["subject"],
        "deep_link": msg["deep_link"],
        "errors": errors,
    }


def dispatch_due_alerts(
    *,
    date_str: str,
    include_watch: bool = False,
    dry_run: bool = False,
    base_url: str = "https://photoplanner.app",
    db_path: Path | None = None,
    evaluator_fn: Any = None,
) -> dict[str, Any]:
    """
    Evaluates all enabled watchlists, deduplicates already-notified `(rule_id, date, phase)`
    alerts, and delivers `TRIGGER` (and optionally `WATCH`) alerts.
    """
    from . import alert_engine as _ae

    resolved_db = db_path or _ae.DEFAULT_WATCHLIST_DB
    rules = [r for r in list_watchlist_rules(db_path=resolved_db) if r.get("enabled")]
    eval_fn = evaluator_fn or evaluate_watchlist_rule_live
    allowed_decisions = {"TRIGGER", "WATCH"} if include_watch else {"TRIGGER"}

    sent: list[dict[str, Any]] = []
    skipped_dedup: list[str] = []
    suppressed: list[dict[str, Any]] = []

    for rule in rules:
        evals = eval_fn(rule, date_str=date_str)
        for ev in evals:
            ev_phase = str(ev.get("phase") or rule.get("phase") or "sunset").lower()
            decision = str(ev.get("decision") or "SUPPRESS").upper()
            if decision not in allowed_decisions:
                suppressed.append({"rule_id": rule["id"], "spot_name": ev.get("spot_name"), "decision": decision})
                continue
            if (
                not dry_run
                and rule.get("last_notified_date") == date_str
                and (rule.get("last_notified_phase") or "").lower() == ev_phase
            ):
                skipped_dedup.append(rule["id"])
                continue
            outcome = deliver_alert_for_evaluation(
                rule, ev, date_str=date_str, base_url=base_url, dry_run=dry_run
            )
            if outcome["delivered"] and not dry_run:
                mark_watchlist_notified(rule["id"], date_str, ev_phase, db_path=resolved_db)
            sent_item = {"rule_id": rule["id"], "spot_name": ev.get("spot_name"), "decision": decision, **outcome}
            if ev.get("matched_archetype_id"):
                sent_item["matched_archetype_id"] = ev.get("matched_archetype_id")
                sent_item["matched_archetype_name"] = ev.get("matched_archetype_name")
                sent_item["combined_target_prob_pct"] = ev.get("combined_target_prob_pct")
            sent.append(sent_item)

    return {
        "date": date_str,
        "dry_run": dry_run,
        "n_rules_checked": len(rules),
        "n_sent": len([s for s in sent if s.get("delivered")]),
        "n_skipped_dedup": len(skipped_dedup),
        "n_suppressed": len(suppressed),
        "sent": sent,
        "skipped_dedup": skipped_dedup,
    }
