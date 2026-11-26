#!/usr/bin/env python3
"""
Antigravity Stop hook: enforces commit + push to GitHub on VM hosts before ending a turn.

Activation policy:
- Enabled automatically on Linux (VM environment) or whenever PHOTOPLANNER_ENFORCE_PUSH=1.
- Disabled on macOS/Windows by default unless PHOTOPLANNER_ENFORCE_PUSH=1, or anywhere when PHOTOPLANNER_ENFORCE_PUSH=0.
- Only triggers on normal turn completion (`NO_TOOL_CALL`) when `continuationActive` is false (so it never loops).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys


TRACKED_SOURCE_PREFIXES = (
    "web/",
    "planner/",
    "raycaster/",
    "scripts/",
    "tests/",
    ".agents/",
    "AGENTS.md",
    "api_server.py",
    "server.py",
)


def _git(repo_root: Path, *args: str, timeout: float = 5.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _should_enforce() -> bool:
    flag = os.environ.get("PHOTOPLANNER_ENFORCE_PUSH", "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return platform.system() == "Linux"


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        print(json.dumps({"decision": "allow"}))
        return

    if not _should_enforce():
        print(json.dumps({"decision": "allow"}))
        return

    if payload.get("terminationReason") != "NO_TOOL_CALL":
        print(json.dumps({"decision": "allow"}))
        return

    if payload.get("continuationActive"):
        print(json.dumps({"decision": "allow"}))
        return

    cwd_str = payload.get("cwd")
    repo_root = Path(cwd_str) if cwd_str else Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        repo_root = Path(__file__).resolve().parents[2]

    try:
        branch_res = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        branch = branch_res.stdout.strip()
        if branch_res.returncode != 0 or not branch or branch == "HEAD":
            print(json.dumps({"decision": "allow"}))
            return

        status_res = _git(repo_root, "status", "--porcelain")
        dirty_tracked: list[str] = []
        untracked_source: list[str] = []
        for line in status_res.stdout.splitlines():
            if len(line) < 4:
                continue
            code = line[:2]
            path = line[3:].strip()
            if code == "??":
                if path.startswith(TRACKED_SOURCE_PREFIXES):
                    untracked_source.append(path)
            else:
                dirty_tracked.append(path)

        ahead_res = _git(repo_root, "rev-list", "--count", "@{u}..HEAD")
        ahead = int(ahead_res.stdout.strip()) if ahead_res.returncode == 0 and ahead_res.stdout.strip().isdigit() else 0

        issues: list[str] = []
        if dirty_tracked:
            sample = ", ".join(dirty_tracked[:5])
            more = f" (+{len(dirty_tracked) - 5} more)" if len(dirty_tracked) > 5 else ""
            issues.append(f"{len(dirty_tracked)} modified file(s): {sample}{more}")
        if untracked_source:
            sample = ", ".join(untracked_source[:5])
            more = f" (+{len(untracked_source) - 5} more)" if len(untracked_source) > 5 else ""
            issues.append(f"{len(untracked_source)} untracked source file(s): {sample}{more}")
        if ahead > 0:
            issues.append(f"{ahead} unpushed commit(s) on `{branch}`")

        if not issues:
            print(json.dumps({"decision": "allow"}))
            return

        reason = (
            "GitHub push gate: before finishing your turn, sync your work to GitHub. "
            f"Pending: {'; '.join(issues)}. "
            "Run quick syntax/test checks on modified files, commit any intentional changes, "
            f"then run `git pull --rebase origin {branch} && git push origin {branch}`."
        )
        print(json.dumps({"decision": "continue", "reason": reason}))
    except Exception:
        print(json.dumps({"decision": "allow"}))


if __name__ == "__main__":
    main()
