#!/usr/bin/env python3
"""Antigravity SessionStart hook: syncs clean branches with origin/main or warns if behind."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def _git(repo_root: Path, *args: str, timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}

    cwd_str = payload.get("cwd")
    repo_root = Path(cwd_str) if cwd_str else Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        repo_root = Path(__file__).resolve().parents[2]

    try:
        branch_res = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD", timeout=3.0)
        branch = branch_res.stdout.strip()
        if branch_res.returncode != 0 or branch != "main":
            print("{}")
            return

        fetch_res = _git(repo_root, "fetch", "--quiet", "origin", "main", timeout=8.0)
        if fetch_res.returncode != 0:
            print("{}")
            return

        counts_res = _git(repo_root, "rev-list", "--left-right", "--count", "HEAD...origin/main", timeout=3.0)
        if counts_res.returncode != 0:
            print("{}")
            return

        parts = counts_res.stdout.strip().split()
        ahead = int(parts[0]) if len(parts) >= 1 else 0
        behind = int(parts[1]) if len(parts) >= 2 else 0

        if behind == 0:
            print("{}")
            return

        dirty_res = _git(repo_root, "status", "--porcelain", "--untracked-files=no", timeout=3.0)
        is_clean = not dirty_res.stdout.strip()

        if is_clean and ahead == 0:
            pull_res = _git(repo_root, "pull", "--ff-only", "--quiet", "origin", "main", timeout=8.0)
            if pull_res.returncode == 0:
                tip = _git(repo_root, "log", "-1", "--oneline", timeout=3.0).stdout.strip()
                msg = f"SessionStart auto-sync: fast-forwarded `main` by {behind} commit(s) to `{tip}`."
                print(json.dumps({"injectSteps": [{"systemMessage": {"systemMessage": msg}}]}))
                return

        msg = (
            f"Git sync notice: local `main` is {behind} commit(s) behind `origin/main` "
            f"(ahead: {ahead}, clean: {is_clean}). Run `git pull --rebase --autostash origin main` "
            "before making edits so you work from the latest GitHub commit."
        )
        print(json.dumps({"injectSteps": [{"systemMessage": {"systemMessage": msg}}]}))
    except Exception:
        print("{}")


if __name__ == "__main__":
    main()
