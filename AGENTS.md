# PhotoPlanner Workspace Rules

## GitHub Sync & Push Protocol (VM ↔ Local Preview Workflow)

This repository is developed collaboratively across a remote VM agent and a local Mac preview server synced via `origin/main`. Follow these rules on every task:

1. **Pull before starting work**:
   - Before editing files, sync with GitHub:
     ```bash
     git fetch origin && git pull --rebase --autostash origin main
     ```
2. **Verify before committing**:
   - JS changes (`web/`): run `node --check` on modified `.js` files.
   - Python changes (`api_server.py`, `server.py`, `planner/`): run `python3 -m py_compile` on modified files and relevant `pytest` suites.
   - Rust changes (`raycaster/`): run `cargo check --manifest-path raycaster/Cargo.toml` (or `cargo build --release`).
3. **Always commit and push to GitHub when done**:
   - Stage and commit all intentional changes with a concise conventional commit message (e.g., `feat(...)`, `fix(...)`, `perf(...)`).
   - Never leave completed work uncommitted or unpushed on the VM.
   - Rebase if `origin/main` moved while you were working, then push:
     ```bash
     git pull --rebase origin main && git push origin main
     ```
4. **Keep preview machines clean**:
   - On the local Mac preview host, prefer running `./scripts/dev-sync.sh` so `origin/main` pulls automatically without accumulating conflicting local edits.
