"""A throwaway git repository laid out like OrbitGuard, for freeze tests (PRD §33 one-way freeze).

Freeze tests must not depend on the real checkout's git state (it is dirty while anyone is editing), so
they point `evaluation.freeze` at this repository instead. Commits bypass the user's global hooks and
signing so the fixture behaves the same on every machine.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_GIT_IDENTITY = (
    "-c", "user.name=freeze-test", "-c", "user.email=freeze-test@example.invalid",
    "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
)


def git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *_GIT_IDENTITY, *args], capture_output=True, text=True, check=True, timeout=30
    )
    return completed.stdout.strip()


def commit_all(root: Path, message: str) -> str:
    git(root, "add", "-A")
    git(root, "commit", "-q", "--no-verify", "-m", message)
    return git(root, "rev-parse", "HEAD")


def make_repo(root: Path) -> Path:
    """A committed repository with orbit_guard/, configs/, scripts/ and docs/; returns its root."""
    root.mkdir(parents=True, exist_ok=True)
    git(root, "init", "-q")
    for relative, content in {
        "orbit_guard/module.py": "VALUE = 1\n",
        "configs/default.yaml": "a: 1\n",
        "scripts/run.sh": "echo run\n",
        "docs/notes.md": "notes\n",
        ".gitignore": "__pycache__/\n",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    commit_all(root, "initial")
    return root
