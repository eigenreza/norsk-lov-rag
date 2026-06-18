"""Fail on em dashes, and on en dashes used as sentence dashes, in public files.

Corpus-derived data is not prose written for this project and is left out: everything under
``data/``, ``demo_data/`` and ``tests/fixtures/``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)
# An en dash with white space on both sides, or at the start of a line, works as a sentence dash.
SENTENCE_EN_DASH = re.compile(rf"(^|\s){EN_DASH}\s")
TEXT_SUFFIXES = {
    ".md", ".py", ".toml", ".yaml", ".yml", ".txt", ".json", ".csv", ".svg", ".html", ".cfg",
    ".bicep", ".sh",
}  # fmt: skip
SKIP_DIRS = {
    ".git", ".claude", ".venv", "venv", "data", "logs", "runs", "models", "demo_data",
    "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build",
}  # fmt: skip
SKIP_PREFIXES = ("tests/fixtures/",)
SKIP_NAMES = ("BRIEF",)


def findings_in(text: str) -> list[tuple[int, str]]:
    """Line numbers and reasons for every dash that breaks the style rule."""
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if EM_DASH in line:
            found.append((number, "em dash"))
        if SENTENCE_EN_DASH.search(line):
            found.append((number, "en dash used as a sentence dash"))
    return found


def public_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS or part.startswith(".venv") for part in relative.parts):
            continue
        if relative.as_posix().startswith(SKIP_PREFIXES) or path.name.startswith(SKIP_NAMES):
            continue
        files.append(path)
    return files


def main() -> int:
    files = public_files(ROOT)
    total = 0
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, reason in findings_in(text):
            total += 1
            print(f"{path.relative_to(ROOT).as_posix()}:{number}: {reason}")
    print(f"style scan: {len(files)} files, {total} findings")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
