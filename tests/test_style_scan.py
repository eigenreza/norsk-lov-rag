"""The style scan flags em dashes and sentence en dashes, and nothing else."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("style_scan", ROOT / "scripts" / "style_scan.py")
style_scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style_scan)

EM = chr(0x2014)
EN = chr(0x2013)


def test_em_dash_is_flagged_anywhere() -> None:
    assert style_scan.findings_in(f"first line\nword{EM}word") == [(2, "em dash")]


def test_en_dash_between_words_with_spaces_is_flagged() -> None:
    assert style_scan.findings_in(f"this {EN} that") == [(1, "en dash used as a sentence dash")]
    assert style_scan.findings_in(f"{EN} a line that opens with one") == [
        (1, "en dash used as a sentence dash")
    ]


def test_en_dash_in_a_range_and_plain_hyphens_pass() -> None:
    assert style_scan.findings_in(f"kapittel 5{EN}7, paragraf 6-44, a - b") == []


def test_repository_passes_the_scan() -> None:
    assert style_scan.main() == 0
