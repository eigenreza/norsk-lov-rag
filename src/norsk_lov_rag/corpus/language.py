"""Tell bokmål from nynorsk with a small list of common words that differ between the two.

Lovdata marks many documents as ``nb`` or ``nn``, but most regulations only say ``no``. The
detector counts words that exist in one written standard and not the other. It is used for the
documents without a clear mark, and checked against the documents that have one.
"""

from __future__ import annotations

import re

# Pairs of common words: the bokmål form and the nynorsk form.
WORD_PAIRS = (
    ("ikke", "ikkje"),
    ("et", "eit"),
    ("en", "ein"),
    ("fra", "frå"),
    ("de", "dei"),
    ("hva", "kva"),
    ("hvordan", "korleis"),
    ("bare", "berre"),
    ("noen", "nokon"),
    ("noe", "noko"),
    ("være", "vere"),
    ("ble", "vart"),
    ("virksomhet", "verksemd"),
    ("ansatte", "tilsette"),
    ("kommunene", "kommunane"),
    ("tjeneste", "teneste"),
    ("tjenester", "tenester"),
    ("selv", "sjølv"),
    ("hører", "høyrer"),
    ("krever", "krev"),
    ("tillatelse", "løyve"),
    ("loven", "lova"),
    ("bestemmelse", "føresegn"),
    ("bestemmelser", "føresegner"),
    ("bestemmelsene", "føresegnene"),
    ("forskriften", "forskrifta"),
    ("gir", "gjev"),
    ("hjemmel", "heimel"),
    ("medhold", "medhald"),
    ("uten", "utan"),
    ("sammen", "saman"),
    ("mye", "mykje"),
    ("mer", "meir"),
    ("flere", "fleire"),
    ("enkelte", "einskilde"),
    ("hver", "kvar"),
    ("hvert", "kvart"),
    ("hvem", "kven"),
    ("nevnte", "nemnde"),
    ("nevnt", "nemnt"),
    ("følgende", "følgjande"),
    ("regnes", "reknast"),
    ("myndighet", "styresmakt"),
    ("myndighetene", "styresmaktene"),
    ("eiendom", "eigedom"),
    ("eier", "eigar"),
    ("virkning", "verknad"),
    ("opplysninger", "opplysningar"),
)
BOKMAL_WORDS = frozenset(pair[0] for pair in WORD_PAIRS)
NYNORSK_WORDS = frozenset(pair[1] for pair in WORD_PAIRS)
WORD = re.compile(r"[a-zæøåA-ZÆØÅ]+")


def marker_counts(text: str) -> tuple[int, int]:
    """Number of bokmål marker words and nynorsk marker words in the text."""
    bokmal = nynorsk = 0
    for match in WORD.finditer(text):
        word = match.group().lower()
        if word in BOKMAL_WORDS:
            bokmal += 1
        elif word in NYNORSK_WORDS:
            nynorsk += 1
    return bokmal, nynorsk


def detect_written_standard(text: str, min_markers: int = 5) -> str | None:
    """``nb`` or ``nn`` for the text, or None when there are too few marker words to tell."""
    bokmal, nynorsk = marker_counts(text)
    if bokmal + nynorsk < min_markers:
        return None
    return "nn" if nynorsk > bokmal else "nb"


def written_standard(declared: str | None, text: str) -> str:
    """The document's written standard: Lovdata's mark when it is clear, else the detector."""
    declared = (declared or "").strip().lower()
    if declared in ("nb", "nn"):
        return declared
    return detect_written_standard(text) or "nb"
