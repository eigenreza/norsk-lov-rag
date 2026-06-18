"""Point references at provisions in the tables.

A reference names a section, sometimes a paragraph and a letter, and sometimes the act. The
resolver finds the document first: the act that is named, the document the reference stands in,
or the act a regulation is based on when the text says ``loven``. It then looks up the section
and, when one is given, the paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from norsk_lov_rag.references import parser
from norsk_lov_rag.references.parser import Reference

# Folders whose documents can be named in a reference, most likely first.
NAME_FOLDERS = ("nl", "sf", "stv", "ins", "del")


@dataclass(frozen=True)
class Target:
    resolved: bool
    reason: str
    doc_key: str | None = None
    section_id: str | None = None
    provision_id: str | None = None

    @property
    def level(self) -> str:
        if not self.resolved:
            return "none"
        return "paragraph" if self.provision_id else "section"


def name_forms(name: str) -> set[str]:
    """Spellings of an act name: bokmål and nynorsk definite forms, with and without ``s``."""
    name = name.lower().strip()
    forms = {name}
    for bokmal, nynorsk in (("loven", "lova"), ("forskriften", "forskrifta")):
        if name.endswith(bokmal):
            forms.add(name[: -len(bokmal)] + nynorsk)
        if name.endswith(nynorsk):
            forms.add(name[: -len(nynorsk)] + bokmal)
    return forms


class Resolver:
    """Lookup tables built once from the document, section and provision tables."""

    def __init__(
        self, documents: pd.DataFrame, sections: pd.DataFrame, provisions: pd.DataFrame
    ) -> None:
        self.sections: dict[str, dict[str, str]] = {}
        self.order: dict[str, list[str]] = {}
        for doc_key, section_id, number in zip(
            sections["doc_key"], sections["section_id"], sections["section_number"], strict=True
        ):
            self.sections.setdefault(doc_key, {}).setdefault(str(number).lower(), section_id)
            self.order.setdefault(doc_key, []).append(section_id)
        self.position = {
            section_id: (doc_key, index)
            for doc_key, ids in self.order.items()
            for index, section_id in enumerate(ids)
        }
        paragraphs = provisions[provisions["paragraph"].notna()]
        self.paragraphs = set(paragraphs["provision_id"])
        self.kind = dict(zip(documents["doc_key"], documents["kind"], strict=True))
        self.parent_law = {
            doc_key: next((key for key in based_on if key.startswith("lov/")), None)
            for doc_key, based_on in zip(documents["doc_key"], documents["based_on"], strict=True)
        }
        self.names: dict[str, str | None] = {}
        self.abbreviations: dict[str, str | None] = {}
        rank = {folder: index for index, folder in enumerate(NAME_FOLDERS)}
        ordered = documents.assign(rank=documents["folder"].map(rank)).sort_values("rank")
        for doc_key, name, abbreviation in zip(
            ordered["doc_key"], ordered["short_name"], ordered["abbreviation"], strict=True
        ):
            if name:
                for form in name_forms(name):
                    self._register(self.names, form, doc_key)
            if abbreviation:
                self._register(self.abbreviations, abbreviation.lower().rstrip("."), doc_key)

    @staticmethod
    def _register(table: dict[str, str | None], key: str, doc_key: str) -> None:
        """Keep the first document for a name. A name shared by two acts is marked ambiguous."""
        if key not in table:
            table[key] = doc_key
        elif table[key] != doc_key and table[key] is not None:
            first = table[key]
            if first.startswith("lov/") and not doc_key.startswith("lov/"):
                return
            if first.split("-")[0] == doc_key.split("-")[0] and doc_key.endswith(("-nn", "-nb")):
                return
            table[key] = None

    def document_for(self, reference: Reference, source_doc_key: str) -> tuple[str | None, str]:
        """The document a reference points into, and how it was found."""
        kind = reference.act_kind
        if kind in (parser.ACT_NONE, parser.ACT_SELF, parser.ACT_SAME_SECTION):
            return source_doc_key, "same_document"
        if kind == parser.ACT_CITATION:
            if reference.act_key in self.kind:
                return reference.act_key, "citation"
            return None, "cited_document_not_in_corpus"
        if kind == parser.ACT_NAMED:
            if reference.act not in self.names:
                return None, "unknown_act_name"
            doc_key = self.names[reference.act]
            return (doc_key, "named") if doc_key else (None, "ambiguous_act_name")
        if kind == parser.ACT_ABBREVIATION:
            doc_key = self.abbreviations.get(reference.act or "")
            # Most short words before a section sign are not acts ("jf.", "se", "mv.").
            return (doc_key, "abbreviation") if doc_key else (source_doc_key, "same_document")
        if kind == parser.ACT_THE_LAW:
            if self.kind.get(source_doc_key) == "lov":
                return source_doc_key, "same_document"
            parent = self.parent_law.get(source_doc_key)
            if parent in self.kind:
                return parent, "parent_law"
            return None, "parent_law_unknown"
        if kind == parser.ACT_THE_REGULATION:
            if self.kind.get(source_doc_key) != "lov":
                return source_doc_key, "same_document"
            return None, "regulation_not_named"
        return None, "unknown_reference_kind"

    def resolve(
        self,
        reference: Reference,
        source_doc_key: str,
        source_section_id: str | None,
        last_named: str | None = None,
    ) -> Target:
        if reference.act_kind == parser.ACT_SAME_SECTION:
            if not source_section_id:
                return Target(False, "no_section_around_reference")
            provision_id = f"{source_section_id}/ledd/{reference.paragraph}"
            if provision_id in self.paragraphs:
                return Target(True, "same_section", source_doc_key, source_section_id, provision_id)
            return Target(False, "paragraph_not_found", source_doc_key, source_section_id)
        doc_key, how = self.document_for(reference, source_doc_key)
        if doc_key is None:
            return Target(False, how)
        number = (reference.section or "").lower()
        section_id = self.sections.get(doc_key, {}).get(number)
        if section_id is None and reference.act_kind == parser.ACT_NONE:
            # A bare section number that the document itself does not have: try the act named
            # last in the same provision, then the act a regulation is based on. A reference
            # that says "denne lov" or "denne forskrift" never moves to another act.
            fallbacks = (
                (last_named, "last_named_act"),
                (self.parent_law.get(source_doc_key), "parent_law_fallback"),
            )
            for candidate, label in fallbacks:
                if (
                    candidate
                    and candidate != doc_key
                    and number in self.sections.get(candidate, {})
                ):
                    doc_key, how = candidate, label
                    section_id = self.sections[candidate][number]
                    break
        if section_id is None:
            return Target(False, "section_not_found", doc_key)
        if reference.paragraph is None:
            return Target(True, how, doc_key, section_id)
        provision_id = f"{section_id}/ledd/{reference.paragraph}"
        if provision_id in self.paragraphs:
            return Target(True, how, doc_key, section_id, provision_id)
        return Target(True, "paragraph_not_found_section_used", doc_key, section_id)

    def between(self, first_section_id: str, last_section_id: str) -> list[str]:
        """Sections strictly between two sections of the same document, in document order."""
        first, last = self.position.get(first_section_id), self.position.get(last_section_id)
        if first is None or last is None or first[0] != last[0] or last[1] <= first[1]:
            return []
        return self.order[first[0]][first[1] + 1 : last[1]]
