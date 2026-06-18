"""Build the reference graph: one edge per reference found in a provision.

The edges are compared with the links Lovdata's own editors have put in the text. Those links
are not used to build the graph. They only show how well the parser and the resolver do.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from norsk_lov_rag.references import parser
from norsk_lov_rag.references.resolve import Resolver

# Ways of finding the document that say which act the text is talking about.
NAMING_REASONS = ("named", "citation", "abbreviation", "parent_law", "last_named_act")
EDGE_COLUMNS = [
    "source_provision_id",
    "source_section_id",
    "source_doc_key",
    "raw",
    "act_kind",
    "resolved",
    "reason",
    "level",
    "scope",
    "target_doc_key",
    "target_section_id",
    "target_provision_id",
    "letter",
]


def build_edges(provisions: pd.DataFrame, resolver: Resolver) -> pd.DataFrame:
    """One row per reference. A range such as ``§§ 5-10 til 5-15`` adds the sections between."""
    rows = []
    for provision_id, section_id, doc_key, text in zip(
        provisions["provision_id"],
        provisions["section_id"],
        provisions["doc_key"],
        provisions["text"],
        strict=True,
    ):
        if "§" not in text and "ledd" not in text:
            continue
        section_id = section_id if isinstance(section_id, str) else None
        previous = None
        last_named = None
        for reference in parser.find_references(text):
            target = resolver.resolve(reference, doc_key, section_id, last_named)
            if target.resolved and target.reason in NAMING_REASONS:
                last_named = target.doc_key
            if reference.act_kind == parser.ACT_SAME_SECTION:
                scope = "same_section"
            elif target.doc_key and target.doc_key != doc_key:
                scope = "other_document"
            else:
                scope = "same_document"
            row = {
                "source_provision_id": provision_id,
                "source_section_id": section_id,
                "source_doc_key": doc_key,
                "raw": reference.raw,
                "act_kind": reference.act_kind,
                "resolved": target.resolved,
                "reason": target.reason,
                "level": target.level,
                "scope": scope,
                "target_doc_key": target.doc_key,
                "target_section_id": target.section_id,
                "target_provision_id": target.provision_id,
                "letter": reference.letter,
            }
            if reference.range_end and previous is not None and previous.section_id:
                if target.section_id:
                    for between in resolver.between(previous.section_id, target.section_id):
                        rows.append(
                            {
                                **row,
                                "reason": "inside_range",
                                "level": "section",
                                "target_section_id": between,
                                "target_provision_id": None,
                                "letter": None,
                            }
                        )
            rows.append(row)
            previous = target
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def link_section_target(href: str) -> str | None:
    """The section a Lovdata link points at, such as ``lov/1999-03-26-14/§6-13``, or None."""
    parts = href.strip().split("/")
    if len(parts) >= 3 and parts[0] in ("lov", "forskrift") and parts[2].startswith("§"):
        return "/".join(parts[:3])
    return None


def compare_with_links(edges: pd.DataFrame, links: pd.DataFrame, sections: pd.DataFrame) -> dict:
    """Agreement between the graph and Lovdata's own links, at section level per provision.

    Only links to sections that exist in the corpus count, and only references that leave the
    section they stand in.
    """
    known = set(sections["section_id"])
    link_targets = links["href"].map(link_section_target)
    linked = links.assign(target=link_targets)
    linked = linked[linked["target"].isin(known)]
    theirs = set(zip(linked["source_provision_id"], linked["target"], strict=True))
    direct = edges["reason"] != "inside_range"
    found = edges[edges["resolved"] & direct & (edges["scope"] != "same_section")]
    ours = set(zip(found["source_provision_id"], found["target_section_id"], strict=True))
    both = len(ours & theirs)
    return {
        "pairs_in_lovdata_links": len(theirs),
        "pairs_in_graph": len(ours),
        "pairs_in_both": both,
        "recall_against_links": round(both / len(theirs), 4) if theirs else None,
        "precision_against_links": round(both / len(ours), 4) if ours else None,
    }


def summarise(edges: pd.DataFrame, label: str) -> dict:
    """Counts and resolution rates for a set of edges."""
    direct = edges[edges["reason"] != "inside_range"]
    sections = direct[direct["scope"] != "same_section"]
    same = direct[direct["scope"] == "same_section"]
    return {
        "scope": label,
        "references": int(len(direct)),
        "references_to_sections": int(len(sections)),
        "resolved_share": round(float(sections["resolved"].mean()), 4) if len(sections) else None,
        "resolved_to_paragraph": int((sections["level"] == "paragraph").sum()),
        "to_other_documents": int((sections["scope"] == "other_document").sum()),
        "same_section_references": int(len(same)),
        "same_section_resolved_share": round(float(same["resolved"].mean()), 4)
        if len(same)
        else None,
        "unresolved_reasons": sections.loc[~sections["resolved"], "reason"]
        .value_counts()
        .head(8)
        .to_dict(),
        "edges_added_inside_ranges": int((edges["reason"] == "inside_range").sum()),
    }


def build_graph(processed: Path, tax_documents: list[str], full_folders: list[str]) -> dict:
    """Read the tables, build the graph, write it and return the summary."""
    documents = pd.read_parquet(processed / "documents.parquet")
    sections = pd.read_parquet(processed / "sections.parquet")
    provisions = pd.read_parquet(processed / "provisions.parquet")
    links = pd.read_parquet(processed / "links.parquet")
    edges = build_edges(provisions, Resolver(documents, sections, provisions))
    edges.to_parquet(processed / "references.parquet", index=False)

    folder = dict(zip(documents["doc_key"], documents["folder"], strict=True))
    in_full = edges["source_doc_key"].map(folder).isin(full_folders)
    in_tax = edges["source_doc_key"].isin(tax_documents)
    summary = {
        "all_documents": {
            **summarise(edges, "all documents"),
            "agreement_with_lovdata_links": compare_with_links(edges, links, sections),
        },
        "full_corpus": {
            **summarise(edges[in_full], "full corpus"),
            "agreement_with_lovdata_links": compare_with_links(
                edges[in_full],
                links[links["source_doc_key"].map(folder).isin(full_folders)],
                sections,
            ),
        },
        "tax_corpus": {
            **summarise(edges[in_tax], "tax corpus"),
            "agreement_with_lovdata_links": compare_with_links(
                edges[in_tax], links[links["source_doc_key"].isin(tax_documents)], sections
            ),
        },
    }
    (processed / "references_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary
