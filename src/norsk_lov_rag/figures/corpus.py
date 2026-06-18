"""Figures for the corpus and the reference graph.

The counts come from the provision tables under ``data/processed``. They are first written to
``results/corpus/``, so that the figures can be built again from the repository alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from norsk_lov_rag.figures.charts import bars, dot_rows, legend_below
from norsk_lov_rag.figures.style import CORPORA, PALETTE, figure, pick, save, t

REASONS = {
    "section_not_found": ("Paragrafen finnes ikke", "Section not found"),
    "unknown_act_name": ("Ukjent lovnavn", "Unknown act name"),
    "ambiguous_act_name": ("Tvetydig lovnavn", "Ambiguous act name"),
    "cited_document_not_in_corpus": ("Dokumentet er ikke i korpuset", "Document not in corpus"),
    "parent_law_unknown": ("Ukjent hjemmelslov", "Parent act unknown"),
    "regulation_not_named": ("Forskriften er ikke navngitt", "Regulation not named"),
}
FOLDERS = {"nl": ("Lover", "Acts"), "sf": ("Sentrale forskrifter", "Central regulations")}


def _short_name(name: str) -> str:
    """'FSFIN Forskrift til skatteloven' becomes 'FSFIN'; other names stay as they are."""
    first = name.split()[0] if name else name
    return first if first.isupper() and len(first) > 1 else name


def corpus_summary(settings) -> dict:
    """Counts per tax document, and the share of nynorsk in the full corpus."""
    from norsk_lov_rag.corpus.select import load_corpus

    processed = settings.paths.processed
    tax = load_corpus(processed, settings.corpus, "tax")
    documents = tax["documents"].set_index("doc_key")
    order = [k for k in settings.corpus.tax_documents if k in documents.index]
    sections = tax["sections"].groupby("doc_key").size()
    provisions = tax["provisions"].groupby("doc_key").size()
    tax_rows = [
        {
            "doc_key": key,
            "name": _short_name(str(documents.loc[key, "short_name"])),
            "sections": int(sections.get(key, 0)),
            "provisions": int(provisions.get(key, 0)),
            "written_standard": str(documents.loc[key, "written_standard"]),
        }
        for key in order
    ]
    full = load_corpus(processed, settings.corpus, "full")
    folders = {}
    for folder, group in full["documents"].groupby("folder"):
        keys = set(group["doc_key"])
        provs = full["provisions"][full["provisions"]["doc_key"].isin(keys)]
        folders[folder] = {
            "documents": len(group),
            "nynorsk_documents": int((group["written_standard"] == "nn").sum()),
            "provisions": len(provs),
            "nynorsk_provisions": int((provs["written_standard"] == "nn").sum()),
        }
    manifest = processed / "manifest.json"
    corpus_hash = None
    if manifest.exists():
        corpus_hash = json.loads(manifest.read_text(encoding="utf-8")).get("corpus_hash")
    return {"corpus_hash": corpus_hash, "tax_documents": tax_rows, "full_folders": folders}


def corpus_overview(summary: dict, out: Path, lang: str) -> list[Path]:
    rows = summary["tax_documents"]
    folders = summary["full_folders"]
    fig, axes = figure(10, 0.36 * len(rows) + 1.6, ncols=2,
                       gridspec_kw={"width_ratios": [1.5, 1]})  # fmt: skip
    bars(axes[0], [r["name"] for r in rows], [r["sections"] for r in rows],
         t("Paragrafer i kraft", "Sections in force", lang), sep=t(" ", ",", lang))  # fmt: skip
    axes[0].set_title(t("Skattekorpuset", "The tax corpus", lang))
    names = [pick(FOLDERS, f, lang) for f in folders]
    series = [
        (t("Dokumenter", "Documents", lang),
         [v["nynorsk_documents"] / v["documents"] if v["documents"] else None
          for v in folders.values()]),
        (t("Tekstledd", "Paragraphs", lang),
         [v["nynorsk_provisions"] / v["provisions"] if v["provisions"] else None
          for v in folders.values()]),
    ]  # fmt: skip
    dot_rows(axes[1], names, series, t("Andel på nynorsk", "Share in Nynorsk", lang),
             full_range=True)  # fmt: skip
    top = max((v for _, vals in series for v in vals if v is not None), default=0.2)
    axes[1].set_xlim(0, min(1.0, top * 1.4 + 0.02))
    axes[1].set_title(t("Hele korpuset: nynorsk", "Full corpus: Nynorsk", lang))
    legend_below(fig, [axes[1]])
    fig.suptitle(t("Korpus", "Corpus", lang), x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return save(fig, out, "korpus_oversikt")


def reference_graph(summary: dict, out: Path, lang: str) -> list[Path]:
    scopes = [s for s in ("tax_corpus", "full_corpus") if s in summary]
    labels = [pick(CORPORA, s.split("_")[0], lang) for s in scopes]
    series = [
        (t("Løste henvisninger", "References resolved", lang),
         [summary[s]["resolved_share"] for s in scopes]),
        (t("Treff mot Lovdatas lenker (recall)", "Match with Lovdata links (recall)", lang),
         [summary[s]["agreement_with_lovdata_links"]["recall_against_links"] for s in scopes]),
        (t("Treff mot Lovdatas lenker (presisjon)", "Match with Lovdata links (precision)", lang),
         [summary[s]["agreement_with_lovdata_links"]["precision_against_links"]
          for s in scopes]),
    ]  # fmt: skip
    reasons = summary["full_corpus"]["unresolved_reasons"] if "full_corpus" in summary else {}
    ordered = sorted(reasons.items(), key=lambda kv: -kv[1])
    fig, axes = figure(10, 3.2, ncols=2, gridspec_kw={"width_ratios": [1, 1.1]})
    dot_rows(axes[0], labels, series, t("Andel", "Share", lang))
    axes[0].set_title(t("Henvisninger til paragrafer", "References to sections", lang))
    bars(axes[1], [pick(REASONS, k, lang) for k, _ in ordered], [v for _, v in ordered],
         t("Antall henvisninger", "References", lang), color=PALETTE[1],
         sep=t(" ", ",", lang))  # fmt: skip
    axes[1].set_title(t("Hele korpuset: uløste henvisninger", "Full corpus: unresolved", lang))
    legend_below(fig, [axes[0]], ncol=1)
    fig.suptitle(t("Henvisningsgrafen", "The reference graph", lang),
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    fig.tight_layout()
    return save(fig, out, "henvisninger")
