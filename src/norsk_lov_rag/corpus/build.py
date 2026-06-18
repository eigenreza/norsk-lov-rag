"""Build the document, section, provision and link tables from the downloaded archives."""

from __future__ import annotations

import hashlib
import json
import logging
import tarfile
from pathlib import Path

import pandas as pd

from norsk_lov_rag.config import Settings
from norsk_lov_rag.corpus.download import download_datasets, sha256_of
from norsk_lov_rag.corpus.parse import parse_document
from norsk_lov_rag.corpus.quality import run_checks

log = logging.getLogger(__name__)

TABLES = ("documents", "sections", "provisions", "links")


def corpus_hash(archives: dict[str, dict]) -> str:
    """One hash for the corpus version, from the hashes of the archives it was built from."""
    lines = "\n".join(f"{name}:{info['sha256']}" for name, info in sorted(archives.items()))
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def parse_archives(raw_dir: Path, names: list[str]) -> dict[str, pd.DataFrame]:
    """Parse every XML document in the archives into the four tables."""
    rows: dict[str, list[dict]] = {name: [] for name in TABLES}
    for name in names:
        with tarfile.open(raw_dir / name, "r:bz2") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".xml"):
                    continue
                raw = archive.extractfile(member).read().decode("utf-8")
                parsed = parse_document(raw, source_file=member.name)
                rows["documents"].append(parsed.document)
                rows["sections"].extend(parsed.sections)
                rows["provisions"].extend(parsed.provisions)
                rows["links"].extend(parsed.links)
        log.info("parsed %s: %d documents so far", name, len(rows["documents"]))
    return {name: pd.DataFrame(rows[name]) for name in TABLES}


def summarise(tables: dict[str, pd.DataFrame]) -> dict:
    """Counts for the manifest and the checkpoint report. No dates."""
    documents, sections, provisions = tables["documents"], tables["sections"], tables["provisions"]
    per_folder = documents.groupby("folder").agg(
        documents=("doc_key", "count"),
        sections=("n_sections", "sum"),
        provisions=("n_provisions", "sum"),
        repealed_sections=("n_repealed_sections", "sum"),
        future_sections_skipped=("n_future_sections_skipped", "sum"),
    )
    standard = documents.groupby(["folder", "written_standard"]).size().unstack(fill_value=0)
    return {
        "documents": int(len(documents)),
        "sections": int(len(sections)),
        "provisions": int(len(provisions)),
        "links": int(len(tables["links"])),
        "per_folder": json.loads(per_folder.to_json(orient="index")),
        "written_standard_documents": json.loads(standard.to_json(orient="index")),
        "empty_sections": int(sections["is_empty"].sum()),
        "repealed_sections": int(sections["is_repealed"].sum()),
        "documents_without_sections": int((documents["n_sections"] == 0).sum()),
    }


def build_corpus(settings: Settings, download: bool = True, force: bool = False) -> dict:
    """Download when asked, parse, check and write the tables. Returns the manifest."""
    raw_dir, out_dir = settings.paths.raw, settings.paths.processed
    names = settings.lovdata.datasets
    if download:
        archives = download_datasets(settings.lovdata, raw_dir, force=force)
    else:
        archives = {
            name: {"bytes": (raw_dir / name).stat().st_size, "sha256": sha256_of(raw_dir / name)}
            for name in names
        }
    tables = parse_archives(raw_dir, names)
    checks = run_checks(tables)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    manifest = {
        "source": "Lovdata, open datasets under NLOD 2.0",
        "archives": archives,
        "corpus_hash": corpus_hash(archives),
        "counts": summarise(tables),
        "checks": checks,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest
