"""Data quality checks on the parsed tables. Each check gives a name, a result and a number."""

from __future__ import annotations

import pandas as pd


def _check(name: str, passed: bool, observed, expected: str) -> dict:
    return {"check": name, "passed": bool(passed), "observed": observed, "expected": expected}


def run_checks(tables: dict[str, pd.DataFrame]) -> list[dict]:
    documents, sections, provisions = tables["documents"], tables["sections"], tables["provisions"]
    checks = []

    duplicates = int(documents["doc_key"].duplicated().sum())
    checks.append(_check("document_keys_unique", duplicates == 0, duplicates, "0 duplicates"))

    missing = int((documents["doc_key"] == "").sum() + (documents["legacy_id"] == "").sum())
    checks.append(_check("documents_have_identifiers", missing == 0, missing, "0 missing"))

    duplicates = int(sections["section_id"].duplicated().sum())
    checks.append(_check("section_ids_unique", duplicates == 0, duplicates, "0 duplicates"))

    duplicates = int(provisions["provision_id"].duplicated().sum())
    checks.append(_check("provision_ids_unique", duplicates == 0, duplicates, "0 duplicates"))

    empty = int((provisions["text"].str.strip() == "").sum())
    checks.append(_check("no_empty_provisions", empty == 0, empty, "0 empty"))

    known = set(documents["doc_key"])
    orphans = int((~provisions["doc_key"].isin(known)).sum())
    checks.append(_check("provisions_belong_to_documents", orphans == 0, orphans, "0 orphans"))

    in_sections = provisions["section_id"].notna()
    known_sections = set(sections["section_id"])
    orphans = int((~provisions.loc[in_sections, "section_id"].isin(known_sections)).sum())
    checks.append(_check("provisions_belong_to_sections", orphans == 0, orphans, "0 orphans"))

    standards = set(documents["written_standard"].unique())
    checks.append(
        _check("written_standard_set", standards <= {"nb", "nn"}, sorted(standards), "nb or nn")
    )

    live = sections[~sections["is_repealed"]]
    empty_live = int(live["is_empty"].sum())
    checks.append(
        _check(
            "sections_in_force_have_text",
            empty_live <= 0.01 * max(1, len(live)),
            empty_live,
            "at most 1 percent of sections in force are empty",
        )
    )
    return checks
