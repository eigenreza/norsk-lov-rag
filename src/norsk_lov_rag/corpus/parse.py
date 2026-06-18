"""Parse one Lovdata document into its header, sections, provisions and links.

The files are XML-compatible HTML. A section (``§``) is an ``article`` with class
``legalArticle``. Its paragraphs (``ledd``) are child articles with class ``legalP`` or
``numberedLegalP``. Letters and numbered points are list items inside a paragraph. Amendment
notes, footnotes and sections that are not yet in force are left out of the text.

Identifiers follow Lovdata's own link format, so that a link such as
``lov/1999-03-26-14/§6-44/ledd/1`` points straight at a row in the provision table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lxml import html as lhtml

from norsk_lov_rag.corpus.language import written_standard

PARAGRAPH_CLASSES = ("legalP", "numberedLegalP")
SKIPPED_CLASSES = ("changesToParent", "footnotes", "footnotereference", "futureLegalArticle")
HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
FOLDER_KINDS = {
    "nl": "lov",
    "sf": "forskrift",
    "stv": "stortingsvedtak",
    "del": "delegering",
    "ins": "instruks",
}
EN_DASH = chr(0x2013)
NEWLINE = chr(10)
SPACES = re.compile("[ \t\r\f\v" + chr(0xA0) + "]+")
REPEALED = re.compile(r"^\(?\s*opphev(et|a)\b", re.IGNORECASE)
LANGUAGE_SAMPLE_CHARS = 20_000
LANGUAGE_VARIANTS = ("nn", "nb", "en", "se")


@dataclass
class ParsedDocument:
    document: dict
    sections: list[dict] = field(default_factory=list)
    provisions: list[dict] = field(default_factory=list)
    links: list[dict] = field(default_factory=list)


@dataclass
class _Walk:
    doc_key: str
    base_key: str
    folder: str
    parsed: ParsedDocument
    path: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    seen: set[str] = field(default_factory=set)
    future_skipped: int = 0
    order: int = 0


def clean_text(text: str) -> str:
    """Collapse runs of spaces and drop empty lines. Non-breaking spaces become plain spaces."""
    lines = (SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _class(element) -> str:
    return element.get("class") or ""


def _render(element, out: list[str]) -> None:
    """Append the readable text of an element. List items and table rows get their own lines."""
    if not isinstance(element.tag, str) or _class(element) in SKIPPED_CLASSES:
        return
    tag = element.tag
    if tag == "br":
        out.append("\n")
        return
    if tag == "table":
        for row in element.iter("tr"):
            cells = []
            for cell in row:
                if isinstance(cell.tag, str) and cell.tag in ("td", "th"):
                    parts: list[str] = []
                    _render(cell, parts)
                    cells.append(clean_text("".join(parts)).replace("\n", " "))
            out.append("\n" + " | ".join(cells))
        out.append("\n")
        return
    if tag == "li":
        label = (element.get("data-name") or "").strip()
        out.append("\n" + (label + " " if label else ""))
    elif tag in ("p", "div", "tr") or tag in HEADING_TAGS:
        out.append("\n")
    # Line breaks inside the source text are plain white space. Only the structure adds lines.
    if element.text:
        out.append(element.text.replace(NEWLINE, " "))
    for child in element:
        _render(child, out)
        if child.tail:
            out.append(child.tail.replace(NEWLINE, " "))
    if tag in ("li", "p", "div") or tag in HEADING_TAGS:
        out.append("\n")


def element_text(element) -> str:
    out: list[str] = []
    _render(element, out)
    return clean_text("".join(out))


def _plain(element) -> str:
    return clean_text(" ".join(element.itertext())) if element is not None else ""


def _header(root) -> dict:
    """The key information block at the top of a document, as class name to element."""
    info = {}
    for term in root.xpath("//header//dl/dt"):
        value = term.getnext()
        if value is not None and value.tag == "dd":
            info[term.get("class")] = value
    return info


def _doc_key(dokid: str) -> str:
    """``NL/lov/1999-03-26-14`` becomes ``lov/1999-03-26-14``, the form Lovdata's links use."""
    return dokid.split("/", 1)[1] if "/" in dokid else dokid


def _link_doc_key(href: str) -> str | None:
    """The document part of a link such as ``lov/1999-03-26-14/§6-13``."""
    parts = href.strip().split("/")
    if len(parts) >= 2 and parts[0] in ("lov", "forskrift"):
        return "/".join(parts[:2])
    return None


def _letters(element) -> list[str]:
    """Labels of the first level of list items in a paragraph, such as a, b and c."""
    labels = []
    for item in element.xpath("./ol/li[@data-name] | ./ul/li[@data-name]"):
        labels.append(item.get("data-name").strip().rstrip(".)").strip())
    return labels


def _add_links(element, provision_id: str, walk: _Walk) -> None:
    for link in element.iter("a"):
        href = (link.get("href") or "").strip()
        if not href or any(_class(parent) in SKIPPED_CLASSES for parent in link.iterancestors()):
            continue
        walk.parsed.links.append(
            {
                "source_provision_id": provision_id,
                "source_doc_key": walk.doc_key,
                "href": href,
                "text": _plain(link),
            }
        )


def _unique(identifier: str, walk: _Walk) -> str:
    """Keep identifiers unique inside a document by numbering later duplicates."""
    candidate, count = identifier, 1
    while candidate in walk.seen:
        count += 1
        candidate = f"{identifier}~{count}"
    walk.seen.add(candidate)
    return candidate


def _provision(walk: _Walk, element, provision_id: str, section: dict | None, **fields) -> None:
    text = element_text(element)
    if not text:
        return
    walk.order += 1
    row = {
        "provision_id": provision_id,
        "section_id": section["section_id"] if section else None,
        "doc_key": walk.doc_key,
        "folder": walk.folder,
        "section_number": section["section_number"] if section else None,
        "section_title": section["section_title"] if section else None,
        "chapter_path": " > ".join(p for p in walk.path if p),
        "paragraph": None,
        "kind": _class(element) or element.tag,
        "letters": _letters(element),
        "text": text,
        "order": walk.order,
    }
    row.update(fields)
    walk.parsed.provisions.append(row)
    _add_links(element, provision_id, walk)


def _section(element, walk: _Walk, label: str, title: str) -> None:
    """One section with its paragraphs. Works for ``§`` articles and for margin-numbered units."""
    url = element.get("data-lovdata-url") or ""
    name = (element.get("data-name") or "").strip()
    if url:
        section_id = _doc_key(url)
        if walk.doc_key != walk.base_key and section_id.startswith(walk.base_key):
            section_id = walk.doc_key + section_id[len(walk.base_key) :]
    else:
        section_id = f"{walk.doc_key}/#{element.get('id') or name or label}"
    section_id = _unique(section_id, walk)
    number = name.lstrip("§").strip() or label.lstrip("§").replace(" ", "").strip()
    section = {
        "section_id": section_id,
        "doc_key": walk.doc_key,
        "folder": walk.folder,
        "section_number": number,
        "section_label": label,
        "section_title": title,
        "chapter_path": " > ".join(p for p in walk.path if p),
        "chapter_key": next((k for k in reversed(walk.keys) if k), ""),
        "element_id": element.get("id") or "",
    }
    first = len(walk.parsed.provisions)
    notes, paragraph, other = [], 0, 0
    for child in element:
        if not isinstance(child.tag, str):
            continue
        child_class = _class(child)
        if "legalArticleHeader" in child_class or child_class == "data-marginOriginalId":
            continue
        if child_class == "changesToParent":
            notes.append(_plain(child))
            continue
        if child.tag == "article" and child_class in PARAGRAPH_CLASSES:
            paragraph += 1
            _provision(walk, child, f"{section_id}/ledd/{paragraph}", section, paragraph=paragraph)
        else:
            other += 1
            _provision(walk, child, f"{section_id}/annet/{other}", section)
    rows = walk.parsed.provisions[first:]
    text = "\n".join(row["text"] for row in rows)
    section["n_paragraphs"] = sum(1 for row in rows if row["paragraph"] is not None)
    section["text"] = text
    section["is_empty"] = not text
    section["is_repealed"] = bool(
        element.get("data-repealeddate")
        or REPEALED.match(title)
        or (not text and any(REPEALED.match(note) for note in notes))
    )
    walk.parsed.sections.append(section)


def _walk_container(container, walk: _Walk) -> None:
    for element in container:
        if not isinstance(element.tag, str):
            continue
        element_class = _class(element)
        if element.tag == "section":
            heading = next((_plain(c) for c in element if c.tag in HEADING_TAGS), "")
            walk.path.append(heading)
            walk.keys.append(element.get("data-name") or "")
            _walk_container(element, walk)
            walk.path.pop()
            walk.keys.pop()
        elif element.tag == "article" and element_class == "legalArticle":
            headers = element.xpath("./*[contains(@class, 'legalArticleHeader')]")
            header = headers[0] if headers else None
            label = title = ""
            if header is not None:
                value = header.xpath(".//*[contains(@class, 'legalArticleValue')]")
                name = header.xpath(".//*[contains(@class, 'legalArticleTitle')]")
                label = _plain(value[0]) if value else ""
                title = _plain(name[0]) if name else ""
            _section(element, walk, label, title)
        elif element.tag == "article" and element_class == "futureLegalArticle":
            walk.future_skipped += 1
        elif element.tag == "article" and element_class == "marginIdArticle":
            marks = element.xpath("./*[@class='data-marginOriginalId']")
            _section(element, walk, _plain(marks[0]) if marks else "", "")
        elif (
            element.tag in HEADING_TAGS
            or element.tag == "footer"
            or element_class in SKIPPED_CLASSES
        ):
            continue
        else:
            identifier = _unique(f"{walk.doc_key}/#{element.get('id') or 'blokk'}", walk)
            _provision(walk, element, identifier, None)


def parse_document(raw: str, source_file: str) -> ParsedDocument:
    """Parse one document. ``source_file`` is its path in the archive, such as ``nl/nl-...xml``."""
    root = lhtml.fromstring(raw)
    info = _header(root)
    folder = source_file.split("/")[0]
    dokid = _plain(info.get("dokid"))
    base_key = _doc_key(dokid)
    # A few documents come in two language versions with the same identifier. The file name
    # of the second version ends with its language code, which then becomes part of the key.
    stem = source_file.rsplit("/", 1)[-1].removesuffix(".xml")
    variant = stem.rsplit("-", 1)[-1] if stem.rsplit("-", 1)[-1] in LANGUAGE_VARIANTS else ""
    doc_key = f"{base_key}-{variant}" if variant else base_key
    short = _plain(info.get("titleShort"))
    name, _, abbreviation = short.partition(f" {EN_DASH} ")
    based_on = []
    if info.get("basedOn") is not None:
        for link in info["basedOn"].iter("a"):
            key = _link_doc_key(link.get("href") or "")
            if key and key not in based_on:
                based_on.append(key)
    declared = (root.get("lang") or "").strip().lower()[:2]
    document = {
        "doc_key": doc_key,
        "base_doc_key": base_key,
        "doc_id": dokid,
        "legacy_id": _plain(info.get("legacyID")),
        "folder": folder,
        "kind": FOLDER_KINDS.get(folder, "annet"),
        "title": _plain(info.get("title")),
        "short_title": short,
        "short_name": name.strip(),
        "abbreviation": abbreviation.strip(),
        "ministry": _plain(info.get("ministry")),
        "legal_area": _plain(info.get("legalArea")),
        "based_on": based_on,
        "lang_declared": declared,
        "source_file": source_file,
    }
    parsed = ParsedDocument(document=document)
    walk = _Walk(doc_key=doc_key, base_key=base_key, folder=folder, parsed=parsed)
    main = root.xpath("//main")
    if main:
        _walk_container(main[0], walk)
    sample = "\n".join(row["text"] for row in parsed.provisions)[:LANGUAGE_SAMPLE_CHARS]
    document["written_standard"] = written_standard(declared, sample)
    document["n_sections"] = len(parsed.sections)
    document["n_provisions"] = len(parsed.provisions)
    document["n_repealed_sections"] = sum(1 for s in parsed.sections if s["is_repealed"])
    document["n_future_sections_skipped"] = walk.future_skipped
    for row in parsed.provisions:
        row["written_standard"] = document["written_standard"]
    for row in parsed.sections:
        row["written_standard"] = document["written_standard"]
    return parsed
