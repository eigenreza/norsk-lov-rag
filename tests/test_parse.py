"""The structure parser, on a small made-up document in Lovdata's markup."""

from __future__ import annotations

from norsk_lov_rag.corpus.language import detect_written_standard, written_standard
from norsk_lov_rag.corpus.parse import clean_text, parse_document

EN_DASH = chr(0x2013)
NBSP = chr(0xA0)

DOCUMENT = f"""<!DOCTYPE html><html lang="nb"><head><title>Lov om prøver</title></head><body>
<header class="documentHeader"><dl class="data-document-key-info">
<dt class="legacyID">Datokode</dt><dd class="legacyID">LOV-2001-02-03-4</dd>
<dt class="dokid">DokumentID</dt><dd class="dokid">NL/lov/2001-02-03-4</dd>
<dt class="titleShort">Korttittel</dt><dd class="titleShort">Prøveloven {EN_DASH} prl</dd>
<dt class="title">Tittel</dt><dd class="title">Lov om prøver (prøveloven)</dd>
<dt class="basedOn">Hjemmel</dt><dd class="basedOn"><ul>
<li><a href="lov/1990-01-01-1/§9">x</a></li></ul></dd>
</dl></header>
<main class="documentBody" data-lovdata-URL="NL/lov/2001-02-03-4" id="dokument">
<h1>Lov om prøver</h1>
<section class="section" data-name="kap1" id="kapittel-1"><h2>Kapittel 1. Innledning</h2>
<article class="legalArticle" data-lovdata-URL="NL/lov/2001-02-03-4/§1-1" data-name="§1-1" id="p1">
<h3 class="legalArticleHeader"><span class="legalArticleValue">§ 1-1</span>.
<span class="legalArticleTitle">Virkeområde</span></h3>
<article class="numberedLegalP" data-numerator="1" id="p1-1">(1) Loven gjelder for
<ol class="defaultList" type="a">
<li data-name="a." value="1"><article class="listArticle"><article class="legalP">prøver i
riket, jf. <a href="lov/2001-02-03-4/§1-2">§ 1-2</a></article></article></li>
<li data-name="b." value="2"><article class="listArticle"><article class="legalP">beløp over
12{NBSP}000 kroner<sup class="footnotereference">1</sup></article></article></li>
</ol><p class="leddfortsettelse">Dette gjelder ikke på Svalbard.</p></article>
<article class="numberedLegalP" data-numerator="2" id="p1-2">(2) Satsene er:
<table><tbody><tr><td>Klasse</td><td>Sats</td></tr><tr><td>A</td><td>10</td></tr></tbody></table>
</article>
<article class="changesToParent">Endret ved lov 1 jan 2020 nr. 1.</article>
</article>
<article class="legalArticle" data-lovdata-URL="NL/lov/2001-02-03-4/§1-2" data-name="§1-2"
data-repealeddate="2020-01-01" id="p2"><h3 class="legalArticleHeader">
<span class="legalArticleValue">§ 1-2</span>. <span class="legalArticleTitle">(Opphevet)</span></h3>
<article class="changesToParent">Opphevet ved lov 1 jan 2020 nr. 1.</article></article>
<article class="futureLegalArticle" data-name="§1-3" id="f1"><span class="futureLegalArticleHeader">
<span class="legalArticleValue">§ 1-3</span></span><article class="legalP">Kommer senere.</article>
</article>
</section>
<article class="legalP" id="ledd-9">En løs tekst uten paragraf.</article>
<footer class="footnotes"><p>1 En fotnote.</p></footer>
</main></body></html>"""


def parsed():
    return parse_document(DOCUMENT, "nl/nl-20010203-004.xml")


def test_header_fields_and_keys() -> None:
    document = parsed().document
    assert document["doc_key"] == "lov/2001-02-03-4"
    assert document["legacy_id"] == "LOV-2001-02-03-4"
    assert document["kind"] == "lov" and document["folder"] == "nl"
    assert document["short_name"] == "Prøveloven" and document["abbreviation"] == "prl"
    assert list(document["based_on"]) == ["lov/1990-01-01-1"]
    assert document["written_standard"] == "nb"
    assert document["n_sections"] == 2 and document["n_future_sections_skipped"] == 1


def test_sections_get_lovdata_identifiers_and_chapter() -> None:
    sections = {s["section_id"]: s for s in parsed().sections}
    first = sections["lov/2001-02-03-4/§1-1"]
    assert first["section_number"] == "1-1" and first["section_label"] == "§ 1-1"
    assert first["section_title"] == "Virkeområde"
    assert first["chapter_path"] == "Kapittel 1. Innledning" and first["chapter_key"] == "kap1"
    assert first["n_paragraphs"] == 2 and not first["is_repealed"] and not first["is_empty"]
    repealed = sections["lov/2001-02-03-4/§1-2"]
    assert repealed["is_repealed"] and repealed["is_empty"] and repealed["n_paragraphs"] == 0


def test_paragraph_text_keeps_letters_and_drops_notes_and_footnotes() -> None:
    provisions = {p["provision_id"]: p for p in parsed().provisions}
    first = provisions["lov/2001-02-03-4/§1-1/ledd/1"]
    assert first["paragraph"] == 1 and first["letters"] == ["a", "b"]
    assert first["text"].split("\n") == [
        "(1) Loven gjelder for",
        "a. prøver i riket, jf. § 1-2",
        "b. beløp over 12 000 kroner",
        "Dette gjelder ikke på Svalbard.",
    ]
    second = provisions["lov/2001-02-03-4/§1-1/ledd/2"]
    assert second["text"].split("\n") == ["(2) Satsene er:", "Klasse | Sats", "A | 10"]
    everything = "\n".join(p["text"] for p in parsed().provisions)
    assert "Endret ved" not in everything and "fotnote" not in everything
    assert "Kommer senere" not in everything


def test_loose_text_outside_sections_is_kept_with_its_own_identifier() -> None:
    loose = [p for p in parsed().provisions if p["section_id"] is None]
    assert len(loose) == 1
    assert loose[0]["provision_id"] == "lov/2001-02-03-4/#ledd-9"
    assert loose[0]["text"] == "En løs tekst uten paragraf."


def test_links_keep_source_and_target() -> None:
    links = parsed().links
    assert links == [
        {
            "source_provision_id": "lov/2001-02-03-4/§1-1/ledd/1",
            "source_doc_key": "lov/2001-02-03-4",
            "href": "lov/2001-02-03-4/§1-2",
            "text": "§ 1-2",
        }
    ]


def test_second_language_version_gets_its_own_key() -> None:
    nynorsk = parse_document(
        DOCUMENT.replace('lang="nb"', 'lang="nn"'), "nl/nl-20010203-004-nn.xml"
    )
    assert nynorsk.document["doc_key"] == "lov/2001-02-03-4-nn"
    assert nynorsk.document["base_doc_key"] == "lov/2001-02-03-4"
    assert nynorsk.sections[0]["section_id"] == "lov/2001-02-03-4-nn/§1-1"
    assert nynorsk.provisions[0]["provision_id"] == "lov/2001-02-03-4-nn/§1-1/ledd/1"
    assert nynorsk.document["written_standard"] == "nn"


def test_clean_text_collapses_spaces_and_drops_empty_lines() -> None:
    assert clean_text(f"  a{NBSP}{NBSP}b \n\n  c\t d ") == "a b\nc d"


def test_language_detector_tells_bokmal_from_nynorsk() -> None:
    bokmal = (
        "Loven gjelder ikke for virksomhet som bare har en ansatt, og heller ikke fra utlandet."
    )
    nynorsk = (
        "Lova gjeld ikkje for verksemd som berre har ein tilsett, og heller ikkje frå utlandet."
    )
    assert detect_written_standard(bokmal) == "nb"
    assert detect_written_standard(nynorsk) == "nn"
    assert detect_written_standard("Kort tekst.") is None
    assert written_standard("nn", bokmal) == "nn"
    assert written_standard("no", nynorsk) == "nn"
    assert written_standard("no", "Kort tekst.") == "nb"
