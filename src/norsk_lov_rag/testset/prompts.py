"""Prompts for building the test set. The prompts are in Norwegian, like the questions.

Each prompt has a version. The version is part of the cache key, so a changed prompt gives new
calls, and an unchanged prompt reads its answers from the cache.
"""

from __future__ import annotations

WRITER_VERSION = "writer-v2"
# Cross-reference tasks have their own instructions, so their batches carry their own version.
CROSS_WRITER_VERSION = "writer-v2-cross-v3"
UNANSWERABLE_VERSION = "unanswerable-v2"
CHECKER_VERSION = "checker-v1"
SEARCH_CHECK_VERSION = "search-check-v1"
REVIEWER_VERSION = "reviewer-v2"

STANDARD_NAMES = {"nb": "bokmål", "nn": "nynorsk"}

WRITER_SYSTEM = """\
Du lager testspørsmål til en evaluering av et system som søker i norske lover og forskrifter \
om skatt og avgift. Spørsmålene skal måle om systemet finner riktig bestemmelse og gir riktig \
svar.

Regler for alle spørsmål:
- Spørsmålet skal kunne besvares ut fra tekstene du får, uten annen kunnskap.
- Ikke nevn paragrafnummer, kapittelnummer eller navnet på loven eller forskriften.
- Ikke vis til tekst som leseren ikke ser, som «første ledd», «denne paragrafen» eller «loven».
- Bruk egne ord. Ta aldri med mer enn fire ord på rad fra teksten.
- Skriv ett spørsmål som er kort, tydelig og gir mening alene, uten teksten ved siden av.
- Svaret skal være kort, en til tre setninger, og bygge bare på teksten.
- Skriv på den målformen oppgaven ber om.
- Svar bare med JSON i formatet som er beskrevet."""

TYPE_INSTRUCTIONS = {
    "direct": (
        "Type: direkte. Skriv et presist spørsmål med juridiske fagord, slik en rådgiver "
        "eller saksbehandler ville spurt."
    ),
    "everyday": (
        "Type: hverdag. Skriv spørsmålet slik en vanlig person ville spurt, med hverdagsord og "
        "gjerne om sin egen situasjon. Unngå juridiske fagord der det går. Eksempel på stil: "
        "«Kan jeg trekke fra reiseutgifter til jobben?»"
    ),
    "cross_reference": (
        "Type: krysshenvisning. Tekst A viser til tekst B. Lag et spørsmål som verken tekst A "
        "eller tekst B kan besvare alene. Et godt mønster: tekst A gir en regel eller en følge, "
        "og bare tekst B forklarer hvem, hvilke tilfeller eller hvilke vilkår regelen gjelder "
        "for. Spør om et konkret tilfelle der man må lese begge tekstene for å svare. "
        "Spørsmålet skal ikke nevne at det finnes en henvisning, og verken spørsmålet eller "
        'svaret skal nevne «tekst A» eller «tekst B». I "deler" bruker du numrene i '
        'hakeparentes, for eksempel [1, 2]. Legg til feltet "hvorfor_begge" med én setning '
        "om hva hver av tekstene bidrar med."
    ),
}

WRITER_FORMAT = """\
Svar med JSON i dette formatet, med ett element per oppgave:
{"oppgaver": [{"id": "<oppgave-id>", "sporsmal": "...", "svar": "...", "deler": [<nummer>]}]}
I "deler" skriver du numrene på tekstdelene svaret bygger på. Måldelen skal alltid være med."""

UNANSWERABLE_KINDS = {
    "annual_rate": (
        "satser, beløpsgrenser og fradragsbeløp som Stortinget fastsetter hvert år, og som "
        "derfor ikke står i lovene og forskriftene. Spør etter et konkret tall eller en konkret "
        "sats for i år."
    ),
    "administrative": (
        "praktiske forhold hos Skatteetaten som ikke står i lovtekst, for eksempel "
        "åpningstider, telefonnummer, nettsider eller hvor lang tid saksbehandlingen tar."
    ),
    "foreign": "skatteregler i andre land enn Norge.",
    "outside_tax_law": (
        "rettsområder utenfor skatt og avgift. Spørsmålet skal ikke handle om skatt, avgift "
        "eller trygd."
    ),
}

UNANSWERABLE_SYSTEM = """\
Du lager testspørsmål til en evaluering av et system som søker i norske lover og forskrifter \
om skatt og avgift. Disse spørsmålene skal systemet ikke kunne svare på ut fra lovteksten. Det \
riktige svaret er at teksten ikke svarer på spørsmålet.

Regler:
- Ett spørsmål per tema i listen, i samme rekkefølge.
- Skriv kort og naturlig, slik en vanlig person ville spurt.
- Ikke nevn et bestemt årstall. Skriv heller «i år» der det trengs.
- Ikke nevn paragrafnummer eller navnet på en lov.
- Skriv på den målformen oppgaven ber om.
- Svar bare med JSON i formatet som er beskrevet."""

UNANSWERABLE_FORMAT = """\
Svar med JSON i dette formatet:
{"sporsmal": [{"tema": "<tema fra listen>", "sporsmal": "..."}]}"""

CHECKER_SYSTEM = """\
Du kontrollerer testspørsmål til en evaluering av søk i norske lover. For hvert spørsmål får \
du kildetekstene og et referansesvar. Vurder hvert spørsmål for seg.

- "kan_besvares": true hvis spørsmålet kan besvares bare ut fra kildetekstene.
- "svar_stemmer": true hvis referansesvaret stemmer med kildetekstene.
- "trenger_begge": bare for spørsmål med tekst A og tekst B. true hvis et fullt svar krever \
begge tekstene, false hvis én av dem er nok.
- "begrunnelse": én kort setning.

Svar bare med JSON i dette formatet:
{"vurderinger": [{"id": "...", "kan_besvares": true, "svar_stemmer": true, \
"trenger_begge": null, "begrunnelse": "..."}]}"""

SEARCH_CHECK_SYSTEM = """\
Du kontrollerer spørsmål som lovteksten ikke skal kunne svare på. For hvert spørsmål får du \
de tekstene fra lover og forskrifter som et søk fant. Avgjør om noen av tekstene gir svaret \
på spørsmålet, helt eller med det konkrete tallet eller regelen det spørres etter.

Svar bare med JSON i dette formatet:
{"vurderinger": [{"id": "...", "besvart": false, "begrunnelse": "..."}]}"""

REVIEWER_SYSTEM = """\
Du går gjennom et testsett for søk og spørsmålssvar i norske lover og forskrifter om skatt og \
avgift. For hvert element får du spørsmålet, typen, målformen, referansesvaret og de \
kildetekstene svaret skal bygge på. Vurder hvert element for seg.

Typer:
- direkte: presist spørsmål med juridiske fagord, besvares av kildeteksten.
- hverdag: samme slags spørsmål med hverdagsord, slik en vanlig person spør.
- krysshenvisning: et fullt svar krever både tekst A og tekst B.
- ubesvarbar: tekstene fra søket svarer ikke på spørsmålet, og lover og forskrifter om skatt \
og avgift gjør det heller ikke. Riktig svar er at teksten ikke svarer.

Vurdering:
- "korrekt": spørsmålet er tydelig, gir mening alene, passer til typen, er skrevet på riktig \
målform, og referansesvaret er riktig og dekker det viktigste i kildetekstene. For ubesvarbare \
spørsmål: ingen av tekstene gir svaret.
- "feil": svaret er galt, spørsmålet kan ikke besvares slik typen krever, en tekst svarer på \
et spørsmål som skal være ubesvarbart, eller målformen er feil.
- "uklar": spørsmålet kan forstås på flere måter, viser til tekst leseren ikke ser (som \
«første ledd» eller «loven»), eller svaret avhenger av forhold som ikke er oppgitt.

Svar bare med JSON i dette formatet:
{"vurderinger": [{"id": "...", "vurdering": "korrekt", "begrunnelse": "én kort setning"}]}"""
