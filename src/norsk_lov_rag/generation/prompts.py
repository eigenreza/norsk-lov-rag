"""Prompts for answers and for the judge. The prompts are in Norwegian, like the questions.

A change to a prompt must change its version, so that cached answers from an older prompt are
never mixed with new ones.
"""

from __future__ import annotations

from norsk_lov_rag.generation.sources import Source

ANSWER_PROMPT_VERSION = "answer-v4"
JUDGE_PROMPT_VERSION = "judge-v1"

ABSTAIN = {
    "nb": "Jeg finner ikke svaret i lovtekstene jeg har fått.",
    "nn": "Eg finn ikkje svaret i lovtekstane eg har fått.",
}

# The rules go in the user message, not in a system message. The local model was tuned on
# conversations without system messages, and one format for both generators keeps them equal.
# The models write plain text; the structured answer is built from it (see answers.py), because
# the small local model breaks down when it is forced to write JSON.
ANSWER_RULES = f"""\
Du svarer på spørsmål om norsk skatterett. Under får du noen utdrag fra lover og forskrifter \
og et spørsmål. Følg disse reglene:

1. Svar bare ut fra utdragene. Ikke bruk annen kunnskap.
2. Skriv på samme skriftform som spørsmålet. Er spørsmålet på nynorsk, skal svaret være på \
nynorsk. Er det på bokmål, skal svaret være på bokmål.
3. Hver setning som sier noe om reglene, skal slutte med en kilde i hakeparentes før \
punktumet. Skriv kilden nøyaktig slik den står i hakeparentesen foran utdraget, for eksempel \
[skatteloven § 6-44]. Bruk bare kilder som står foran et utdrag.
4. Hvis utdragene ikke gir svar på spørsmålet, skal du ikke gjette. Skriv da bare denne \
setningen på bokmål: «{ABSTAIN["nb"]}» eller denne på nynorsk: «{ABSTAIN["nn"]}».
5. Svar kort og klart, med høyst fem setninger.
6. Skriv bare selve svaret som vanlig tekst, uten overskrift, liste eller JSON."""

ANSWER_REMINDER = """\
Husk: Svar bare ut fra utdragene, sett en kilde i hakeparentes etter hver setning, og skriv \
den faste setningen hvis utdragene ikke svarer på spørsmålet."""


def answer_messages(question: str, sources: list[Source]) -> list[dict]:
    """One user message: rules, sources, question and a short reminder of the rules."""
    blocks = []
    for source in sources:
        head = f"[{source.label}]" + (f" {source.title}" if source.title else "")
        blocks.append(f"{head}\n{source.text}")
    context = "\n\n".join(blocks) if blocks else "(ingen utdrag)"
    user = f"{ANSWER_RULES}\n\nUtdrag:\n\n{context}\n\nSpørsmål: {question}\n\n{ANSWER_REMINDER}"
    return [{"role": "user", "content": user}]


JUDGE_SYSTEM = """\
Du vurderer svar på spørsmål om norsk skatterett. Svarene kommer fra forskjellige systemer, \
men du får ikke vite hvilke. Vurder hvert punkt for seg.

For hvert punkt får du et spørsmål, et referansesvar når det finnes, svaret som skal \
vurderes, og for hver kilde svaret viser til: påstandene som viser til kilden og teksten i \
bestemmelsen.

Vurder to ting:

1. correctness: Stemmer svaret med referansesvaret og lovteksten?
   - "riktig": Hovedpoenget i referansesvaret er med, og ingenting viktig er feil.
   - "delvis": Noe av hovedpoenget er med, men noe viktig mangler eller er uklart.
   - "feil": Svaret er feil, sier noe annet enn referansesvaret eller svarer ikke på \
spørsmålet.
   Et svar kan være riktig selv om det bruker andre ord eller er kortere enn referansesvaret. \
Mangler referansesvaret, skal correctness være null.

2. support: For hver kilde, finnes regelen i påstandene i teksten til bestemmelsen? Svar true \
hvis teksten sier det påstanden sier, og false hvis teksten ikke sier det eller sier noe \
annet.

Svar med JSON i denne formen:
{"items": [{"id": "...", "correctness": "riktig" | "delvis" | "feil" | null, "support": \
[{"citation": "...", "supported": true | false}], "reason": "én kort setning"}]}
Ta med alle punktene, i samme rekkefølge, og alle kildene i hvert punkt."""


def judge_messages(items: list[dict]) -> list[dict]:
    """Items carry id, question, reference (or None), answer and claims per citation."""
    blocks = []
    for item in items:
        lines = [f"### Punkt {item['id']}", f"Spørsmål: {item['question']}"]
        if item.get("reference"):
            lines.append(f"Referansesvar: {item['reference']}")
        lines.append(f"Svar som skal vurderes: {item['answer']}")
        for cite in item["citations"]:
            lines.append(f"Kilde [{cite['citation']}]")
            lines.append("Påstander: " + " ".join(cite["claims"]))
            lines.append(f"Tekst i bestemmelsen: {cite['text']}")
        blocks.append("\n".join(lines))
    user = "\n\n".join(blocks)
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
