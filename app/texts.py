"""Interface text in Norwegian and English. Questions and answers stay in Norwegian."""

REPO_URL = "https://github.com/eigenreza/norsk-lov-rag"
AUTHOR = "Reza Azad Gholami"

TEXT = {
    "title": {"nb": "Norsk lov-RAG", "en": "Norwegian law RAG"},
    "intro": {
        "nb": "Still et spørsmål om norsk skatterett, på norsk eller engelsk. Systemet finner "
        "paragrafene i loven som passer, og svarer ut fra dem med kilde til hver påstand.",
        "en": "Ask a question about Norwegian tax law, in Norwegian or English. The system "
        "finds the sections of the law that fit, and answers from them with a source for "
        "every claim.",
    },
    "author": {
        "nb": f"Laget av {AUTHOR} · [Kildekode på GitHub]({REPO_URL})",
        "en": f"Made by {AUTHOR} · [Source code on GitHub]({REPO_URL})",
    },
    "language": {"nb": "Språk", "en": "Language"},
    "page_ask": {"nb": "Still et spørsmål", "en": "Ask a question"},
    "page_examples": {"nb": "Eksempler", "en": "Examples"},
    "page_compare": {"nb": "Sammenlign søk", "en": "Compare search"},
    "page_results": {"nb": "Resultater", "en": "Results"},
    "page_about": {"nb": "Om prosjektet", "en": "About"},
    "disclaimer": {
        "nb": "Svarene lages av kunstig intelligens og kan være feil. De er ikke juridisk råd "
        "eller skatteråd. Lovteksten ble hentet på et bestemt tidspunkt og kan ha endret seg "
        "siden. Tjenesten er ikke knyttet til Skatteetaten eller andre offentlige etater.",
        "en": "The answers are made by artificial intelligence and may be wrong. They are not "
        "legal or tax advice. The law text was collected at one point in time and may have "
        "changed since. The service is not connected to the Norwegian Tax Administration or "
        "any other public body.",
    },
    "attribution_intro": {
        "nb": "Lovteksten kommer fra Lovdatas åpne datasett.",
        "en": "The law text comes from Lovdata's open datasets.",
    },
    "privacy": {
        "nb": "Ikke skriv navn, fødselsnummer eller andre personlige opplysninger. For å lage "
        "svaret sendes spørsmålet til Groq (modellen gpt-oss-120b). Groq lagrer ikke "
        "innholdet som standard, men kan beholde logger i inntil 30 dager for å hindre misbruk. "
        "Demoen lagrer ikke spørsmålet ditt. Du finner mer om "
        "personvern under «Om prosjektet».",
        "en": "Do not type names, national identity numbers or other personal information. To "
        "write the answer, the question is sent to Groq (the model gpt-oss-120b). Groq does "
        "not store the content by default, but may keep logs for up to 30 days to prevent "
        "misuse. The demo does not store your question. There is "
        "more about privacy under «About».",
    },
    "question": {
        "nb": "Ditt spørsmål, på norsk eller engelsk",
        "en": "Your question, in Norwegian or English",
    },
    "ask_button": {"nb": "Få svar", "en": "Get an answer"},
    "placeholder": {
        "nb": "Kan jeg trekke fra reiseutgifter mellom hjem og arbeid?",
        "en": "Kan jeg trekke fra reiseutgifter mellom hjem og arbeid?",
    },
    "empty": {"nb": "Skriv et spørsmål først.", "en": "Type a question first."},
    "too_long": {
        "nb": "Spørsmålet er for langt. Bruk høyst {n} tegn.",
        "en": "The question is too long. Use at most {n} characters.",
    },
    "left": {
        "nb": "Du kan få {n} svar til i dag.",
        "en": "You can get {n} more answers today.",
    },
    "waiting": {
        "nb": "Finner paragrafene og skriver svaret. Det tar vanligvis noen sekunder.",
        "en": "Finding the sections and writing the answer. This usually takes a few seconds.",
    },
    "terms": {
        "nb": "Søkeord fra modellen, i lovens egne ord: {terms}",
        "en": "Search terms from the model, in the words of the law: {terms}",
    },
    "chat_hello": {
        "nb": "Hei! Spør meg om norsk skatterett, på norsk eller engelsk, så svarer jeg ut fra "
        "lovteksten og viser hvilke paragrafer svaret bygger på. Du kan også spørre om "
        "hvordan demoen virker. Prøv for eksempel et av disse spørsmålene:",
        "en": "Hi! Ask me about Norwegian tax law, in Norwegian or English, and I answer from "
        "the law text and show which sections the answer is based on. You can also ask how the "
        "demo works. Try one of these questions, for example:",
    },
    "chat_placeholder": {
        "nb": "Skriv spørsmålet ditt her, for eksempel: Er barnetrygd skattepliktig?",
        "en": "Type your question here, for example: Is child benefit taxable?",
    },
    "details": {"nb": "Hvordan søket fant svaret", "en": "How the search found the answer"},
    "new_chat": {"nb": "Ny samtale", "en": "New conversation"},
    "answer": {"nb": "Svar", "en": "Answer"},
    "abstained": {
        "nb": "Svaret står ikke i paragrafene som ble funnet. Systemet gjetter ikke, og "
        "svarer derfor ikke. Du kan prøve å stille spørsmålet på en annen måte.",
        "en": "The answer is not in the sections that were found. The system does not guess, "
        "so it does not answer. You can try asking the question in another way.",
    },
    "citations": {"nb": "Paragrafene svaret bygger på", "en": "Sections the answer is based on"},
    "no_citations": {
        "nb": "Svaret viser ikke til noen paragraf.",
        "en": "The answer does not cite any section.",
    },
    "in_sources": {"nb": "funnet av søket", "en": "found by the search"},
    "not_in_sources": {"nb": "ikke funnet av søket", "en": "not found by the search"},
    "unresolved": {
        "nb": "Denne paragrafen finnes ikke i lovtekstene demoen har.",
        "en": "This section is not in the law texts the demo holds.",
    },
    "passages": {"nb": "Paragrafer søket fant", "en": "Sections the search found"},
    "shown": {"nb": "Sendt til modellen", "en": "Sent to the model"},
    "rank": {"nb": "Plass", "en": "Rank"},
    "provision": {"nb": "Paragraf", "en": "Section"},
    "heading": {"nb": "Overskrift", "en": "Heading"},
    "score": {"nb": "Treffpoeng", "en": "Match score"},
    "usage": {
        "nb": "Tid: {s:.1f} s. Modell: {m}. Tokens inn og ut: {p} og {c}.",
        "en": "Time: {s:.1f} s. Model: {m}. Tokens in and out: {p} and {c}.",
    },
    "replay_no_key": {
        "nb": "Svar fra språkmodellen er slått av akkurat nå.",
        "en": "Answers from the language model are turned off right now.",
    },
    "replay_quota": {
        "nb": "Den gratis kvoten for språkmodellen er brukt opp for i dag.",
        "en": "The free quota for the language model is used up for today.",
    },
    "replay_ip_limit": {
        "nb": "Du har brukt opp svarene dine for i dag.",
        "en": "You have used up your answers for today.",
    },
    "error": {
        "nb": "Språkmodellen svarte ikke. Prøv igjen om litt.",
        "en": "The language model did not answer. Try again in a little while.",
    },
    "replay_hint": {
        "nb": "Under ser du paragrafene søket fant, og et ferdig svar på et av "
        "eksempelspørsmålene.",
        "en": "Below you see the sections the search found, and a ready answer to one of the "
        "example questions.",
    },
    "closest": {"nb": "Et ferdig eksempel", "en": "A ready example"},
    "examples_intro": {
        "nb": "Her er ferdige svar på spørsmål fra testsettet. De er valgt for å vise ulike "
        "typer spørsmål, også spørsmål på nynorsk og spørsmål der svaret ikke står i loven. "
        "Siden bruker ingen språkmodell. Hvor godt systemet gjør det i snitt, ser du under "
        "«Resultater».",
        "en": "Here are ready answers to questions from the test set. They were chosen to show "
        "different kinds of questions, also questions in nynorsk and questions the law does "
        "not answer. This page uses no language model. How well the system does on average is "
        "shown under «Results».",
    },
    "pick_example": {"nb": "Velg et spørsmål", "en": "Pick a question"},
    "no_examples": {"nb": "Ingen ferdige eksempler er lagt inn.", "en": "No ready examples."},
    "type": {"nb": "Type", "en": "Type"},
    "standard": {"nb": "Skriftform", "en": "Written standard"},
    "reference": {"nb": "Fasitsvar", "en": "Reference answer"},
    "judge": {"nb": "Vurdering fra dommermodellen", "en": "Verdict from the judge model"},
    "recorded_with": {
        "nb": "Svaret er laget med {model} og søkesystemet {system}.",
        "en": "The answer was made with {model} and the search system {system}.",
    },
    "compare_intro": {
        "nb": "Her ser du hvordan to varianter av ordsøket finner paragrafer for samme "
        "spørsmål. Den ene deler også lange sammensatte ord, som «merverdiavgiftsregisteret», "
        "i mindre ord. Her lages det ikke noe svar.",
        "en": "Here you see how two kinds of word search find sections for the same question. "
        "One also splits long compound words, such as «merverdiavgiftsregisteret», into "
        "smaller words. No answer is written here.",
    },
    "bm25": {"nb": "Ordsøk (BM25)", "en": "Word search (BM25)"},
    "bm25-compound": {
        "nb": "Ordsøk med orddeling",
        "en": "Word search with compound splitting",
    },
    "search_button": {"nb": "Søk", "en": "Search"},
    "tradeoff": {
        "nb": "Demoen bruker ordsøk, fordi den gratis serveren har for lite minne til de "
        "tyngre søkemodellene. I testen fant ordsøket riktig paragraf blant de ti første "
        "treffene i {b_r:.0%} av spørsmålene. Det beste søket i prosjektet, hybrid søk med "
        "omrangering, klarte {h_r:.0%}.",
        "en": "The demo uses word search, because the free server has too little memory for "
        "the heavier search models. On the test set, word search found a right section among "
        "the first ten hits for {b_r:.0%} of the questions. The best search in the project, "
        "hybrid search with reranking, managed {h_r:.0%}.",
    },
    "results_retrieval": {"nb": "Søk på testsettet", "en": "Search on the test set"},
    "results_generation": {"nb": "Svar på testsettet", "en": "Answers on the test set"},
    "results_figures": {"nb": "Figurer", "en": "Figures"},
    "results_failures": {"nb": "Eksempler på feil", "en": "Examples of mistakes"},
    "corpus": {"nb": "Lovtekster og oppdeling", "en": "Law texts and splitting"},
    "no_generation": {
        "nb": "Resultatene for svar er ikke lagt inn ennå.",
        "en": "The answer results are not added yet.",
    },
    "results_note": {
        "nb": "Tallene er snitt over spørsmålene i testsettet. Intervallene viser hvor sikre "
        "tallene er (95 prosent konfidensintervall).",
        "en": "The numbers are averages over the test questions. The intervals show how "
        "certain the numbers are (95 percent confidence interval).",
    },
    "count": {"nb": "Antall", "en": "Count"},
}

FAILURE_KINDS = {
    "wrong_answer": {"nb": "Feil svar", "en": "Wrong answer"},
    "false_refusal": {
        "nb": "Svarte ikke, selv om svaret står i loven",
        "en": "Did not answer, although the law has the answer",
    },
    "missed_refusal": {
        "nb": "Svarte, selv om svaret ikke står i loven",
        "en": "Answered, although the law does not have the answer",
    },
    "unsupported_citation": {
        "nb": "Paragrafen sier ikke det svaret påstår",
        "en": "The section does not say what the answer claims",
    },
    "citation_outside_sources": {
        "nb": "Viste til en paragraf søket ikke fant",
        "en": "Cited a section the search did not find",
    },
    "wrong_written_standard": {
        "nb": "Feil skriftform (bokmål eller nynorsk)",
        "en": "Wrong written standard (bokmål or nynorsk)",
    },
    "invalid_output": {"nb": "Svaret hadde feil format", "en": "The answer had the wrong format"},
}

ABOUT = {
    "nb": """\
### Hva er dette?

Et system som svarer på spørsmål om norsk lov ut fra selve lovteksten. Det finner paragrafene
som passer best til spørsmålet, og lar en språkmodell svare bare ut fra dem. Hver setning i
svaret skal vise til en paragraf, så du kan sjekke svaret selv. Står ikke svaret i teksten,
sier systemet det i stedet for å gjette.

Metoden virker på alle norske lover og forskrifter. Skatteretten er brukt til å teste den.

Laget av Reza Azad Gholami. Kildekoden ligger på
[github.com/eigenreza/norsk-lov-rag](https://github.com/eigenreza/norsk-lov-rag).

### Slik virker det

1. Lovtekstene er delt opp i paragrafer, med navnet på loven og overskriften først.
2. Søket finner de paragrafene som passer best til spørsmålet ditt.
3. De fem beste sendes til språkmodellen sammen med spørsmålet og faste regler for svaret.
4. Systemet sjekker at paragrafene i svaret var blant dem søket fant.

Demoen bruker ordsøk (BM25), som passer på en liten gratis server. Ordsøk finner bare lovens
egne ord. Derfor gjør språkmodellen først spørsmålet om til søkeord slik loven skriver dem,
og søket bruker både spørsmålet og søkeordene. Slik kan du også spørre på engelsk.
Prosjektet testet også tyngre søkemodeller, som er bedre. Tallene står under «Resultater».

For at svaret skal komme på noen sekunder, bruker demoen modellen `gpt-oss-120b` hos Groq.
Evalueringen i prosjektet brukte `gemma-4-31b-it`, som er mye tregere på sitt gratis nivå.

### Hvor kommer lovteksten fra?

Fra Lovdatas åpne datasett med gjeldende lover og sentrale forskrifter. Demoen har de viktigste
skattelovene og skatteforskriftene, til sammen {sections} paragrafer. Versjonen av lovteksten
er merket med koden `{corpus_hash}`.

### Hvordan er systemet testet?

Med et testsett av spørsmål som er laget og gått gjennom med språkmodeller. Det har direkte
spørsmål, spørsmål på hverdagsspråk, spørsmål som trenger to paragrafer, spørsmål der svaret
ikke står i loven, og spørsmål på nynorsk.

### Hva bør du vite?

Svarene kan være feil, også når de viser til en paragraf. Lovteksten ble hentet på et bestemt
tidspunkt og kan ha endret seg. Satser og beløpsgrenser for et bestemt år står ofte i
Stortingets årlige vedtak, som ikke er med. Svarene er ikke juridisk råd eller skatteråd.

### Personvern

Spørsmålet og paragrafene søket fant, sendes til Groq, som lager søkeordene og svaret med
modellen `gpt-oss-120b` på gratis nivå. Groq lagrer ikke innholdet i forespørslene som
standard, men kan beholde logger i inntil 30 dager for å hindre misbruk. Demoen lagrer ikke
spørsmålet ditt. Loggen har bare en kode laget av spørsmålet
(en hash), lengden, hvor lang tid svaret tok, antall tokens og utfallet. IP-adressen din brukes
bare til å telle svar per dag, og den skrives ikke i loggen.

Du kan få {per_ip} svar per dag, og et spørsmål kan ha høyst {max_chars} tegn. Når grensen er
nådd, eller den gratis kvoten er brukt opp, viser demoen ferdige svar i stedet, og da sendes
ingenting til Groq. Sidene «Eksempler» og «Sammenlign søk» bruker aldri en språkmodell.
""",
    "en": """\
### What is this?

A system that answers questions about Norwegian law from the law text itself. It finds the
sections that fit the question best and lets a language model answer from them only. Every
sentence in the answer must cite a section, so you can check the answer yourself. When the text
does not have the answer, the system says so instead of guessing.

The method works on any Norwegian law or regulation. Tax law is used to test it.

Made by Reza Azad Gholami. The source code is at
[github.com/eigenreza/norsk-lov-rag](https://github.com/eigenreza/norsk-lov-rag).

### How it works

1. The law texts are split into sections, each starting with the name of the law and the
   heading.
2. The search finds the sections that fit your question best.
3. The top five go to the language model with the question and fixed rules for the answer.
4. The system checks that the sections cited in the answer were among those the search found.

The demo uses word search (BM25), which fits on a small free server. Word search only finds
the law's own words. So the language model first turns the question into search terms the way
the law writes them, and the search uses both the question and the terms. This is also why you
can ask in English. The project also tested heavier search models, which do better. The
numbers are under «Results».

To answer in a few seconds, the demo uses the model `gpt-oss-120b` at Groq. The evaluation in
the project used `gemma-4-31b-it`, which is much slower on its free tier.

### Where does the law text come from?

From Lovdata's open datasets of current laws and central regulations. The demo holds the main
tax laws and tax regulations, {sections} sections in all. The version of the law text is marked
with the code `{corpus_hash}`.

### How was the system tested?

With a test set of questions that were written and reviewed with language models. It has
direct questions, everyday questions, questions that need two sections, questions the law does
not answer, and questions in nynorsk.

### What should you know?

Answers can be wrong, also when they cite a section. The law text was collected at one point in
time and may have changed. Rates and amount limits for a given year are often set in the yearly
decisions of the Storting, which are not included. The answers are not legal or tax advice.

### Privacy

The question and the sections the search found are sent to Groq, which writes the search terms
and the answer with the model `gpt-oss-120b` on its free tier. Groq does not store the content
of requests by default, but may keep logs for up to 30 days to prevent misuse. The demo does
not store your question. The log keeps only a
code made from the question (a hash), its length, how long the answer took, token counts and
the outcome. Your IP address is used only to count answers per day, and it is not logged.

You can get {per_ip} answers per day, and a question can have at most {max_chars} characters.
When the limit is reached, or the free quota is used up, the demo shows ready answers instead,
and nothing is sent to Groq. The pages «Examples» and «Compare search» never use a language
model.
""",
}
