"""Logic of the demo, kept apart from Streamlit so that it can be tested without a browser.

Privacy: the question text is never written anywhere. The log keeps a hash of the question,
its length, the time taken, token counts and the outcome. Answers are not cached, so that no
answer that repeats a question is stored either.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from norsk_lov_rag.config import Settings
from norsk_lov_rag.corpus.language import marker_counts
from norsk_lov_rag.generation.answers import analyse, parse_output
from norsk_lov_rag.generation.sources import Source, SourceBook
from norsk_lov_rag.llm import MissingKey, ModelClient, ProviderError, QuotaExhausted
from norsk_lov_rag.llm.client import read_env_file
from norsk_lov_rag.retrieval.lexical import BM25Index, CompoundSplitter

log = logging.getLogger("norsk_lov_rag.demo")

ROOT = Path(__file__).resolve().parents[3]
MAX_QUESTION_CHARS = 500
# The demo answers with a fast model on a free tier, so that an answer takes seconds. The
# evaluation in docs/valideringsrapport.md used Gemma, which is much slower on its free tier.
ANSWER_ROLE = os.environ.get("DEMO_MODEL_ROLE", "demo")
# Search terms and the pick among candidates run on a second model with its own budget.
SEARCH_ROLE = os.environ.get("DEMO_SEARCH_ROLE", "demo_search")
DEMO_ANSWER_MAX_TOKENS = 800
TERMS_MAX_TOKENS = 200
TERMS_PROMPT_VERSION = "demo-terms-v2"
DEMO_ANSWER_PROMPT_VERSION = "demo-answer-v2"
RRF_K = 60
TERMS_WEIGHT = 1.0
NYNORSK_QUESTION_WEIGHT = 0.3
# The demo model has a long context, so it gets more sections than the 4k model in the
# evaluation could take.
DEMO_TOP_K = 6
DEMO_WORDS_PER_SOURCE = 230
DEMO_CONTEXT_WORDS = 1400
# Live answers per visitor per day, and for the whole demo per day. Both can be set in the
# app settings of the host. The free quota of the model service is checked as well.
DAILY_LIMIT_PER_IP = int(os.environ.get("DEMO_DAILY_LIMIT_PER_IP", "5"))
DAILY_LIMIT_TOTAL = int(os.environ.get("DEMO_DAILY_LIMIT_TOTAL", "100"))
# The demo gives up sooner than the evaluation runs when the service has a bad moment.
DEMO_MAX_RETRIES = 2
SYSTEMS = ("bm25", "bm25-compound")
WORD = re.compile(r"[0-9a-zæøåéü]+")


def data_dir() -> Path:
    return Path(os.environ.get("DEMO_DATA_DIR", ROOT / "demo_data"))


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_book(data: dict) -> SourceBook:
    """A source book rebuilt from demo_data, with the same methods as in the evaluation."""
    book = object.__new__(SourceBook)
    book.labels = data["labels"]
    book.titles = data["titles"]
    book.texts = data["texts"]
    book._lookup = {(name, number): unit for name, number, unit in data["lookup"]}
    return book


@dataclass
class DemoData:
    folder: Path
    texts: list[str]
    chunk_units: list[list[str]]
    book: SourceBook
    examples: list[dict]
    manifest: dict
    results: dict[str, dict]
    figures: list[Path]
    _indexes: dict[str, BM25Index] = field(default_factory=dict)

    @classmethod
    def load(cls, folder: Path | None = None) -> DemoData:
        folder = folder or data_dir()
        chunks = _read(folder / "chunks.json")
        results = {path.stem: _read(path) for path in sorted((folder / "results").glob("*.json"))}
        return cls(
            folder=folder,
            texts=chunks["texts"],
            chunk_units=chunks["units"],
            book=load_book(_read(folder / "sections.json")),
            examples=_read(folder / "examples.json")["items"],
            manifest=_read(folder / "manifest.json"),
            results=results,
            figures=sorted((folder / "figures").glob("*.png")),
        )

    def index(self, system: str) -> BM25Index:
        if system not in SYSTEMS:
            raise ValueError(f"the demo has no retrieval system {system!r}")
        if system not in self._indexes:
            splitter = CompoundSplitter.from_texts(self.texts) if system != "bm25" else None
            self._indexes[system] = BM25Index(self.texts, splitter=splitter)
        return self._indexes[system]

    def _ranked(self, text: str, system: str) -> list[tuple[str, float]]:
        """Sections in ranked order for one search text, each with its best chunk score."""
        out: list[tuple[str, float]] = []
        seen: set[str] = set()
        for row, score in self.index(system).search(text, 100):
            for unit in self.chunk_units[row]:
                if unit not in seen:
                    seen.add(unit)
                    out.append((unit, score))
        return out

    def search(
        self, question: str, system: str = "bm25", k: int = 10, terms: str = ""
    ) -> list[dict]:
        """Sections in ranked order, each with the score of its best chunk.

        With search terms from the model, the ranking for the question and the ranking for the
        terms are fused with reciprocal rank fusion, and the score is the fused score.
        """
        ranked = self._ranked(search_text(question), system)
        if terms.strip():
            # The law is written in bokmål. An English question shares no words with it, and a
            # nynorsk question shares few, so the model's terms count most for those.
            lists = [(TERMS_WEIGHT, self._ranked(terms, system))]
            if not is_english(question):
                own = 1.0 if guess_standard(question) == "nb" else NYNORSK_QUESTION_WEIGHT
                lists.append((own, ranked))
            fused: dict[str, float] = {}
            for weight, ranking in lists:
                for rank, (unit, _) in enumerate(ranking, start=1):
                    fused[unit] = fused.get(unit, 0.0) + weight / (RRF_K + rank)
            ranked = sorted(((u, s * 1000) for u, s in fused.items()), key=lambda x: -x[1])
        out = [
            {"unit": unit, "label": self.book.labels.get(unit, unit),
             "title": self.book.titles.get(unit, ""), "score": score}
            for unit, score in ranked
        ]  # fmt: skip
        return out[:k] if k else out


# Words that ask for a kind of answer ("kort svar", "forklar enkelt") rather than name the
# topic. They are left out of the search only; the model still gets the full question.
INSTRUCTION_WORDS = re.compile(
    r"\b(kort|kortfatta|kortfattet|svar|svare|forklar|forklaring|enkelt|enkel|vennligst|"
    r"takk|detaljert|utfyllende|oppsummer|oppsummering)\b",
    re.IGNORECASE,
)


def search_text(question: str) -> str:
    """The question without words that only say how to answer."""
    cleaned = " ".join(INSTRUCTION_WORDS.sub(" ", question).split())
    return cleaned or question


# Word search only finds the law's own words. A visitor asks in everyday words, or in English,
# so the model first turns the question into the terms the law uses.
TERMS_PROMPT = (
    "Du hjelper et søk i norske skattelover og skatteforskrifter. Skriv om spørsmålet under til "
    "8 til 15 søkeord på bokmål. Bruk de formelle ordene loven bruker, også synonymer og nære "
    "begreper, for eksempel «foreldrefradrag» for barnepass, «fritidsbolig» for hytte, "
    "«ektefeller» for gift og «arbeidsreise» for pendling. Rett opp skrivefeil. Spørsmålet kan "
    "være på bokmål, nynorsk eller engelsk. Svar bare med søkeordene, skilt med mellomrom.\n\n"
    "Spørsmål: {question}"
)

ENGLISH_WORDS = frozenset(
    "the is are am was will would can could should do does did what which who whom how when "
    "where why my me you your if have has had pay paid tax taxes less more children child "
    "get an of and with about there this that from income deduct deduction".split()
)
ENGLISH_RULE = (
    "Spørsmålet er på engelsk. Svar på engelsk, og sett kildene i hakeparentes som før. Hvis "
    "ingen av utdragene handler om spørsmålet, skriv bare: «{refusal}»"
)
ENGLISH_REFUSAL = "I cannot find the answer in the law texts I was given."
ENGLISH_RULE2 = (
    "2. Spørsmålet er på engelsk. Skriv hele svaret på engelsk, uten å kommentere språket."
)

# The demo is more helpful than the strict contract in the evaluation: it answers with what the
# excerpts do say, and says what they leave open, instead of refusing as soon as an excerpt
# does not answer the question word for word.
PARTIAL_RULE = (
    "Tillegg til reglene: Mange spørsmål er stilt med vanlige ord og har ikke et ja eller nei i "
    "loven. Hvis utdragene gir regler om temaet, forklar hva reglene sier og hva de betyr for "
    "situasjonen i spørsmålet, og si kort hva utdragene ikke sier. Et eksempel: på spørsmålet "
    "«gir det lavere skatt å gifte seg?» kan svaret forklare hvordan ektefeller skattlegges, "
    "hvis utdragene sier det. Bruk den faste setningen for avslag bare hvis ingen av utdragene "
    "handler om temaet. Svar aldri med noe som ikke står i utdragene."
)

# Word search finds candidates. The model then reads the name, heading and start of each
# candidate and picks the sections most likely to hold the answer, like the reranker in the
# evaluation, which gave the largest gain.
SELECT_CANDIDATES = 30
SELECT_WORDS = 12
SELECT_MAX_TOKENS = 300
SELECT_PROMPT_VERSION = "demo-select-v1"
SELECT_PROMPT = (
    "Under er et spørsmål om norsk skatterett og en nummerert liste med paragrafer fra "
    "skattelovene, med navn, overskrift og starten av teksten. Velg de paragrafene som mest "
    "sannsynlig gir svaret eller regler som trengs for å svare, høyst {k}, den beste først. "
    "Spørsmålet kan ha skrivefeil eller være på engelsk. Svar bare med numrene, skilt med "
    "komma.\n\nSpørsmål: {question}\n\nParagrafer:\n{items}"
)


def chat_with_fallback(client, roles: tuple[str, ...], messages, version: str, max_tokens: int):
    """Ask the first model; if its free daily quota is used up, ask the next one.

    Both models are on free tiers with their own quotas, so this never leads to a paid option.
    """
    models = getattr(getattr(client, "cfg", None), "models", None)
    error: QuotaExhausted | None = None
    for role in roles:
        if models is not None and role not in models:
            continue
        try:
            return client.chat(role, messages, version, max_tokens)
        except QuotaExhausted as exc:
            error = exc
    raise error or QuotaExhausted("no demo model has quota left")


def select_sections(client, data, question: str, candidates: list[dict], k: int) -> list[dict]:
    """The candidates the model picks first, then the rest in search order."""
    items = "\n".join(
        f"{n}. {c['label']} {c['title']}: "
        + " ".join(data.book.text(c["unit"]).split()[:SELECT_WORDS])
        for n, c in enumerate(candidates, start=1)
    )
    prompt = SELECT_PROMPT.format(k=k, question=question, items=items)
    try:
        result = chat_with_fallback(client, (SEARCH_ROLE, ANSWER_ROLE),
                                    [{"role": "user", "content": prompt}],
                                    SELECT_PROMPT_VERSION, SELECT_MAX_TOKENS)  # fmt: skip
    except (MissingKey, QuotaExhausted, ProviderError):
        return candidates
    picked: list[int] = []
    for number in re.findall(r"\d+", result.text):
        index = int(number) - 1
        if 0 <= index < len(candidates) and index not in picked:
            picked.append(index)
    picked = picked[:k]
    rest = [n for n in range(len(candidates)) if n not in picked]
    return [candidates[n] for n in picked + rest]


def smart_search(
    client, data, question: str, k: int = 10, terms: str | None = None
) -> tuple[str, list[dict]]:
    """Search terms from the model, word search, then the model's pick among the candidates."""
    if terms is None:
        terms = search_terms(client, question)
    candidates = data.search(question, "bm25", k=SELECT_CANDIDATES, terms=terms)
    candidates += core_candidates(data, {c["unit"] for c in candidates})
    ranked = select_sections(client, data, question, candidates, DEMO_TOP_K)
    return terms, ranked[:k]


# The main rules of the tax act. They are always among the candidates, so that a question with
# no special rule in the law (a pet, being a student) can be answered with the general rule.
CORE_SECTIONS = tuple(
    f"skatteloven § {n}"
    for n in (
        "5-1",
        "6-1",
        "15-4",
        "5-10",
        "5-12",
        "5-50",
        "5-40",
        "5-43",
        "6-30",
        "9-2",
        "12-2",
        "17-1",
        "2-1",
    )  # fmt: skip
)


def core_candidates(data, present: set[str]) -> list[dict]:
    units = {label: unit for unit, label in data.book.labels.items()}
    return [
        {"unit": units[label], "label": label, "title": data.book.titles.get(units[label], ""),
         "score": 0.0}
        for label in CORE_SECTIONS
        if label in units and units[label] not in present
    ]  # fmt: skip


# -- conversation --------------------------------------------------------------------------------
# Every message is sorted first. A tax question goes to the law search. A question about the
# demo itself, small talk, or a question outside tax law gets a fitting answer instead of a
# refusal. The same call turns a follow-up into a question that stands on its own.

ROUTE_PROMPT_VERSION = "demo-route-v1"
ROUTE_MAX_TOKENS = 500
HISTORY_TURNS = 4
ROUTE_PROMPT = (
    "Du er første steg i en demo som svarer på spørsmål om norsk skatterett ut fra lovteksten. "
    "Demoen har disse lovene og forskriftene: skatteloven, skatteforvaltningsloven og "
    "-forskriften, Finansdepartementets og Skattedirektoratets skatteforskrifter, "
    "merverdiavgiftsloven og -forskriften, og skattebetalingsloven og -forskriften.\n\n"
    "Les den siste meldingen fra brukeren, med samtalen før som bakgrunn, og svar bare med "
    "JSON i denne formen:\n"
    '{{"type": "tax" | "about" | "other" | "chat", "question": "...", "terms": "..."}}\n\n'
    "type: tax for spørsmål om skatt, avgift, fradrag, merverdiavgift, skattemelding, "
    "skattebetaling, arv, gaver, bolig, bil, jobb og lignende der skattereglene kan ha noe å "
    "si. about for spørsmål om demoen eller prosjektet: hva den kan, hvordan den virker, "
    "hvem som laget den, data, modeller og resultater. other for spørsmål om annen lov eller "
    "andre temaer. chat for hilsen, takk og småprat.\n"
    "question: den siste meldingen skrevet om til et spørsmål som står på egne ben, på samme "
    "språk som brukeren, med det som trengs fra samtalen før. Rett opp skrivefeil.\n"
    "terms: bare for tax, 8 til 15 søkeord på bokmål slik loven skriver dem, med formelle ord "
    "og synonymer, for eksempel «fritidsbolig» for hytte og «ektefeller» for gift. Ellers tom."
    "\n\nSamtalen før:\n{history}\n\nSiste melding: {message}"
)

ABOUT_FACTS = (
    "Demoen er laget av Reza Azad Gholami. Kildekoden ligger på "
    "github.com/eigenreza/norsk-lov-rag. Den svarer på spørsmål om norsk skatterett ut fra "
    "lovteksten i Lovdatas åpne datasett, med kilde til hver påstand, og den sier fra når "
    "loven ikke har svaret. Metoden virker på alle norske lover, men demoen har bare de ni "
    "skattelovene og -forskriftene. Lovteksten er hentet på et bestemt tidspunkt. Satser og "
    "beløpsgrenser for et bestemt år står i Stortingets årlige vedtak, som ikke er med. "
    "Svarene er ikke juridisk råd. Slik virker demoen: en språkmodell gjør spørsmålet om til "
    "lovens ord, ordsøk finner 30 kandidater, modellen velger de beste paragrafene, og "
    "modellen svarer bare ut fra dem. Demoen bruker gratis modeller hos Groq. "
    "Evalueringen i prosjektet: et testsett med 240 spørsmål. Det beste søket, hybrid søk med "
    "omrangering, fant riktig paragraf blant de ti første i 96 prosent av spørsmålene. Med "
    "Gemma 4 31B som svarmodell var 92 prosent av svarene riktige ifølge en dommermodell, og "
    "modellen avsto riktig på 41 av 42 spørsmål uten svar i loven. Den lokale norske modellen "
    "NorMistral 7B var svakere, med 64 prosent riktige svar."
)
TALK_PROMPT_VERSION = "demo-talk-v1"
TALK_MAX_TOKENS = 600
TALK_PROMPT = {
    "about": (
        "Du er assistenten i en demo. Svar kort, vennlig og presist på brukerens spørsmål om "
        "demoen, bare ut fra fakta under. Svar på samme språk som brukeren. Finnes ikke svaret "
        "i fakta, si det.\n\nFakta:\n{facts}\n\nSamtalen før:\n{history}\n\nSpørsmål: {question}"
    ),
    "chat": (
        "Du er assistenten i en demo som svarer på spørsmål om norsk skatterett ut fra "
        "lovteksten. Svar kort og vennlig på meldingen, på samme språk som brukeren, og foreslå "
        "gjerne et par eksempler på spørsmål demoen kan svare på, som fradrag for reise til "
        "jobben, skatt ved salg av bolig eller merverdiavgift.\n\nFakta om demoen:\n{facts}\n\n"
        "Samtalen før:\n{history}\n\nMelding: {question}"
    ),
    "other": (
        "Du er assistenten i en demo som bare har norske skattelover og skatteforskrifter. "
        "Brukeren spør om noe utenfor dette. Svar kort, vennlig og nyttig på samme språk som "
        "brukeren: si at demoen bare har skatteretten og derfor ikke kan svare ut fra loven her, "
        "nevn hvilket område spørsmålet hører til og at gjeldende lover finnes på lovdata.no, og "
        "foreslå et skattespørsmål i samme tema hvis det finnes et. Ikke gi juridiske råd og "
        "ikke svar på selve spørsmålet.\n\nSamtalen før:\n{history}\n\nSpørsmål: {question}"
    ),
}


@dataclass
class Route:
    kind: str
    question: str
    terms: str


def history_text(history: list[dict]) -> str:
    """The last turns of the conversation, short, for the model."""
    lines = []
    for turn in history[-HISTORY_TURNS:]:
        who = "Bruker" if turn["role"] == "user" else "Assistent"
        lines.append(f"{who}: {' '.join(str(turn['content']).split())[:400]}")
    return "\n".join(lines) or "(ingen)"


def _json_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        value = json.loads(text[start : end + 1])
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def route_message(client, message: str, history: list[dict]) -> Route:
    """Sort the message and make it a question that stands on its own."""
    prompt = ROUTE_PROMPT.format(history=history_text(history), message=message)
    try:
        result = chat_with_fallback(client, (SEARCH_ROLE, ANSWER_ROLE),
                                    [{"role": "user", "content": prompt}],
                                    ROUTE_PROMPT_VERSION, ROUTE_MAX_TOKENS)  # fmt: skip
    except (MissingKey, QuotaExhausted, ProviderError):
        return Route("tax", message, "")
    data = _json_object(result.text)
    kind = data.get("type") if data.get("type") in ("tax", "about", "other", "chat") else "tax"
    question = str(data.get("question") or message).strip() or message
    terms = " ".join(WORD.findall(str(data.get("terms") or "").lower())[:20])
    return Route(kind, question, terms)


def talk_answer(client, kind: str, question: str, history: list[dict]) -> Answer:
    """An answer about the demo, to small talk, or to a question outside tax law."""
    started = time.perf_counter()
    prompt = TALK_PROMPT[kind].format(
        facts=ABOUT_FACTS, history=history_text(history), question=question
    )
    try:
        result = chat_with_fallback(client, (ANSWER_ROLE, SEARCH_ROLE),
                                    [{"role": "user", "content": prompt}],
                                    TALK_PROMPT_VERSION, TALK_MAX_TOKENS)  # fmt: skip
    except MissingKey:
        return Answer("replay_no_key", seconds=time.perf_counter() - started)
    except QuotaExhausted:
        return Answer("replay_quota", seconds=time.perf_counter() - started)
    except ProviderError:
        return Answer("error", seconds=time.perf_counter() - started)
    return Answer(
        outcome="talk",
        answer=result.text.strip(),
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        seconds=time.perf_counter() - started,
        model_version=result.model_version,
    )


def is_english(question: str) -> bool:
    words = WORD.findall(question.lower())
    return bool(words) and sum(w in ENGLISH_WORDS for w in words) / len(words) >= 0.3


def search_terms(client, question: str) -> str:
    """Search terms in the words of the law, or an empty string if the model fails."""
    messages = [{"role": "user", "content": TERMS_PROMPT.format(question=question)}]
    try:
        result = chat_with_fallback(client, (SEARCH_ROLE, ANSWER_ROLE), messages,
                                    TERMS_PROMPT_VERSION, TERMS_MAX_TOKENS)  # fmt: skip
    except (MissingKey, QuotaExhausted, ProviderError):
        return ""
    return " ".join(WORD.findall(result.text.lower())[:20])


def question_hash(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def log_event(question: str, outcome: str, seconds: float, prompt_tokens=0, completion_tokens=0):
    """One log line per question, without the question text."""
    record = {
        "event": "question",
        "sha256": question_hash(question),
        "length": len(question),
        "seconds": round(seconds, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "outcome": outcome,
    }
    log.info(json.dumps(record))
    return record


def guess_standard(text: str) -> str:
    bokmal, nynorsk = marker_counts(text)
    return "nn" if nynorsk > bokmal else "nb"


def check_question(question: str) -> str | None:
    """A reason to refuse the question before any search, or None."""
    if not question.strip():
        return "empty"
    if len(question) > MAX_QUESTION_CHARS:
        return "too_long"
    return None


class RateLimiter:
    """Daily counts per visitor and in total, kept in memory only.

    The visitor is a salted hash of the IP address, and the counts reset each day in UTC.
    """

    def __init__(self, per_ip: int = DAILY_LIMIT_PER_IP, total: int = DAILY_LIMIT_TOTAL,
                 clock=time.time):  # fmt: skip
        self.per_ip = per_ip
        self.total = total
        self.clock = clock
        self._salt = os.urandom(16)
        self._day = ""
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def _visitor(self, ip: str | None) -> str:
        return hashlib.sha256(self._salt + (ip or "unknown").encode()).hexdigest()[:16]

    def _roll(self) -> None:
        day = datetime.fromtimestamp(self.clock(), UTC).strftime("%Y-%m-%d")
        if day != self._day:
            self._day = day
            self._counts.clear()

    def allow(self, ip: str | None) -> bool:
        """Count one live answer if both limits allow it."""
        with self._lock:
            self._roll()
            visitor = self._visitor(ip)
            if self._counts[visitor] >= self.per_ip or self._counts["*"] >= self.total:
                return False
            self._counts[visitor] += 1
            self._counts["*"] += 1
            return True

    def left(self, ip: str | None) -> int:
        with self._lock:
            self._roll()
            used = self._counts[self._visitor(ip)]
            return max(0, min(self.per_ip - used, self.total - self._counts["*"]))


class NullCache:
    """No answer is stored, so no trace of a question is kept on disk."""

    def get(self, key: str):
        return None

    def put(self, key: str, record: dict) -> None:
        return None


def has_key(settings: Settings) -> bool:
    """Whether the key for the answer model is set. The key itself is never returned."""
    model = settings.llm.models.get(ANSWER_ROLE)
    if model is None:
        return False
    name = settings.llm.providers[model.provider].api_key_env
    if name is None:
        return True
    return bool(os.environ.get(name) or read_env_file(settings.llm.env_file).get(name))


def make_client(settings: Settings, state_dir: Path | None = None) -> ModelClient:
    """The project's client, with no answer cache and fewer retries."""
    for role in (ANSWER_ROLE, SEARCH_ROLE):
        model = settings.llm.models.get(role)
        if model is not None:
            model.max_retries = min(model.max_retries or DEMO_MAX_RETRIES, DEMO_MAX_RETRIES)
    client = ModelClient(settings.llm, state_dir or settings.paths.llm)
    client.cache = NullCache()
    return client


@dataclass
class Answer:
    outcome: str
    answer: str = ""
    abstained: bool = False
    citations: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    model_version: str = ""


def focused_text(text: str, query: str, max_words: int) -> str:
    """A long section cut to the lines that share the most words with the query.

    The first line is kept for context, the lines stay in their order, and a gap is shown as
    "(…)". Words are compared by their first five letters, so that inflected forms match.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if sum(len(line.split()) for line in lines) <= max_words:
        return " ".join(lines)
    keys = {w[:5] for w in WORD.findall(query.lower()) if len(w) > 3}

    def score(line: str) -> int:
        return sum(w[:5] in keys for w in WORD.findall(line.lower()))

    order = sorted(range(len(lines)), key=lambda i: (-score(lines[i]), i))
    chosen: set[int] = set()
    used = 0
    for i in [0, *order]:
        size = len(lines[i].split())
        if i in chosen or used + size > max_words:
            continue
        chosen.add(i)
        used += size
    out: list[str] = []
    previous = -1
    for i in sorted(chosen):
        if previous >= 0 and i != previous + 1:
            out.append("(…)")
        out.append(lines[i])
        previous = i
    return " ".join(out)


def build_sources(data: DemoData, settings: Settings, hits: list[dict], query: str = ""):
    """The sections for the model, with long sections cut to the lines that fit the query."""
    out: list[Source] = []
    left = DEMO_CONTEXT_WORDS
    for hit in hits[:DEMO_TOP_K]:
        unit = hit["unit"]
        text = focused_text(data.book.text(unit), query, min(DEMO_WORDS_PER_SOURCE, left))
        if len(text.split()) < 30 and left < DEMO_WORDS_PER_SOURCE:
            break
        out.append(Source(unit, data.book.labels[unit], data.book.titles[unit], text))
        left -= len(text.split())
        if left <= 0:
            break
    return out


DEMO_ANSWER_PROMPT = (
    "Du er en kunnskapsrik og hjelpsom assistent som forklarer norsk skatterett for vanlige "
    "folk. Under får du utdrag fra skattelovene og et spørsmål.\n\n"
    "Slik svarer du:\n"
    "1. Finn ut hva brukeren egentlig lurer på, også når spørsmålet er stilt med vanlige ord, "
    "har skrivefeil eller handler om en livssituasjon, som alder, barn, bolig, jobb, gaver "
    "eller dyr.\n"
    "2. Begynn med et direkte svar i én setning: ja, nei eller det kommer an på, og hvorfor.\n"
    "3. Forklar så kort reglene i utdragene som gjelder situasjonen. Sett kilden i "
    "hakeparentes etter hver setning som bygger på et utdrag, nøyaktig slik den står foran "
    "utdraget, for eksempel [skatteloven § 6-44].\n"
    "4. Har utdragene ingen særregel for situasjonen, si tydelig at du ikke fant noen særregel "
    "om dette i skattelovene demoen har, og forklar den generelle regelen som gjelder, hvis et "
    "utdrag viser den, for eksempel hovedregelen om inntekt eller fradrag.\n"
    "5. Du kan bruke vanlig kunnskap til å forklare ord og sammenhenger, men da uten kilde, og "
    "aldri slik at det ser ut som det står i loven. Ikke finn på regler, satser, beløp eller "
    "paragrafer.\n"
    "6. Svar på samme språk som spørsmålet.{language_hint} Skriv klart og kort, høyst seks "
    "setninger, som vanlig tekst uten overskrifter eller lister.\n\n"
    "Utdrag:\n\n{context}\n\nSpørsmål: {question}"
)

# The base rules refuse as soon as the excerpts do not answer. The demo refuses only when no
# excerpt is on the topic, and otherwise explains what the excerpts say.
ROUND_CITATION = re.compile(r"\(([^()\[\]]{2,80}?§\s*\d[^()\[\]]{0,40}?)\)")
STRICT_RULE = "4. Hvis utdragene ikke gir svar på spørsmålet, skal du ikke gjette."
TOPIC_RULE = (
    "4. Hvis ingen av utdragene handler om temaet i spørsmålet, skal du ikke gjette. Hvis de "
    "handler om temaet, men ikke svarer helt, forklar det de sier og si hva de ikke sier."
)
STRICT_REMINDER = "skriv den faste setningen hvis utdragene ikke svarer på spørsmålet"
TOPIC_REMINDER = "skriv den faste setningen bare hvis ingen av utdragene handler om temaet"


def demo_answer_messages(question: str, sources: list[Source], language: str) -> list[dict]:
    """The demo's answer prompt: a helpful explanation for ordinary people, built on the
    excerpts, with a source for every rule, and honest about rules the law does not have."""
    blocks = []
    for source in sources:
        head = f"[{source.label}]" + (f" {source.title}" if source.title else "")
        blocks.append(f"{head}\n{source.text}")
    context = "\n\n".join(blocks) if blocks else "(ingen utdrag)"
    hint = {"en": " Spørsmålet er på engelsk, så svar på engelsk.",
            "nn": " Spørsmålet er på nynorsk, så svar på nynorsk."}.get(language, "")  # fmt: skip
    prompt = DEMO_ANSWER_PROMPT.format(context=context, question=question, language_hint=hint)
    return [{"role": "user", "content": prompt}]


def answer_question(data: DemoData, settings: Settings, client, question: str,
                    hits: list[dict], terms: str = "", original: str = "") -> Answer:  # fmt: skip
    """A live answer from the retrieved sections, written for ordinary people. The language
    of the answer follows the visitor's own message when it is given."""
    started = time.perf_counter()
    sources = build_sources(data, settings, hits, f"{question} {terms}")
    said = original or question
    english = is_english(said)
    language = "en" if english else guess_standard(said)
    messages = demo_answer_messages(question, sources, language)
    try:
        result = chat_with_fallback(client, (ANSWER_ROLE, SEARCH_ROLE), messages,
                                    DEMO_ANSWER_PROMPT_VERSION,
                                    DEMO_ANSWER_MAX_TOKENS)  # fmt: skip
    except MissingKey:
        return Answer("replay_no_key", seconds=time.perf_counter() - started)
    except QuotaExhausted:
        return Answer("replay_quota", seconds=time.perf_counter() - started)
    except ProviderError:
        return Answer("error", seconds=time.perf_counter() - started)
    # Some models write citations in full-width or round brackets. Read them as ordinary ones.
    text = result.text.replace("【", "[").replace("】", "]")
    parsed = parse_output(ROUND_CITATION.sub(r"[\1]", text))
    if english and ENGLISH_REFUSAL.lower().rstrip(".") in parsed["answer"].lower():
        parsed["abstain"] = True
    shown = [s.unit for s in sources]
    checks = analyse(
        parsed, {"written_standard": guess_standard(question), "gold_section_ids": []},
        shown, data.book,
    )  # fmt: skip
    citations = [
        {**c, "label": data.book.labels.get(c["unit"], c["citation"]) if c["unit"] else None,
         "text": data.book.text(c["unit"]) if c["unit"] else ""}
        for c in checks["citations"]
    ]  # fmt: skip
    return Answer(
        outcome="abstain" if checks["abstained"] else "answer",
        answer=parsed["answer"],
        abstained=checks["abstained"],
        citations=citations,
        sources=[
            {"unit": s.unit, "label": s.label, "words": len(s.text.split())} for s in sources
        ],  # fmt: skip
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        seconds=time.perf_counter() - started,
        model_version=result.model_version,
    )


def closest_example(question: str, examples: list[dict]) -> dict | None:
    """The recorded example that shares the most words with the question, for replay."""
    words = set(WORD.findall(question.lower()))
    best, best_score = None, 0
    for example in examples:
        score = len(words & set(WORD.findall(example["question"].lower())))
        if score > best_score:
            best, best_score = example, score
    return best


METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")


def retrieval_rows(results: dict, setting: str) -> list[dict]:
    """One row per retrieval system with the mean and interval of each metric."""
    systems = results.get("settings", {}).get(setting, {}).get("systems", {})
    rows = []
    for name, scores in systems.items():
        row = {"system": name, "n": scores.get("n")}
        for metric in METRICS:
            value = scores.get("all", {}).get(metric)
            if value:
                row[metric] = value["mean"]
                row[f"{metric} CI"] = f"{value['low']:.3f} til {value['high']:.3f}"
        rows.append(row)
    return rows


GENERATION_METRICS = (
    "correctness",
    "citation_precision",
    "citation_recall",
    "support",
    "correct_refusals",
    "false_refusals",
    "language_match",
    "latency_median",
)


def generation_rows(results: dict) -> list[dict]:
    rows = []
    for name, run in results.get("runs", {}).items():
        row = {"run": name}
        for metric in GENERATION_METRICS:
            value = run.get(metric)
            if isinstance(value, dict) and "mean" in value:
                row[metric] = value["mean"]
            elif isinstance(value, int | float):
                row[metric] = value
        rows.append(row)
    return rows


def tradeoff(results: dict) -> dict | None:
    """BM25 against the best evaluated system on the tax corpus, for the demo's choice."""
    systems = results.get("settings", {}).get("tax-structure", {}).get("systems", {})
    if "bm25" not in systems or "hybrid+rerank" not in systems:
        return None
    pick = {}
    for name in ("bm25", "hybrid+rerank"):
        pick[name] = {m: systems[name]["all"][m]["mean"] for m in ("recall@10", "ndcg@10")}
    return pick
