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
from norsk_lov_rag.generation.prompts import ANSWER_PROMPT_VERSION, answer_messages
from norsk_lov_rag.generation.sources import SourceBook
from norsk_lov_rag.llm import MissingKey, ModelClient, ProviderError, QuotaExhausted
from norsk_lov_rag.llm.client import read_env_file
from norsk_lov_rag.retrieval.lexical import BM25Index, CompoundSplitter

log = logging.getLogger("norsk_lov_rag.demo")

ROOT = Path(__file__).resolve().parents[3]
MAX_QUESTION_CHARS = 500
ANSWER_ROLE = "gemma"
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

    def search(self, question: str, system: str = "bm25", k: int = 10) -> list[dict]:
        """Sections in ranked order, each with the score of its best chunk."""
        hits = self.index(system).search(question, 100)
        out: list[dict] = []
        seen: set[str] = set()
        for row, score in hits:
            for unit in self.chunk_units[row]:
                if unit in seen:
                    continue
                seen.add(unit)
                out.append({"unit": unit, "label": self.book.labels.get(unit, unit),
                            "title": self.book.titles.get(unit, ""), "score": score})  # fmt: skip
        return out[:k] if k else out


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
    model = settings.llm.models[ANSWER_ROLE]
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


def build_sources(data: DemoData, settings: Settings, hits: list[dict]):
    gen = settings.generation
    units = [hit["unit"] for hit in hits]
    return data.book.sources(units, gen.top_k, gen.max_words_per_source, gen.max_context_words)


def answer_question(data: DemoData, settings: Settings, client, question: str,
                    hits: list[dict]) -> Answer:  # fmt: skip
    """A live answer from the retrieved sections, with the same prompt as the evaluation."""
    started = time.perf_counter()
    sources = build_sources(data, settings, hits)
    messages = answer_messages(question, sources)
    try:
        result = client.chat(
            ANSWER_ROLE, messages, ANSWER_PROMPT_VERSION, settings.generation.max_tokens
        )
    except MissingKey:
        return Answer("replay_no_key", seconds=time.perf_counter() - started)
    except QuotaExhausted:
        return Answer("replay_quota", seconds=time.perf_counter() - started)
    except ProviderError:
        return Answer("error", seconds=time.perf_counter() - started)
    parsed = parse_output(result.text)
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
