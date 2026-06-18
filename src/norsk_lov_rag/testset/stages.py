"""The model stages of the test set: write, check and review.

Each stage sends small batches to one model and reads a JSON answer. A batch that cannot be
read marks its items as failed, with the reason. When a model's free quota is used up, the stage
stops and the remaining items stay pending, so the next run continues from the cache.
"""

from __future__ import annotations

import json
import logging

from norsk_lov_rag.llm import CacheMiss, ModelClient, ProviderError, QuotaExhausted
from norsk_lov_rag.testset import prompts as p

log = logging.getLogger(__name__)

WRITER_BATCH = 3
WRITER_BATCH_CHARS = 9000
WRITER_MAX_TOKENS = 2500
WRITER_TEMPERATURE = 0.5
CHECK_BATCH = 4
CHECK_BATCH_CHARS = 12000
CHECK_MAX_TOKENS = 1500
SEARCH_HITS = 8
HIT_CHARS = 800
REVIEW_HITS = 5
REVIEW_HIT_CHARS = 800
# Cross-reference questions keep the batch size their cached reviews were made with. The other
# types use larger batches, because each review call counts against a low daily limit.
REVIEW_BATCH = 10
REVIEW_BATCH_CHARS = 30000
REVIEW_MAX_TOKENS = 8192
REVIEW_BATCH_OTHER = 20
REVIEW_BATCH_CHARS_OTHER = 50000
REVIEW_MAX_TOKENS_OTHER = 16384


def batches(entries: list[tuple[dict, str]], size: int, max_chars: int) -> list[list]:
    """Group (item, text) pairs in order, at most ``size`` items and ``max_chars`` per group."""
    groups: list[list] = []
    current: list = []
    length = 0
    for item, text in entries:
        if current and (len(current) >= size or length + len(text) > max_chars):
            groups.append(current)
            current, length = [], 0
        current.append((item, text))
        length += len(text)
    if current:
        groups.append(current)
    return groups


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def heading(h: dict) -> str:
    parts = [h["document"], h["chapter"], f"{h['section_label']} {h['section_title']}".strip()]
    return ", ".join(part for part in parts if part)


def task_block(task: dict) -> str:
    standard = p.STANDARD_NAMES[task["written_standard"]]
    lines = [
        f"Oppgave {task['task_id']}",
        p.TYPE_INSTRUCTIONS[task["question_type"]],
        f"Målform: {standard}.",
    ]
    if task["question_type"] == "cross_reference":
        first, rest = task["parts"][0], task["parts"][1:]
        lines += [f"Tekst A: {heading(task)}", f"[1] {first['text']}"]
        lines.append(f"Tekst B: {heading(task['b_heading'])}")
        lines += [f"[{part['number']}] {part['text']}" for part in rest]
        lines.append("Måldelen er [1]. Ta med de delene av tekst B som svaret bygger på.")
    else:
        lines.append(f"Kilde: {heading(task)}")
        target = next(part["number"] for part in task["parts"] if part["target"])
        for part in task["parts"]:
            mark = " (måldel)" if part["target"] else ""
            lines.append(f"[{part['number']}]{mark} {part['text']}")
        lines.append(
            f"Lag spørsmålet ut fra måldelen [{target}]. Andre deler kan brukes hvis de trengs "
            "for å forstå den."
        )
    return "\n".join(lines)


def _record(result) -> dict:
    return {"model": result.model, "model_version": result.model_version}


def write_tasks(client: ModelClient, tasks: list[dict]) -> tuple[list[dict], bool]:
    """Write questions for the tasks. Returns the items and whether the quota stopped the run.

    Batches are made inside each question type, then taken in turns across the types, so that a
    run stopped by the quota has made progress on every type. Taking turns does not change what
    is in a batch, so cached answers stay valid.
    """
    groups: dict[str, list] = {}
    for task in tasks:
        groups.setdefault(task["task_id"].rsplit("-", 1)[0], []).append(task)
    queues = []
    for name in sorted(groups):
        members = groups[name]
        if members[0]["question_type"] == "unanswerable":
            queues.append([[(task, "")] for task in members])
        else:
            entries = [(task, task_block(task)) for task in members]
            queues.append(batches(entries, WRITER_BATCH, WRITER_BATCH_CHARS))
    planned = []
    for position in range(max((len(queue) for queue in queues), default=0)):
        planned += [queue[position] for queue in queues if position < len(queue)]
    items: list[dict] = []
    stopped = False
    for batch in planned:
        try:
            if stopped:
                with client.cache_only():
                    items += _write_one(client, batch)
            else:
                items += _write_one(client, batch)
        except QuotaExhausted as exc:
            log.warning("writing stopped: %s", exc)
            stopped = True
            items += [_pending(task, "write") for task, _ in batch]
        except CacheMiss:
            items += [_pending(task, "write") for task, _ in batch]
    return sorted(items, key=lambda item: _item_order(item, tasks)), stopped


def _write_one(client: ModelClient, batch: list) -> list[dict]:
    if batch[0][0]["question_type"] == "unanswerable":
        return _write_unanswerable(client, batch[0][0])
    return _write_batch(client, batch)


def _item_order(item: dict, tasks: list[dict]) -> tuple:
    """A fixed order for items: answerable ones by type and number, unanswerable ones last in
    task order. Check and review batches follow this order, so their cached answers stay valid
    however the writing was scheduled."""
    positions = {task["task_id"]: index for index, task in enumerate(tasks)}
    task_id = item["task_id"]
    if item["question_type"] == "unanswerable":
        suffix = item["item_id"][len(task_id) + 1 :] if item["item_id"] != task_id else "0"
        return (1, "", positions[task_id], int(suffix))
    return (0, task_id, 0, 0)


def _base_item(task: dict, item_id: str) -> dict:
    keep = (
        "question_type",
        "written_standard",
        "doc_key",
        "section_id",
        "target_provision_id",
        "kind",
        "unanswerable_in",
    )
    item = {"item_id": item_id, "task_id": task["task_id"]}
    item.update({key: task[key] for key in keep if key in task})
    return item


def _pending(task: dict, stage: str) -> dict:
    item = _base_item(task, task["task_id"])
    item["pending"] = stage
    return item


def _failed(task: dict, reason: str) -> dict:
    item = _base_item(task, task["task_id"])
    item["removed_by"] = "format"
    item["removed_reason"] = reason
    return item


def _write_batch(client: ModelClient, batch: list) -> list[dict]:
    content = "\n\n".join(text for _, text in batch) + "\n\n" + p.WRITER_FORMAT
    messages = [
        {"role": "system", "content": p.WRITER_SYSTEM},
        {"role": "user", "content": content},
    ]
    cross = batch[0][0]["question_type"] == "cross_reference"
    version = p.CROSS_WRITER_VERSION if cross else p.WRITER_VERSION
    try:
        result = client.chat(
            "writer",
            messages,
            version,
            WRITER_MAX_TOKENS,
            temperature=WRITER_TEMPERATURE,
            json_mode=True,
        )
    except ProviderError as exc:
        return [_failed(task, f"provider error: {exc}"[:200]) for task, _ in batch]
    data = parse_json(result.text) or {}
    answers = {
        str(entry.get("id")): entry for entry in data.get("oppgaver", []) if isinstance(entry, dict)
    }
    items = []
    for task, _ in batch:
        entry = answers.get(task["task_id"])
        if entry is None:
            reason = "answer cut off" if result.finish_reason == "length" else "no answer"
            items.append(_failed(task, reason))
            continue
        item = _base_item(task, task["task_id"])
        item.update(_record(result))
        item["prompt_version"] = version
        if cross:
            item["why_both"] = str(entry.get("hvorfor_begge") or "")
        question, answer = entry.get("sporsmal"), entry.get("svar")
        numbers = {part["number"]: part for part in task["parts"]}
        used = [n for n in entry.get("deler") or [] if isinstance(n, int) and n in numbers]
        target = [part["number"] for part in task["parts"] if part["target"]]
        used = sorted(set(used) | set(target))
        if cross:
            # When the writer names no part of text B, all of text B is gold.
            b_parts = [n for n, part in numbers.items() if part.get("side") == "B"]
            if not set(used) & set(b_parts):
                used = sorted(set(used) | set(b_parts))
                item["all_of_b_used"] = True
        if not isinstance(question, str) or not question.strip():
            item.update(removed_by="format", removed_reason="no question")
        elif not isinstance(answer, str) or not answer.strip():
            item.update(removed_by="format", removed_reason="no answer text")
        item["question"] = (question or "").strip()
        item["reference_answer"] = (answer or "").strip()
        item["gold_provision_ids"] = [numbers[n]["provision_id"] for n in used]
        item["gold_section_ids"] = sorted({numbers[n]["section_id"] for n in used})
        item["gold_texts"] = [numbers[n]["text"] for n in used]
        item["gold_headings"] = [
            heading(task["b_heading"]) if numbers[n].get("side") == "B" else heading(task)
            for n in used
        ]
        items.append(item)
    return items


def _write_unanswerable(client: ModelClient, task: dict) -> list[dict]:
    standard = p.STANDARD_NAMES[task["written_standard"]]
    topics = "\n".join(f"{n}. {topic}" for n, topic in enumerate(task["topics"], start=1))
    content = (
        f"Målform: {standard}.\nSpørsmålene skal handle om {p.UNANSWERABLE_KINDS[task['kind']]}"
        f"\nLag ett spørsmål for hvert tema:\n{topics}\n\n{p.UNANSWERABLE_FORMAT}"
    )
    messages = [
        {"role": "system", "content": p.UNANSWERABLE_SYSTEM},
        {"role": "user", "content": content},
    ]
    try:
        result = client.chat(
            "writer",
            messages,
            p.UNANSWERABLE_VERSION,
            WRITER_MAX_TOKENS,
            temperature=WRITER_TEMPERATURE,
            json_mode=True,
        )
    except ProviderError as exc:
        return [_failed(task, f"provider error: {exc}"[:200])]
    data = parse_json(result.text) or {}
    entries = [e for e in data.get("sporsmal", []) if isinstance(e, dict)]
    if not entries:
        return [_failed(task, "no answer")]
    items = []
    for number, (topic, entry) in enumerate(zip(task["topics"], entries, strict=False), start=1):
        item = _base_item(task, f"{task['task_id']}-{number}")
        item.update(_record(result))
        item["prompt_version"] = p.UNANSWERABLE_VERSION
        item["topic"] = topic
        item["question"] = str(entry.get("sporsmal") or "").strip()
        item["reference_answer"] = None
        item["gold_provision_ids"] = []
        item["gold_section_ids"] = []
        if not item["question"]:
            item.update(removed_by="format", removed_reason="no question")
        items.append(item)
    return items


def check_block(item: dict) -> str:
    lines = [
        f"Spørsmål {item['item_id']}: {item['question']}",
        f"Referansesvar: {item['reference_answer']}",
    ]
    cross = item["question_type"] == "cross_reference"
    for number, (title, text) in enumerate(
        zip(item["gold_headings"], item["gold_texts"], strict=True)
    ):
        label = ("Tekst A" if number == 0 else "Tekst B") if cross else "Kildetekst"
        lines.append(f"{label} ({title}): {text}")
    return "\n".join(lines)


def search_block(item: dict) -> str:
    lines = [f"Spørsmål {item['item_id']}: {item['question']}", "Tekster fra søket:"]
    for hit in item["search_hits"]:
        lines.append(f"- {hit['heading']}: {hit['text'][:HIT_CHARS]}")
    return "\n".join(lines)


def run_checks(
    client: ModelClient,
    items: list[dict],
    role: str,
    system: str,
    version: str,
    block,
    outcome,
    open_groups: frozenset[str] = frozenset(),
) -> bool:
    """Send items in batches to a checking model and store each verdict with ``outcome``.

    Batches are made inside each question type, so that new questions of one type never move the
    batches of another and cached answers stay valid. The last batch of a type that is still
    being written waits, since it would change when more questions arrive.

    Items the model leaves out of its answer get one more try in smaller batches. Answers are
    cached, so a second run would see the same gap: items still without a verdict are removed.
    """
    entries = [(item, block(item)) for item in items]
    planned, held = _check_batches(entries, role, 1, open_groups)
    for item in held:
        item["pending"] = role
    missing, stopped = _check_pass(client, planned, role, system, version, outcome)
    if missing and not stopped:
        entries = [(item, block(item)) for item in missing]
        planned, _ = _check_batches(entries, role, 2, frozenset())
        missing, stopped = _check_pass(client, planned, role, system, version, outcome)
    for item in missing:
        if "pending" not in item:
            item.update(removed_by=f"{role}_format", removed_reason=item.pop("missing_reason"))
    return stopped


def group_of(item: dict) -> str:
    """The question type and written standard, as in ``direct-nb``."""
    return item["task_id"].rsplit("-", 1)[0]


def _batch_limits(role: str, group: str, divisor: int) -> tuple[int, int]:
    if role == "checker":
        return max(1, CHECK_BATCH // divisor), CHECK_BATCH_CHARS
    if group.startswith("cross_reference"):
        return max(1, REVIEW_BATCH // divisor), REVIEW_BATCH_CHARS
    return max(1, REVIEW_BATCH_OTHER // divisor), REVIEW_BATCH_CHARS_OTHER


def _check_batches(
    entries: list, role: str, divisor: int, open_groups: frozenset[str]
) -> tuple[list[list], list[dict]]:
    """Batches taken in turns across the types, and the items that have to wait."""
    grouped: dict[str, list] = {}
    for entry in entries:
        grouped.setdefault(group_of(entry[0]), []).append(entry)
    queues, held = [], []
    for name in sorted(grouped):
        size, max_chars = _batch_limits(role, name, divisor)
        queue = batches(grouped[name], size, max_chars)
        if name in open_groups and queue:
            held += [item for item, _ in queue.pop()]
        queues.append(queue)
    planned = []
    for position in range(max((len(queue) for queue in queues), default=0)):
        planned += [queue[position] for queue in queues if position < len(queue)]
    if role == "reviewer":
        # Review calls are few. Full batches go first, so a short day still reviews the most items.
        planned.sort(key=len, reverse=True)
    return planned, held


def _check_pass(
    client: ModelClient,
    planned: list[list],
    role: str,
    system: str,
    version: str,
    outcome,
) -> tuple[list[dict], bool]:
    stopped = False
    missing: list[dict] = []
    for batch in planned:
        ids = ", ".join(item["item_id"] for item, _ in batch)
        content = (
            f"Vurder alle {len(batch)} elementene. Svaret skal ha én vurdering for hver av disse "
            f"id-ene: {ids}.\n\n" + "\n\n".join(text for _, text in batch)
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        if role == "checker":
            max_tokens = CHECK_MAX_TOKENS
        elif batch[0][0]["question_type"] == "cross_reference":
            max_tokens = REVIEW_MAX_TOKENS
        else:
            max_tokens = REVIEW_MAX_TOKENS_OTHER
        try:
            if stopped:
                with client.cache_only():
                    result = client.chat(role, messages, version, max_tokens, json_mode=True)
            else:
                result = client.chat(role, messages, version, max_tokens, json_mode=True)
        except CacheMiss:
            for item, _ in batch:
                item["pending"] = role
            continue
        except QuotaExhausted as exc:
            log.warning("%s stopped: %s", role, exc)
            stopped = True
            for item, _ in batch:
                item["pending"] = role
            continue
        except ProviderError as exc:
            for item, _ in batch:
                item["pending"] = role
                item[f"{role}_error"] = str(exc)[:200]
            continue
        data = parse_json(result.text) or {}
        verdicts = {str(v.get("id")): v for v in data.get("vurderinger", []) if isinstance(v, dict)}
        for item, _ in batch:
            verdict = verdicts.get(item["item_id"])
            if verdict is None:
                reason = "answer cut off" if result.finish_reason == "length" else "no verdict"
                item["missing_reason"] = reason
                missing.append(item)
                continue
            item.pop("missing_reason", None)
            outcome(item, verdict, result)
    return missing, stopped
