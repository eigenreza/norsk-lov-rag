"""Figures for the retrieval comparison, read from ``results/retrieval/<split>.json``."""

from __future__ import annotations

from pathlib import Path

from norsk_lov_rag.figures.charts import dot_rows, legend_below
from norsk_lov_rag.figures.style import (
    QUESTION_TYPES,
    SETTINGS,
    STANDARDS,
    SYSTEMS,
    figure,
    pick,
    save,
    t,
)

# Systems in the order they are shown. Systems missing from a setting are skipped.
ORDER = list(SYSTEMS)
MAIN = ["bm25", "bm25-compound", "dense:e5-base", "dense:bge-m3", "hybrid", "hybrid+rerank"]
DENSE = {
    "dense:e5-base": "multilingual",
    "dense:bge-m3": "multilingual",
    "dense:nb-sbert-v2-base": "norwegian",
    "dense:nb-sbert-v2-large": "norwegian",
}
TYPES = ["direct", "everyday", "cross_reference"]


def _systems(data: dict, setting: str, wanted: list[str] | None = None) -> list[str]:
    present = data["settings"].get(setting, {}).get("systems", {})
    return [s for s in ORDER if s in present and (wanted is None or s in wanted)]


def _metric(data: dict, setting: str, system: str, metric: str, group=None, key=None):
    scores = data["settings"][setting]["systems"][system]
    if group is not None:
        scores = scores.get(group, {}).get(key)
        if scores is None:
            return None
    else:
        scores = scores["all"]
    return scores.get(metric)


def _n(data: dict, setting: str, system: str, group: str, key: str) -> int | None:
    entry = data["settings"][setting]["systems"][system].get(group, {}).get(key)
    return entry.get("n") if entry else None


def recall_at_k(data: dict, out: Path, lang: str) -> list[Path]:
    panels = [s for s in ("tax-structure", "full-structure") if s in data["settings"]]
    rows_per = [_systems(data, s, [x for x in ORDER if "expansion" not in x]) for s in panels]
    fig, axes = figure(9.5, 0.42 * max(len(r) for r in rows_per) + 1.4, ncols=len(panels),
                       squeeze=False)  # fmt: skip
    axes = axes[0]
    for ax, setting, systems in zip(axes, panels, rows_per, strict=True):
        series = [
            (f"recall@{k}", [_metric(data, setting, s, f"recall@{k}") for s in systems])
            for k in (1, 5, 10)
        ]
        xlabel = t("Andel gullbestemmelser funnet", "Share of gold provisions found", lang)
        dot_rows(ax, [pick(SYSTEMS, s, lang) for s in systems], series, xlabel)
        ax.set_title(pick(SETTINGS, setting, lang))
    fig.suptitle(
        t("Søk: recall@k med 95 % intervall", "Retrieval: recall@k with 95% interval", lang),
        x=0.01, ha="left", fontweight="bold",
    )  # fmt: skip
    legend_below(fig, axes)
    fig.tight_layout()
    return save(fig, out, "sok_recall_at_k")


def by_type(data: dict, out: Path, lang: str) -> list[Path]:
    panels = [s for s in ("tax-structure", "full-structure") if s in data["settings"]]
    rows_per = [_systems(data, s, MAIN) for s in panels]
    fig, axes = figure(9.5, 0.5 * max(len(r) for r in rows_per) + 1.4, ncols=len(panels),
                       squeeze=False)  # fmt: skip
    axes = axes[0]
    for ax, setting, systems in zip(axes, panels, rows_per, strict=True):
        first = systems[0]
        series = []
        for kind in TYPES:
            n = _n(data, setting, first, "by_type", kind)
            label = f"{pick(QUESTION_TYPES, kind, lang)} (n={n})"
            series.append(
                (label, [_metric(data, setting, s, "recall@10", "by_type", kind) for s in systems])
            )
        dot_rows(ax, [pick(SYSTEMS, s, lang) for s in systems], series, "recall@10")
        ax.set_title(pick(SETTINGS, setting, lang))
    fig.suptitle(t("Søk etter spørsmålstype", "Retrieval by question type", lang),
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    legend_below(fig, axes)
    fig.tight_layout()
    return save(fig, out, "sok_spoersmaalstype")


def embeddings(data: dict, out: Path, lang: str) -> list[Path]:
    setting = "tax-structure"
    systems = [s for s in _systems(data, setting) if s in DENSE]
    systems.sort(key=lambda s: (DENSE[s] != "multilingual", ORDER.index(s)))
    family = {"multilingual": t("flerspråklig", "multilingual", lang),
              "norwegian": t("norsk", "Norwegian", lang)}  # fmt: skip
    rows = [f"{pick(SYSTEMS, s, lang)} ({family[DENSE[s]]})" for s in systems]
    series = [
        (metric.replace("ndcg", "nDCG"), [_metric(data, setting, s, metric) for s in systems])
        for metric in ("recall@1", "recall@10", "ndcg@10")
    ]
    fig, ax = figure(6.5, 0.55 * len(rows) + 1.5)
    dot_rows(ax, rows, series, t("Verdi", "Value", lang))
    split = sum(DENSE[s] == "multilingual" for s in systems)
    if 0 < split < len(systems):
        ax.axhline(split - 0.5, color="#BBBBBB", linewidth=0.8, linestyle="--")
    ax.set_title(pick(SETTINGS, setting, lang), fontweight="normal", fontsize=9)
    fig.suptitle(t("Norske og flerspråklige embeddinger", "Norwegian and multilingual embeddings",
                   lang), x=0.01, ha="left", fontweight="bold")  # fmt: skip
    legend_below(fig, [ax])
    fig.tight_layout()
    return save(fig, out, "sok_embeddinger")


def chunking(data: dict, out: Path, lang: str) -> list[Path]:
    pairs = [("tax-structure", "tax-naive"), ("full-structure", "full-naive")]
    pairs = [p for p in pairs if p[0] in data["settings"] and p[1] in data["settings"]]
    rows_per = [[s for s in _systems(data, a) if s in _systems(data, b)] for a, b in pairs]
    heights = [max(len(r), 1) for r in rows_per]
    fig, axes = figure(6.8, 0.42 * sum(heights) + 1.8, ncols=1, nrows=len(pairs),
                       squeeze=False, gridspec_kw={"height_ratios": heights})  # fmt: skip
    axes = axes[:, 0]
    for ax, (structure, naive), systems in zip(axes, pairs, rows_per, strict=True):
        corpus = t("Skattekorpus", "Tax corpus", lang) if structure.startswith("tax") else t(
            "Hele korpuset", "Full corpus", lang)  # fmt: skip
        series = [
            (t("Strukturbiter", "Structure chunks", lang),
             [_metric(data, structure, s, "ndcg@10") for s in systems]),
            (t("Vinduer på 200 ord", "200-word windows", lang),
             [_metric(data, naive, s, "ndcg@10") for s in systems]),
        ]  # fmt: skip
        dot_rows(ax, [pick(SYSTEMS, s, lang) for s in systems], series, "nDCG@10")
        ax.set_title(corpus)
    fig.suptitle(t("Strukturbiter mot faste vinduer", "Structure chunks against fixed windows",
                   lang), x=0.01, ha="left", fontweight="bold")  # fmt: skip
    legend_below(fig, axes)
    fig.tight_layout()
    return save(fig, out, "sok_tekstbiter")


def standard(data: dict, out: Path, lang: str) -> list[Path]:
    setting = "tax-structure"
    systems = _systems(data, setting, MAIN)
    series = []
    for key in ("nb", "nn"):
        n = _n(data, setting, systems[0], "by_standard", key)
        series.append(
            (f"{pick(STANDARDS, key, lang)} (n={n})",
             [_metric(data, setting, s, "recall@10", "by_standard", key) for s in systems])
        )  # fmt: skip
    fig, ax = figure(6.5, 0.5 * len(systems) + 1.5)
    dot_rows(ax, [pick(SYSTEMS, s, lang) for s in systems], series, "recall@10")
    ax.set_title(pick(SETTINGS, setting, lang), fontweight="normal", fontsize=9)
    fig.suptitle(t("Søk: bokmål og nynorsk", "Retrieval: Bokmål and Nynorsk", lang),
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    legend_below(fig, [ax])
    fig.tight_layout()
    return save(fig, out, "sok_skriftform")


def expansion(data: dict, out: Path, lang: str) -> list[Path]:
    panels = [s for s in ("tax-structure", "full-structure") if s in data["settings"]]
    wanted = ["hybrid", "hybrid+expansion", "hybrid+rerank", "hybrid+rerank+expansion"]
    fig, axes = figure(9.5, 3.4, ncols=len(panels), squeeze=False)
    axes = axes[0]
    for ax, setting in zip(axes, panels, strict=True):
        systems = _systems(data, setting, wanted)
        series = [
            (pick(SYSTEMS, s, lang),
             [_metric(data, setting, s, "recall@10", "by_type", kind) for kind in TYPES])
            for s in systems
        ]  # fmt: skip
        dot_rows(ax, [pick(QUESTION_TYPES, k, lang) for k in TYPES], series, "recall@10")
        ax.set_title(pick(SETTINGS, setting, lang))
    fig.suptitle(t("Utvidelse med henviste bestemmelser", "Expansion with referenced provisions",
                   lang), x=0.01, ha="left", fontweight="bold")  # fmt: skip
    legend_below(fig, axes, ncol=2)
    fig.tight_layout()
    return save(fig, out, "sok_utvidelse")


FIGURES = [recall_at_k, by_type, embeddings, chunking, standard, expansion]
