"""Figures for the answers, read from ``results/generation/<split>.json``."""

from __future__ import annotations

from pathlib import Path

from norsk_lov_rag.figures.charts import dot_rows, legend_below
from norsk_lov_rag.figures.style import (
    MARKERS,
    PALETTE,
    QUESTION_TYPES,
    STANDARDS,
    figure,
    run_label,
    save,
    t,
)

GENERATOR_ORDER = ["gemma", "normistral"]
SYSTEM_ORDER = ["bm25", "hybrid+rerank", "hybrid"]


def _runs(report: dict) -> list[str]:
    def key(run: str):
        parts = run.split("__")
        generator = parts[-1] if len(parts) == 3 else run
        system = parts[1] if len(parts) == 3 else run
        return (
            GENERATOR_ORDER.index(generator) if generator in GENERATOR_ORDER else 99,
            SYSTEM_ORDER.index(system) if system in SYSTEM_ORDER else 99,
            run,
        )

    return sorted(report["runs"], key=key)


def correctness(report: dict, out: Path, lang: str) -> list[Path]:
    runs = _runs(report)
    kinds = ["direct", "everyday", "cross_reference"]
    series = [(t("Alle besvarbare", "All answerable", lang),
               [report["runs"][r]["correctness"] for r in runs])]  # fmt: skip
    for kind in kinds:
        series.append(
            (QUESTION_TYPES[kind][0 if lang == "nb" else 1],
             [report["runs"][r]["by_type"].get(kind, {}).get("correctness") for r in runs])
        )  # fmt: skip
    fig, ax = figure(7.5, 0.75 * len(runs) + 1.6)
    dot_rows(ax, [run_label(r, lang) for r in runs], series,
             t("Riktighet (riktig 1, delvis 0,5)", "Correctness (correct 1, partly 0.5)", lang),
             full_range=True)  # fmt: skip
    fig.suptitle(t("Svar: riktighet etter spørsmålstype", "Answers: correctness by question type",
                   lang), x=0.01, ha="left", fontweight="bold")  # fmt: skip
    legend_below(fig, [ax])
    fig.tight_layout()
    return save(fig, out, "svar_riktighet")


def citations(report: dict, out: Path, lang: str) -> list[Path]:
    runs = _runs(report)
    fields = [
        ("citation_precision", t("Presisjon i kildehenvisninger", "Citation precision", lang)),
        ("citation_recall", t("Dekning av gullbestemmelser", "Citation recall", lang)),
        ("support", t("Støttet av lovteksten", "Supported by the law text", lang)),
    ]
    series = [(label, [report["runs"][r][field] for r in runs]) for field, label in fields]
    fig, ax = figure(7.5, 0.7 * len(runs) + 1.6)
    dot_rows(ax, [run_label(r, lang) for r in runs], series, t("Andel", "Share", lang),
             full_range=True)  # fmt: skip
    fig.suptitle(t("Svar: kildehenvisninger og støtte", "Answers: citations and support", lang),
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    legend_below(fig, [ax], ncol=3)
    fig.tight_layout()
    return save(fig, out, "svar_kilder")


def abstention(report: dict, out: Path, lang: str) -> list[Path]:
    runs = _runs(report)
    series = [
        (t("Riktig avslag (ubesvarbare spørsmål)", "Correct refusal (unanswerable)", lang),
         [report["runs"][r]["correct_refusals"] for r in runs]),
        (t("Feil avslag (besvarbare spørsmål)", "False refusal (answerable)", lang),
         [report["runs"][r]["false_refusals"] for r in runs]),
    ]  # fmt: skip
    fig, ax = figure(7.5, 0.6 * len(runs) + 1.5)
    dot_rows(ax, [run_label(r, lang) for r in runs], series, t("Andel", "Share", lang),
             full_range=True)  # fmt: skip
    fig.suptitle(t("Svar: avslag", "Answers: refusals", lang), x=0.01, ha="left",
                 fontweight="bold")  # fmt: skip
    legend_below(fig, [ax], ncol=2)
    fig.tight_layout()
    return save(fig, out, "svar_avslag")


def cost_quality(report: dict, out: Path, lang: str) -> list[Path]:
    runs = _runs(report)
    fig, axes = figure(9.5, 3.6, ncols=2, sharey=True)
    xs = [
        (lambda s: s["latency_median"], t("Svartid, median (sekunder)", "Latency, median (s)",
                                          lang)),
        (lambda s: (s["prompt_tokens"] or 0) + (s["completion_tokens"] or 0),
         t("Tokens per spørsmål, snitt", "Tokens per question, mean", lang)),
    ]  # fmt: skip
    for ax, (get, label) in zip(axes, xs, strict=True):
        for i, run in enumerate(runs):
            summary = report["runs"][run]
            score = summary["correctness"]
            x = get(summary)
            if x is None or score.get("mean") is None or score.get("low") is None:
                continue
            ax.errorbar(
                [x], [score["mean"]],
                yerr=[[score["mean"] - score["low"]], [score["high"] - score["mean"]]],
                fmt=MARKERS[i % len(MARKERS)], color=PALETTE[i % len(PALETTE)], ms=7,
                elinewidth=1.2, capsize=0, markeredgecolor="white", markeredgewidth=0.6,
                label=run_label(run, lang),
            )  # fmt: skip
        ax.set_xlabel(label)
        ax.set_xlim(left=0)
        ax.margins(x=0.15)
    axes[0].set_ylabel(t("Riktighet", "Correctness", lang))
    axes[0].set_ylim(0, 1.02)
    fig.suptitle(t("Svar: kvalitet mot tid og tokens", "Answers: quality against time and tokens",
                   lang), x=0.01, ha="left", fontweight="bold")  # fmt: skip
    legend_below(fig, axes, ncol=2)
    fig.tight_layout()
    return save(fig, out, "svar_tid_tokens")


def standard(report: dict, out: Path, lang: str) -> list[Path]:
    runs = _runs(report)
    series = []
    for key in ("nb", "nn"):
        groups = [report["runs"][r]["by_standard"].get(key, {}) for r in runs]
        n = groups[0].get("n") if groups else None
        series.append(
            (f"{t('Riktighet', 'Correctness', lang)}, {STANDARDS[key][0 if lang == 'nb' else 1]}"
             f" (n={n})",
             [g.get("correctness") for g in groups])
        )  # fmt: skip
    series.append((t("Samme skriftform som spørsmålet", "Same written standard as question",
                     lang), [report["runs"][r]["language_match"] for r in runs]))  # fmt: skip
    fig, ax = figure(7.5, 0.7 * len(runs) + 1.6)
    dot_rows(ax, [run_label(r, lang) for r in runs], series, t("Andel", "Share", lang),
             full_range=True)  # fmt: skip
    fig.suptitle(t("Svar: bokmål og nynorsk", "Answers: Bokmål and Nynorsk", lang),
                 x=0.01, ha="left", fontweight="bold")  # fmt: skip
    legend_below(fig, [ax], ncol=3)
    fig.tight_layout()
    return save(fig, out, "svar_skriftform")


FIGURES = [correctness, citations, abstention, cost_quality, standard]
