"""One look for every figure: a colour-blind safe palette, Norwegian labels and no dates in files.

Labels are Norwegian by default. ``lang="en"`` gives the same figure with English labels. The
names of systems, question types and written standards are looked up here, so that every
figure uses the same words.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

LANGS = ("nb", "en")

# Okabe-Ito colours, which stay apart for the common forms of colour blindness. They are used
# in this fixed order, so the same kind of series gets the same colour in every figure.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]
INK = "#222222"
MUTED = "#666666"
GRID = "#E3E3E3"

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.labelsize": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 100,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "savefig.facecolor": "white",
    # Fixed ids in SVG files, so that the same input gives the same file.
    "svg.hashsalt": "norsk-lov-rag",
    "svg.fonttype": "path",
}

SYSTEMS = {
    "bm25": ("BM25", "BM25"),
    "bm25-compound": ("BM25 med orddeling", "BM25 with compound splitting"),
    "dense:e5-base": ("e5-base", "e5-base"),
    "dense:bge-m3": ("bge-m3", "bge-m3"),
    "dense:nb-sbert-v2-base": ("nb-sbert-v2-base", "nb-sbert-v2-base"),
    "dense:nb-sbert-v2-large": ("nb-sbert-v2-large", "nb-sbert-v2-large"),
    "hybrid": ("Hybrid", "Hybrid"),
    "hybrid+rerank": ("Hybrid med omrangering", "Hybrid with reranking"),
    "hybrid+expansion": ("Hybrid med utvidelse", "Hybrid with expansion"),
    "hybrid+rerank+expansion": (
        "Hybrid med omrangering og utvidelse",
        "Hybrid with reranking and expansion",
    ),
}
GENERATORS = {
    "normistral": ("NorMistral 7B", "NorMistral 7B"),
    "gemma": ("Gemma 4 31B", "Gemma 4 31B"),
}
QUESTION_TYPES = {
    "direct": ("Direkte", "Direct"),
    "everyday": ("Hverdagsspråk", "Everyday language"),
    "cross_reference": ("Krysshenvisning", "Cross-reference"),
    "unanswerable": ("Ubesvarbar", "Unanswerable"),
}
STANDARDS = {"nb": ("Bokmål", "Bokmål"), "nn": ("Nynorsk", "Nynorsk")}
SETTINGS = {
    "tax-structure": ("Skattekorpus, strukturbiter", "Tax corpus, structure chunks"),
    "tax-naive": ("Skattekorpus, vinduer", "Tax corpus, windows"),
    "full-structure": ("Hele korpuset, strukturbiter", "Full corpus, structure chunks"),
    "full-naive": ("Hele korpuset, vinduer", "Full corpus, windows"),
}
CORPORA = {"tax": ("Skattekorpus", "Tax corpus"), "full": ("Hele korpuset", "Full corpus")}


def pick(table: dict[str, tuple[str, str]], key: str, lang: str) -> str:
    """The label for a key in one language, or the key itself when it has no label."""
    if key not in table:
        return key
    return table[key][LANGS.index(lang)]


def t(nb: str, en: str, lang: str) -> str:
    return nb if lang == "nb" else en


def run_label(run: str, lang: str) -> str:
    """A label such as 'Hybrid med omrangering + Gemma 4 31B' for a run name."""
    parts = run.split("__")
    if len(parts) != 3:
        return run
    _, system, generator = parts
    return f"{pick(SYSTEMS, system, lang)} + {pick(GENERATORS, generator, lang)}"


def figure(width: float, height: float, ncols: int = 1, nrows: int = 1, **kwargs):
    plt.rcParams.update(RC)
    return plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)


def percent_axis(ax, axis: str = "x", top: float = 1.0) -> None:
    from matplotlib.ticker import PercentFormatter

    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_formatter(PercentFormatter(1.0, decimals=0))
    (ax.set_xlim if axis == "x" else ax.set_ylim)(0, top)


def save(fig, out: Path, name: str) -> list[Path]:
    """Write SVG and PNG without dates or version strings in the metadata."""
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.svg", out / f"{name}.png"]
    fig.savefig(paths[0], metadata={"Date": None, "Creator": None})
    fig.savefig(paths[1], dpi=160, metadata={"Software": None})
    plt.close(fig)
    return paths
