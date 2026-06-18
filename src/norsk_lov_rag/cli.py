"""Command line interface. Each pipeline stage gets its own command as it is built."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from norsk_lov_rag import __version__
from norsk_lov_rag.config import load_settings

app = typer.Typer(help="Norwegian legal RAG, evaluated on tax law.", no_args_is_help=True)

ConfigOpt = Annotated[Path | None, typer.Option("--config", help="YAML settings file")]


@app.callback()
def main() -> None:
    """Norwegian legal RAG, evaluated on tax law."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def corpus(
    config: ConfigOpt = None,
    download: Annotated[
        bool, typer.Option(help="Download archives that are not on disk yet")
    ] = True,
    force: Annotated[bool, typer.Option(help="Download again even if the files exist")] = False,
) -> None:
    """Download Lovdata's open datasets and build the provision tables."""
    from norsk_lov_rag.corpus.build import build_corpus

    manifest = build_corpus(load_settings(config), download=download, force=force)
    summary = {
        "corpus_hash": manifest["corpus_hash"],
        "counts": manifest["counts"],
        "failed_checks": [c for c in manifest["checks"] if not c["passed"]],
    }
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed_checks"]:
        raise typer.Exit(code=1)


@app.command()
def references(config: ConfigOpt = None) -> None:
    """Find the references in every provision and build the reference graph."""
    from norsk_lov_rag.references.graph import build_graph

    settings = load_settings(config)
    summary = build_graph(
        settings.paths.processed, settings.corpus.tax_documents, settings.corpus.full_folders
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def testset(
    config: ConfigOpt = None,
    pilot: Annotated[
        int, typer.Option(help="Only the first N tasks of each type, written to a pilot folder")
    ] = 0,
    offline: Annotated[bool, typer.Option(help="Use only cached model answers")] = False,
    final: Annotated[
        bool, typer.Option(help="Leave out questions not written yet and finish the rest")
    ] = False,
    freeze: Annotated[
        Path | None,
        typer.Option(help="Copy the finished sets to this tracked folder; no model calls"),
    ] = None,
) -> None:
    """Write, filter, check, review and split the test questions, within the free limits."""
    from norsk_lov_rag.testset.pipeline import build_testset, freeze_testset

    settings = load_settings(config)
    if freeze is not None:
        corpus = json.loads((settings.paths.processed / "manifest.json").read_text("utf-8"))
        models = {role: f"{m.provider}:{m.model}" for role, m in settings.llm.models.items()}
        manifest = freeze_testset(
            settings.paths.data_dir / "testset", freeze, corpus["corpus_hash"], models
        )
        typer.echo(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    logging.getLogger("httpx").setLevel(logging.WARNING)
    summary = build_testset(settings, pilot=pilot or None, offline=offline, final=final)
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


ListOpt = Annotated[list[str] | None, typer.Option(help="Repeat to give several")]


@app.command()
def index(
    config: ConfigOpt = None,
    scope: ListOpt = None,
    strategy: ListOpt = None,
    encoder: ListOpt = None,
) -> None:
    """Cut the corpora into chunks and embed them with the dense models, on the CPU."""
    from norsk_lov_rag.chunking.build import build_chunks, chunk_path
    from norsk_lov_rag.corpus.select import load_corpus
    from norsk_lov_rag.retrieval.dense import build_dense_index, index_dir

    settings = load_settings(config)
    cfg = settings.retrieval
    summary = {}
    for one_scope in scope or cfg.scopes:
        corpus = None
        for one_strategy in strategy or cfg.strategies:
            path = chunk_path(settings.paths.data_dir, one_scope, one_strategy)
            if not path.exists():
                if corpus is None:
                    corpus = load_corpus(settings.paths.processed, settings.corpus, one_scope)
                path.parent.mkdir(parents=True, exist_ok=True)
                build_chunks(corpus, one_strategy).to_parquet(path, index=False)
            texts = pd_texts(path)
            names = encoder or (
                list(cfg.encoders) if one_scope == "tax" else cfg.full_corpus_encoders
            )
            folder = index_dir(settings.paths.data_dir, one_scope, one_strategy)
            for name in names:
                meta = build_dense_index(
                    cfg.encoders[name], name, texts, folder, cfg.model_dir, log=typer.echo
                )
                summary[f"{one_scope}-{one_strategy}/{name}"] = meta
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


def pd_texts(path: Path) -> list[str]:
    import pandas as pd

    return pd.read_parquet(path, columns=["text"])["text"].tolist()


@app.command("retrieve-eval")
def retrieve_eval(
    config: ConfigOpt = None,
    split: Annotated[str, typer.Option(help="dev or test")] = "dev",
    frozen: Annotated[Path, typer.Option(help="Folder with the frozen sets")] = Path("testsett"),
    results: Annotated[Path, typer.Option(help="Folder for the summaries")] = Path(
        "results/retrieval"
    ),
) -> None:
    """Score every retrieval system on one split, by question type and written standard."""
    from norsk_lov_rag.retrieval.evaluate import evaluate_retrieval

    settings = load_settings(config)
    report = evaluate_retrieval(settings, split, frozen, Path("runs/retrieval"))
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{split}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for setting, data in report["settings"].items():
        typer.echo(f"{setting}  hybrid={data['hybrid']}")
        for system, scores in data["systems"].items():
            a = scores["all"]
            typer.echo(
                f"  {system:32s} R@1 {a['recall@1']['mean']:.3f}  R@10 {a['recall@10']['mean']:.3f}"
                f"  MRR {a['mrr']['mean']:.3f}  nDCG@10 {a['ndcg@10']['mean']:.3f}"
            )
    for row in report["comparisons"]:
        d = row["ndcg@10"]
        typer.echo(
            f"{row['a']} - {row['b']} ({row['question_type']}, n={row['n']}): "
            f"nDCG@10 {d['mean']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]"
        )


def _sample_ids(frozen: Path, split: str, size: int) -> set[str]:
    """A fixed subset of a split: the questions with the lowest hash of their id."""
    import hashlib

    from norsk_lov_rag.retrieval.search import load_split

    ids = sorted(
        (q["item_id"] for q in load_split(frozen, split)),
        key=lambda i: hashlib.sha256(i.encode()).hexdigest(),
    )
    return set(ids[:size])


@app.command()
def generate(
    config: ConfigOpt = None,
    split: Annotated[str, typer.Option(help="dev or test")] = "dev",
    frozen: Annotated[Path, typer.Option(help="Folder with the frozen sets")] = Path("testsett"),
    runs: Annotated[Path, typer.Option(help="Folder for the answers")] = Path("runs/generation"),
    system: ListOpt = None,
    generator: ListOpt = None,
    sample: Annotated[int | None, typer.Option(help="Only this many questions")] = None,
) -> None:
    """Write answers from the retrieved text with each retrieval system and generator."""
    from norsk_lov_rag.generation.run import generate as run_generation

    settings = load_settings(config)
    ids = _sample_ids(frozen, split, sample) if sample else None
    written = run_generation(settings, split, frozen, runs, system, generator, ids)
    for name, count in written.items():
        typer.echo(f"{name}: {count} answers")


@app.command()
def judge(
    config: ConfigOpt = None,
    split: Annotated[str, typer.Option(help="dev or test")] = "dev",
    frozen: Annotated[Path, typer.Option(help="Folder with the frozen sets")] = Path("testsett"),
    runs: Annotated[Path, typer.Option(help="Folder with the answers")] = Path("runs/generation"),
    out: Annotated[Path, typer.Option(help="Folder for the verdicts")] = Path("runs/judge"),
) -> None:
    """Judge correctness and support of the answers, blind to the system."""
    from norsk_lov_rag.generation.judge import judge as run_judge

    summary = run_judge(load_settings(config), split, frozen, runs, out)
    typer.echo(json.dumps(summary))


@app.command()
def report(
    config: ConfigOpt = None,
    split: Annotated[str, typer.Option(help="dev or test")] = "dev",
    frozen: Annotated[Path, typer.Option(help="Folder with the frozen sets")] = Path("testsett"),
    runs: Annotated[Path, typer.Option(help="Folder with the answers")] = Path("runs/generation"),
    judged: Annotated[Path, typer.Option(help="Folder with the verdicts")] = Path("runs/judge"),
    results: Annotated[Path, typer.Option(help="Folder for the summary")] = Path(
        "results/generation"
    ),
) -> None:
    """Score the answers: correctness, citations, support, refusals, language, tokens, time."""
    from norsk_lov_rag.generation.report import report as build_report

    summary = build_report(load_settings(config), split, frozen, runs, judged)
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    def show(value: dict) -> str:
        return "-" if value["mean"] is None else f"{value['mean']:.3f}"

    for name, run in summary["runs"].items():
        typer.echo(
            f"{name:45s} correct {show(run['correctness'])}  P {show(run['citation_precision'])}"
            f"  R {show(run['citation_recall'])}  support {show(run['support'])}"
            f"  refuse ok {show(run['correct_refusals'])}"
            f"  refuse wrong {show(run['false_refusals'])}"
        )


@app.command()
def figures(
    config: ConfigOpt = None,
    lang: Annotated[str, typer.Option(help="nb, en or both")] = "both",
    out: Annotated[Path, typer.Option(help="Folder for the figures")] = Path("figures"),
    split: Annotated[str, typer.Option(help="dev or test")] = "test",
    results: Annotated[Path, typer.Option(help="Folder with the result files")] = Path("results"),
) -> None:
    """Draw every figure from the saved results, Norwegian in figures/ and English in en/."""
    from norsk_lov_rag.figures.build import build_figures, refresh_corpus_results

    if lang not in ("nb", "en", "both"):
        raise typer.BadParameter("lang must be nb, en or both")
    refresh_corpus_results(load_settings(config), results)
    langs = ["nb", "en"] if lang == "both" else [lang]
    written = build_figures(results, out, langs, split)
    typer.echo(f"{len(written)} files written to {out}")


@app.command("demo-export")
def demo_export(
    config: ConfigOpt = None,
    split: Annotated[str, typer.Option(help="Split the recorded answers come from")] = "test",
    frozen: Annotated[Path, typer.Option(help="Folder with the frozen sets")] = Path("testsett"),
    runs: Annotated[Path, typer.Option(help="Folder with the answers")] = Path("runs/generation"),
    judged: Annotated[Path, typer.Option(help="Folder with the verdicts")] = Path("runs/judge"),
    out: Annotated[Path, typer.Option(help="Folder for the demo files")] = Path("demo_data"),
) -> None:
    """Write the files the demo needs: chunks, section texts, examples and results."""
    from norsk_lov_rag.demo.export import export_demo

    summary = export_demo(load_settings(config), out, split, frozen, runs, judged)
    typer.echo(
        f"{out}: {summary['examples']} examples, results {summary['results']}, "
        f"{summary['figures']} figures, {summary['bytes'] / 1e6:.1f} MB"
    )
