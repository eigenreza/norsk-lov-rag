"""The command line starts and reports its version."""

from __future__ import annotations

from typer.testing import CliRunner

from norsk_lov_rag import __version__
from norsk_lov_rag.cli import app


def test_version_command_prints_the_package_version() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__
