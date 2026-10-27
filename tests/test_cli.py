import json
from pathlib import Path

from typer.testing import CliRunner

from llm_hiring_audit.cli import app


def test_generate_writes_jsonl(tmp_path: Path) -> None:
    out = tmp_path / "cvs.jsonl"
    result = CliRunner().invoke(
        app, ["generate", "--out", str(out), "--profiles-per-job", "1", "--seed", "0"]
    )
    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in out.read_text("utf-8").splitlines()]
    assert len(records) == 4 * 12 * 2
    assert {"cv_id", "gender", "origin", "address_area", "text"} <= records[0].keys()


def test_score_with_mock(tmp_path: Path) -> None:
    cvs, out = tmp_path / "cvs.jsonl", tmp_path / "scores.jsonl"
    runner = CliRunner()
    runner.invoke(app, ["generate", "--out", str(cvs), "--profiles-per-job", "1"])
    args = ["score", "--backend", "mock", "--cvs", str(cvs), "--out", str(out)]
    args += ["--limit", "5", "--repetitions", "1", "--mock-penalty", "Sevran=-10"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(out.read_text("utf-8").splitlines()) == 5 * 3


def test_analyze_prints_and_writes_tables(tmp_path: Path) -> None:
    cvs, scores = tmp_path / "cvs.jsonl", tmp_path / "scores.jsonl"
    runner = CliRunner()
    runner.invoke(app, ["generate", "--out", str(cvs), "--profiles-per-job", "1"])
    runner.invoke(app, ["score", "--backend", "mock", "--cvs", str(cvs), "--out", str(scores)])
    result = runner.invoke(
        app,
        ["analyze", "--scores", str(scores), "--n-boot", "5", "--out-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "=== mock ===" in result.output
    assert (tmp_path / "effects_mock.csv").exists()


def test_score_requires_model_for_real_backend() -> None:
    result = CliRunner().invoke(app, ["score", "--backend", "ollama"])
    assert result.exit_code != 0


def test_bad_mock_penalty(tmp_path: Path) -> None:
    cvs = tmp_path / "cvs.jsonl"
    CliRunner().invoke(app, ["generate", "--out", str(cvs), "--profiles-per-job", "1"])
    args = ["score", "--backend", "mock", "--cvs", str(cvs), "--mock-penalty", "oops"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
