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
