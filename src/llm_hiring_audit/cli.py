"""Command-line interface: `llm-hiring-audit --help`."""

import json
from pathlib import Path
from typing import Annotated

import typer

from llm_hiring_audit.generator import CVGenerator
from llm_hiring_audit.render import render_cv

app = typer.Typer(help="Statistical audit of LLMs used to screen French CVs.")


@app.callback()
def main() -> None:
    """Statistical audit of LLMs used to screen French CVs."""


@app.command()
def generate(
    out: Annotated[Path, typer.Option(help="Output JSONL file.")] = Path("cvs.jsonl"),
    profiles_per_job: Annotated[int, typer.Option(min=1)] = 5,
    names_per_cell: Annotated[int, typer.Option(min=1)] = 2,
    seed: Annotated[int, typer.Option(help="Random seed (reproducibility).")] = 0,
) -> None:
    """Generate matched CVs and write one JSON record per line."""
    generator = CVGenerator(seed=seed, names_per_cell=names_per_cell)
    cvs = generator.generate(profiles_per_job)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for cv in cvs:
            record = {
                "cv_id": cv.cv_id,
                "profile_id": cv.profile.profile_id,
                "job_id": cv.profile.job_id,
                "gender": cv.treatment.gender,
                "origin": cv.treatment.origin,
                "address_area": cv.treatment.address_area,
                "name_index": cv.name_index,
                "text": render_cv(cv),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    typer.echo(f"Wrote {len(cvs)} CVs to {out}")
