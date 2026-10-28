"""Command-line interface: `llm-hiring-audit --help`."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from llm_hiring_audit.analysis import analyze as run_analysis
from llm_hiring_audit.analysis import prepare, subgroup_effects
from llm_hiring_audit.backends import LLMBackend, MistralBackend, MockBackend, OllamaBackend
from llm_hiring_audit.generator import CVGenerator
from llm_hiring_audit.power import Design, estimate_power
from llm_hiring_audit.prompts import PROMPTS
from llm_hiring_audit.render import cv_to_record
from llm_hiring_audit.scoring import Scorer, read_jsonl

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
            f.write(json.dumps(cv_to_record(cv), ensure_ascii=False) + "\n")
    typer.echo(f"Wrote {len(cvs)} CVs to {out}")


class BackendName(StrEnum):
    MOCK = "mock"
    OLLAMA = "ollama"
    MISTRAL = "mistral"


def _parse_penalties(items: list[str]) -> dict[str, float]:
    penalties = {}
    for item in items:
        word, sep, value = item.rpartition("=")
        if not sep or not word:
            raise typer.BadParameter(f"Expected WORD=VALUE, got {item!r}")
        penalties[word] = float(value)
    return penalties


def _make_backend(backend: BackendName, model: str, penalties: list[str]) -> LLMBackend:
    match backend:
        case BackendName.MOCK:
            return MockBackend(penalties=_parse_penalties(penalties))
        case BackendName.OLLAMA:
            return OllamaBackend(model)
        case BackendName.MISTRAL:
            return MistralBackend(model)


@app.command()
def score(
    backend: Annotated[BackendName, typer.Option(help="Which model provider to use.")],
    model: Annotated[str, typer.Option(help="Model name, e.g. mistral-small-latest.")] = "",
    cvs: Annotated[Path, typer.Option(help="CVs from `generate`.")] = Path("cvs.jsonl"),
    out: Annotated[Path, typer.Option(help="Output JSONL (appended).")] = Path("scores.jsonl"),
    prompt: Annotated[list[str] | None, typer.Option(help="Prompt id; repeatable.")] = None,
    repetitions: Annotated[int, typer.Option(min=1)] = 3,
    temperature: Annotated[float, typer.Option(min=0.0)] = 0.7,
    seed: int = 0,
    limit: Annotated[int | None, typer.Option(help="Only score the first N CVs.")] = None,
    mock_penalty: Annotated[
        list[str] | None, typer.Option(help="Mock only: WORD=POINTS, e.g. Sevran=-5.")
    ] = None,
) -> None:
    """Ask a model to score every CV. Re-running resumes where it stopped."""
    if backend is not BackendName.MOCK and not model:
        raise typer.BadParameter("--model is required for real backends")
    scorer = Scorer(
        _make_backend(backend, model, mock_penalty or []),
        prompt_ids=prompt or list(PROMPTS),
        repetitions=repetitions,
        temperature=temperature,
        seed=seed,
    )
    records = read_jsonl(cvs)[:limit]
    n = scorer.run(records, out)
    typer.echo(f"Scored {n} new queries into {out}")


@app.command()
def analyze(
    scores: Annotated[Path, typer.Option(help="Scores from `score`.")] = Path("scores.jsonl"),
    n_boot: Annotated[int, typer.Option(min=0, help="Bootstrap resamples (0 = skip).")] = 500,
    seed: int = 0,
    out_dir: Annotated[Path | None, typer.Option(help="Also write CSV tables here.")] = None,
) -> None:
    """Estimate discrimination effects for each model in the scores file."""
    data = pd.DataFrame(read_jsonl(scores))
    for report in run_analysis(data, n_boot=n_boot, seed=seed):
        dist = report.distribution
        typer.echo(f"\n=== {report.model} ===")
        typer.echo(
            f"{int(dist['n'])} valid scores, mean {dist['mean']:.1f}, sd {dist['std']:.1f}, "
            f"{dist['share_at_or_above_ceiling']:.0%} at 90 or above"
        )
        typer.echo("\nEffects in score points (95% CI; p-values Holm-corrected):")
        columns = [c for c in EFFECT_COLUMNS if c in report.effects]
        typer.echo(report.effects[columns].round(3).to_string(index=False))
        typer.echo("\nRefusal / unparseable rate:")
        typer.echo(report.refusals.round(3).to_string(index=False))
        typer.echo(
            f"\nNoise (inputs for `power`): residual sd {report.noise.residual_sd:.2f}, "
            f"name sd {report.noise.name_sd:.2f}"
        )
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            slug = report.model.replace("/", "_")
            report.effects.to_csv(out_dir / f"effects_{slug}.csv", index=False)
            report.refusals.to_csv(out_dir / f"refusals_{slug}.csv", index=False)


@app.command()
def robustness(
    scores: Annotated[Path, typer.Option(help="Scores from `score`.")] = Path("scores.jsonl"),
    out_dir: Annotated[Path | None, typer.Option(help="Also write CSV tables here.")] = None,
) -> None:
    """Estimate the effects separately for each prompt wording and each job."""
    data = pd.DataFrame(read_jsonl(scores))
    for model, raw in data.groupby("model"):
        table = subgroup_effects(prepare(raw))
        typer.echo(f"\n=== {model} ===")
        typer.echo("Effects by subgroup (Holm correction across all rows below):")
        typer.echo(table.round(3).to_string(index=False))
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            table.to_csv(out_dir / f"subgroups_{str(model).replace('/', '_')}.csv", index=False)


@app.command()
def power(
    residual_sd: Annotated[float, typer.Option(help="From `analyze` on a pilot run.")],
    name_sd: Annotated[float, typer.Option(help="From `analyze` on a pilot run.")] = 0.0,
    effect: Annotated[float, typer.Option(help="Effect to detect, in points.")] = 2.0,
    profiles_per_job: Annotated[
        list[int] | None, typer.Option(help="Sizes to compare; repeatable.")
    ] = None,
    names_per_cell: Annotated[int, typer.Option(min=1)] = 2,
    n_prompts: Annotated[int, typer.Option(min=1)] = 3,
    repetitions: Annotated[int, typer.Option(min=1)] = 1,
    n_sims: Annotated[int, typer.Option(min=1)] = 200,
    seed: int = 0,
) -> None:
    """Probability of detecting an effect of EFFECT points, for several audit sizes."""
    typer.echo(f"Power to detect a {effect:g}-point gap (Holm-corrected, alpha = 0.05):")
    typer.echo(f"{'profiles/job':>12} {'CVs':>6} {'queries':>8} {'power':>6}")
    for size in profiles_per_job or [2, 5, 10, 20]:
        design = Design(
            profiles_per_job=size,
            names_per_cell=names_per_cell,
            n_prompts=n_prompts,
            repetitions=repetitions,
        )
        p = estimate_power(design, effect, residual_sd, name_sd, n_sims=n_sims, seed=seed)
        typer.echo(f"{size:>12} {design.n_cvs:>6} {design.n_queries:>8} {p:>6.0%}")


EFFECT_COLUMNS = [
    "label",
    "estimate",
    "ci_low",
    "ci_high",
    "p_holm",
    "significant",
    "boot_ci_low",
    "boot_ci_high",
]
