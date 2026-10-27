import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from llm_hiring_audit.backends import MockBackend
from llm_hiring_audit.generator import CVGenerator
from llm_hiring_audit.render import cv_to_record
from llm_hiring_audit.scoring import Scorer, parse_score, read_jsonl


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("72", 72),
        ("**88**", 88),
        ("Note : 85/100", 85),
        ("Je donnerais 100.", 100),
        ("0 puis 64", 64),
        ("Sur une échelle de 1 à 100, je donne 72.", 72),
        ("De 1 (très improbable) à 100 : 40", 40),
        ("Entre 60 et 70", None),
        ("72 ou 80", None),
        ("Je ne peux pas évaluer ce CV.", None),
        ("250", None),
        ("250/100", None),
        ("", None),
    ],
)
def test_parse_score(answer: str, expected: int | None) -> None:
    assert parse_score(answer) == expected


def _cv_records(n_profiles: int = 1) -> list[dict[str, Any]]:
    return [cv_to_record(cv) for cv in CVGenerator(seed=0, names_per_cell=1).generate(n_profiles)]


def test_run_scores_every_query(tmp_path: Path) -> None:
    cvs = _cv_records()
    out = tmp_path / "scores.jsonl"
    n = Scorer(MockBackend(), repetitions=2).run(cvs, out)
    records = read_jsonl(out)
    assert n == len(records) == len(cvs) * 3 * 2
    assert all(r["score"] is not None for r in records)


def test_run_resumes_without_duplicates(tmp_path: Path) -> None:
    cvs = _cv_records()
    out = tmp_path / "scores.jsonl"
    Scorer(MockBackend(), repetitions=1).run(cvs[:10], out)
    n_new = Scorer(MockBackend(), repetitions=1).run(cvs, out)
    records = read_jsonl(out)
    assert n_new == (len(cvs) - 10) * 3
    keys = [(r["cv_id"], r["prompt_id"], r["repetition"]) for r in records]
    assert len(keys) == len(set(keys)) == len(cvs) * 3


def test_queries_are_shuffled(tmp_path: Path) -> None:
    cvs = _cv_records()
    out = tmp_path / "scores.jsonl"
    Scorer(MockBackend(), prompt_ids=["rating"], repetitions=1).run(cvs, out)
    order = [r["cv_id"] for r in read_jsonl(out)]
    assert sorted(order) == sorted(c["cv_id"] for c in cvs)
    assert order != [c["cv_id"] for c in cvs]


def test_unparseable_answers_are_kept(tmp_path: Path) -> None:
    class Refuser(MockBackend):
        def complete(self, prompt: str, *, temperature: float, seed: int) -> str:
            return "Je ne peux pas répondre."

    out = tmp_path / "scores.jsonl"
    Scorer(Refuser(), prompt_ids=["rating"], repetitions=1).run(_cv_records()[:2], out)
    records = read_jsonl(out)
    assert [r["score"] for r in records] == [None, None]
    assert records[0]["raw_answer"] == "Je ne peux pas répondre."


def test_retries_then_succeeds() -> None:
    class Flaky(MockBackend):
        calls = 0

        def complete(self, prompt: str, *, temperature: float, seed: int) -> str:
            Flaky.calls += 1
            if Flaky.calls < 3:
                raise httpx.ConnectError("down")
            return "60"

    waits: list[float] = []
    scorer = Scorer(Flaky(), sleep=waits.append)
    assert scorer._complete_with_retry("x", seed=0) == "60"
    assert waits == [1, 2]


def test_gives_up_after_max_retries() -> None:
    class Down(MockBackend):
        def complete(self, prompt: str, *, temperature: float, seed: int) -> str:
            raise httpx.ConnectError("down")

    scorer = Scorer(Down(), max_retries=2, sleep=lambda _: None)
    with pytest.raises(httpx.ConnectError):
        scorer._complete_with_retry("x", seed=0)


def test_unknown_prompt_id() -> None:
    with pytest.raises(ValueError, match="Unknown prompt"):
        Scorer(MockBackend(), prompt_ids=["nope"])


def test_records_are_valid_json(tmp_path: Path) -> None:
    out = tmp_path / "scores.jsonl"
    Scorer(MockBackend(), prompt_ids=["rating"], repetitions=1).run(_cv_records()[:1], out)
    record = json.loads(out.read_text("utf-8"))
    assert record["model"] == "mock"
