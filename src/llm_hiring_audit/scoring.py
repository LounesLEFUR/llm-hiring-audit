"""Running the audit: send every (CV, prompt, repetition) to a model.

Design choices:
- Queries are shuffled, so a drift of the model or API over a long run
  cannot line up with one demographic group.
- Results are appended line by line and a run can be resumed: a crash
  after 3 hours of scoring loses nothing.
- Unparseable answers are kept with `score=None` instead of being
  silently dropped, so the refusal rate can itself be analysed.
"""

import json
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pydantic import BaseModel

from llm_hiring_audit.backends import LLMBackend
from llm_hiring_audit.generator import load_job_postings
from llm_hiring_audit.prompts import PROMPTS

_NUMBER = re.compile(r"\d{1,3}")


def parse_score(answer: str) -> int | None:
    """Return the first integer in [1, 100] found in the answer, if any."""
    for match in _NUMBER.finditer(answer):
        value = int(match.group())
        if 1 <= value <= 100:
            return value
    return None


class ScoreRecord(BaseModel):
    cv_id: str
    profile_id: str
    job_id: str
    gender: str
    origin: str
    address_area: str
    name_index: int
    model: str
    prompt_id: str
    repetition: int
    temperature: float
    raw_answer: str
    score: int | None

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.cv_id, self.model, self.prompt_id, self.repetition)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Scorer:
    def __init__(
        self,
        backend: LLMBackend,
        prompt_ids: Iterable[str] = tuple(PROMPTS),
        repetitions: int = 3,
        temperature: float = 0.7,
        seed: int = 0,
        max_retries: int = 5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.backend = backend
        self.prompt_ids = list(prompt_ids)
        unknown = set(self.prompt_ids) - PROMPTS.keys()
        if unknown:
            raise ValueError(f"Unknown prompt ids: {sorted(unknown)}")
        self.repetitions = repetitions
        self.temperature = temperature
        self.seed = seed
        self.max_retries = max_retries
        self._sleep = sleep
        self._jobs = {job.job_id: job for job in load_job_postings()}

    def run(self, cvs: list[dict[str, Any]], out: Path) -> int:
        """Score all CVs, appending to `out`. Returns the number of new records."""
        done = {ScoreRecord(**r).key for r in read_jsonl(out)} if out.exists() else set()
        tasks = [
            (cv, prompt_id, rep)
            for cv in cvs
            for prompt_id in self.prompt_ids
            for rep in range(self.repetitions)
            if (cv["cv_id"], self.backend.name, prompt_id, rep) not in done
        ]
        order = np.random.default_rng(self.seed).permutation(len(tasks))

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for i in order:
                record = self._score_one(*tasks[i])
                f.write(record.model_dump_json() + "\n")
                f.flush()
        return len(tasks)

    def _score_one(self, cv: dict[str, Any], prompt_id: str, rep: int) -> ScoreRecord:
        prompt = PROMPTS[prompt_id].build(self._jobs[cv["job_id"]], cv["text"])
        answer = self._complete_with_retry(prompt, seed=self.seed * 1000 + rep)
        return ScoreRecord(
            cv_id=cv["cv_id"],
            profile_id=cv["profile_id"],
            job_id=cv["job_id"],
            gender=cv["gender"],
            origin=cv["origin"],
            address_area=cv["address_area"],
            name_index=cv["name_index"],
            model=self.backend.name,
            prompt_id=prompt_id,
            repetition=rep,
            temperature=self.temperature,
            raw_answer=answer,
            score=parse_score(answer),
        )

    def _complete_with_retry(self, prompt: str, seed: int) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                return self.backend.complete(prompt, temperature=self.temperature, seed=seed)
            except httpx.HTTPError:
                if attempt == self.max_retries:
                    raise
                self._sleep(2**attempt)  # exponential backoff, e.g. on rate limits
        raise AssertionError("unreachable")
