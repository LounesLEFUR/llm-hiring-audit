"""Generation of matched CVs following a full factorial design.

For each base profile we create one CV per treatment cell
(gender x origin x address area) and per name draw. All CVs derived from
the same profile share every qualification, so any difference in how a
model scores them can only come from the name or the address.
"""

import itertools
import json
from functools import cache
from importlib import resources
from typing import Any

import numpy as np

from llm_hiring_audit.models import (
    CV,
    AddressArea,
    BaseProfile,
    Education,
    Experience,
    Gender,
    Identity,
    JobPosting,
    Origin,
    Treatment,
)

LAST_YEAR = 2025


@cache
def _load_json(filename: str) -> dict[str, Any]:
    text = resources.files("llm_hiring_audit.data").joinpath(filename).read_text("utf-8")
    data: dict[str, Any] = json.loads(text)
    return data


def all_treatments() -> list[Treatment]:
    """Every cell of the factorial design, in a stable order."""
    return [
        Treatment(gender=g, origin=o, address_area=a)
        for g, o, a in itertools.product(Gender, Origin, AddressArea)
    ]


def load_job_postings() -> list[JobPosting]:
    return [
        JobPosting(job_id=j["job_id"], title=j["title"], description=j["description"])
        for j in _load_json("jobs.json")["jobs"]
    ]


class CVGenerator:
    """Builds base profiles and crosses them with every treatment.

    Args:
        seed: Seed of the random generator; the same seed always yields
            the same CVs, which makes every audit reproducible.
        names_per_cell: How many distinct names are drawn per treatment
            cell. Using several names lets the analysis separate the effect
            of a group from the idiosyncratic effect of one particular name.
    """

    def __init__(self, seed: int = 0, names_per_cell: int = 2) -> None:
        names = _load_json("names.json")
        pools = [v for g in names["first_names"].values() for v in g.values()]
        pools += list(names["last_names"].values())
        max_names = min(len(pool) for pool in pools)
        if not 1 <= names_per_cell <= max_names:
            raise ValueError(f"names_per_cell must be between 1 and {max_names}")
        self._rng = np.random.default_rng(seed)
        self._names = names
        self._jobs = _load_json("jobs.json")["jobs"]
        self.names_per_cell = names_per_cell

    def generate(self, profiles_per_job: int) -> list[CV]:
        """Return `profiles_per_job * n_jobs * n_cells * names_per_cell` CVs."""
        if profiles_per_job < 1:
            raise ValueError("profiles_per_job must be at least 1")
        cvs: list[CV] = []
        for job in self._jobs:
            for k in range(profiles_per_job):
                profile = self._make_profile(job, f"{job['job_id']}-{k:03d}")
                cvs.extend(self._cross_with_treatments(profile))
        return cvs

    def _pick(self, pool: list[str], size: int) -> list[str]:
        idx = self._rng.choice(len(pool), size=size, replace=False)
        return [pool[i] for i in idx]

    def _make_profile(self, job: dict[str, Any], profile_id: str) -> BaseProfile:
        n_exp = int(self._rng.integers(1, 4))
        titles = self._pick(job["experience_titles"], n_exp)
        employers = self._pick(job["employers"], n_exp)
        descriptions = self._pick(job["experience_descriptions"], n_exp)

        experiences = []
        end = LAST_YEAR
        for title, employer, desc in zip(titles, employers, descriptions, strict=True):
            start = end - int(self._rng.integers(1, 5))
            experiences.append(
                Experience(
                    title=title,
                    employer=employer,
                    start_year=start,
                    end_year=end,
                    description=desc,
                )
            )
            end = start

        education = Education(
            degree=self._pick(job["degrees"], 1)[0],
            school=self._pick(job["schools"], 1)[0],
            year=end,
        )
        n_skills = int(self._rng.integers(3, len(job["skills"]) + 1))
        english = self._pick(["Anglais (B2)", "Anglais (C1)", "Anglais (B1)"], 1)[0]
        return BaseProfile(
            profile_id=profile_id,
            job_id=job["job_id"],
            headline=job["headline"],
            experiences=tuple(experiences),
            education=(education,),
            skills=tuple(self._pick(job["skills"], n_skills)),
            languages=("Français (langue maternelle)", english),
        )

    def _cross_with_treatments(self, profile: BaseProfile) -> list[CV]:
        street_number = int(self._rng.integers(1, 120))
        cvs = []
        for treatment in all_treatments():
            first_pool = self._names["first_names"][treatment.origin][treatment.gender]
            last_pool = self._names["last_names"][treatment.origin]
            firsts = self._pick(first_pool, self.names_per_cell)
            lasts = self._pick(last_pool, self.names_per_cell)
            for i, (first, last) in enumerate(zip(firsts, lasts, strict=True)):
                addresses = self._names["addresses"][treatment.address_area]
                address = addresses[int(self._rng.integers(len(addresses)))]
                identity = Identity(
                    first_name=first,
                    last_name=last,
                    street=f"{street_number} {address['street']}",
                    postcode=address["postcode"],
                    city=address["city"],
                )
                cv_id = (
                    f"{profile.profile_id}-{treatment.gender}-{treatment.origin}"
                    f"-{treatment.address_area}-{i}"
                )
                cvs.append(
                    CV(
                        cv_id=cv_id,
                        profile=profile,
                        treatment=treatment,
                        identity=identity,
                        name_index=i,
                    )
                )
        return cvs
