import re
from collections import Counter

import pytest

from llm_hiring_audit.generator import (
    CVGenerator,
    _load_json,
    all_treatments,
    load_job_postings,
)
from llm_hiring_audit.render import render_cv

N_CELLS = 2 * 3 * 2  # gender x origin x address area


def test_factorial_design_has_twelve_cells() -> None:
    assert len(all_treatments()) == N_CELLS
    assert len(set(all_treatments())) == N_CELLS


def test_number_of_cvs() -> None:
    cvs = CVGenerator(seed=1, names_per_cell=3).generate(profiles_per_job=2)
    n_jobs = len(_load_json("jobs.json")["jobs"])
    assert len(cvs) == n_jobs * 2 * N_CELLS * 3
    assert len({cv.cv_id for cv in cvs}) == len(cvs)


def test_design_is_balanced() -> None:
    cvs = CVGenerator(seed=2).generate(profiles_per_job=3)
    counts = Counter(cv.treatment for cv in cvs)
    assert len(set(counts.values())) == 1


def test_same_seed_gives_same_cvs() -> None:
    a = CVGenerator(seed=42).generate(profiles_per_job=2)
    b = CVGenerator(seed=42).generate(profiles_per_job=2)
    assert [render_cv(cv) for cv in a] == [render_cv(cv) for cv in b]


def test_different_seeds_give_different_cvs() -> None:
    a = CVGenerator(seed=1).generate(profiles_per_job=2)
    b = CVGenerator(seed=2).generate(profiles_per_job=2)
    assert [render_cv(cv) for cv in a] != [render_cv(cv) for cv in b]


def test_cvs_of_a_profile_differ_only_by_identity() -> None:
    """The core guarantee of the experiment: qualifications are held constant."""
    cvs = CVGenerator(seed=3).generate(profiles_per_job=2)
    by_profile: dict[str, set[str]] = {}
    for cv in cvs:
        # Drop the 3 identity lines (name, address, email); keep the rest.
        body = "\n".join(render_cv(cv).splitlines()[3:])
        by_profile.setdefault(cv.profile.profile_id, set()).add(body)
    assert all(len(bodies) == 1 for bodies in by_profile.values())


def test_names_match_their_cell() -> None:
    names = _load_json("names.json")
    for cv in CVGenerator(seed=4).generate(profiles_per_job=1):
        t = cv.treatment
        assert cv.identity.first_name in names["first_names"][t.origin][t.gender]
        assert cv.identity.last_name in names["last_names"][t.origin]
        cities = {a["city"] for a in names["addresses"][t.address_area]}
        assert cv.identity.city in cities


# Masculine-only forms that would leak gender through grammar rather than the name.
GENDERED_WORDS = re.compile(
    r"\b(employé|adjoint|technicien|collaborateur|chargé|confirmé|vendeur|directeur"
    r"|employée|adjointe|technicienne|collaboratrice|chargée|confirmée|vendeuse|directrice)\b",
    re.IGNORECASE,
)


def test_cv_wording_is_gender_neutral() -> None:
    for cv in CVGenerator(seed=5).generate(profiles_per_job=5):
        body = "\n".join(render_cv(cv).splitlines()[3:])
        assert not GENDERED_WORDS.search(body), body


def test_email_is_ascii() -> None:
    for cv in CVGenerator(seed=6).generate(profiles_per_job=1):
        assert cv.identity.email.isascii()


@pytest.mark.parametrize("bad", [0, 99])
def test_invalid_names_per_cell(bad: int) -> None:
    with pytest.raises(ValueError):
        CVGenerator(names_per_cell=bad)


def test_every_profile_has_a_job_posting() -> None:
    job_ids = {job.job_id for job in load_job_postings()}
    assert {cv.profile.job_id for cv in CVGenerator().generate(1)} == job_ids


def test_invalid_profiles_per_job() -> None:
    with pytest.raises(ValueError):
        CVGenerator().generate(profiles_per_job=0)
