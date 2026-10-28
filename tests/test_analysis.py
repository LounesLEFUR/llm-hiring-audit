"""Validation of the analysis on simulated audits where the truth is known."""

from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from llm_hiring_audit.analysis import (
    analyze,
    bootstrap_effects,
    estimate_effects,
    prepare,
    refusal_rates,
    score_distribution,
    subgroup_effects,
)
from llm_hiring_audit.backends import MockBackend
from llm_hiring_audit.generator import CVGenerator, _load_json
from llm_hiring_audit.render import cv_to_record
from llm_hiring_audit.scoring import Scorer, read_jsonl

TRUE_MAGHREBI = -6.0
TRUE_DEPRIVED = -4.0


def _simulate(tmp_path: Path, mock: MockBackend, profiles_per_job: int = 3) -> pd.DataFrame:
    cvs = [cv_to_record(cv) for cv in CVGenerator(seed=0).generate(profiles_per_job)]
    out = tmp_path / "scores.jsonl"
    Scorer(mock, repetitions=1).run(cvs, out)
    return pd.DataFrame(read_jsonl(out))


@pytest.fixture(scope="module")
def biased_scores(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    names = _load_json("names.json")
    penalties = {name: TRUE_MAGHREBI for name in names["last_names"]["maghrebi"]}
    penalties |= {a["city"]: TRUE_DEPRIVED for a in names["addresses"]["deprived"]}
    return _simulate(tmp_path_factory.mktemp("biased"), MockBackend(penalties=penalties))


@pytest.fixture(scope="module")
def fair_scores(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    return _simulate(tmp_path_factory.mktemp("fair"), MockBackend())


def _by_effect(table: pd.DataFrame) -> dict[Hashable, dict[Hashable, Any]]:
    return table.set_index("effect").to_dict("index")


def test_recovers_planted_bias(biased_scores: pd.DataFrame) -> None:
    effects = _by_effect(estimate_effects(prepare(biased_scores)))
    for effect, truth in [("maghrebi", TRUE_MAGHREBI), ("deprived", TRUE_DEPRIVED)]:
        row = effects[effect]
        assert row["ci_low"] <= truth <= row["ci_high"]
        assert row["significant"]


def test_no_false_positive_on_unbiased_effects(biased_scores: pd.DataFrame) -> None:
    effects = _by_effect(estimate_effects(prepare(biased_scores)))
    for effect in ["female", "subsaharan"]:
        row = effects[effect]
        assert not row["significant"]
        assert row["ci_low"] <= 0 <= row["ci_high"]


def test_fair_model_shows_no_significant_effect(fair_scores: pd.DataFrame) -> None:
    effects = estimate_effects(prepare(fair_scores))
    assert not effects["significant"].any()


def test_holm_p_values_are_never_smaller(biased_scores: pd.DataFrame) -> None:
    effects = estimate_effects(prepare(biased_scores))
    assert (effects["p_holm"] >= effects["p_value"]).all()


def test_bootstrap_interval_contains_truth(biased_scores: pd.DataFrame) -> None:
    row = _by_effect(bootstrap_effects(prepare(biased_scores), n_boot=50))["maghrebi"]
    assert row["boot_ci_low"] <= TRUE_MAGHREBI <= row["boot_ci_high"] < 0


def test_refusals_are_detected(biased_scores: pd.DataFrame) -> None:
    scores = biased_scores.copy()
    # Simulate a model that refuses to score half of the female CVs.
    female_rows = scores.index[scores["gender"] == "female"]
    scores.loc[female_rows[::2], "score"] = None
    rates = refusal_rates(scores).set_index(["factor", "level"]).to_dict("index")
    assert rates[("gender", "female")]["refusal_rate"] == pytest.approx(0.5, abs=0.01)
    assert rates[("gender", "male")]["refusal_rate"] == 0
    assert rates[("gender", "female")]["p_value"] < 0.001


def test_no_refusals_gives_no_test(fair_scores: pd.DataFrame) -> None:
    rates = refusal_rates(fair_scores)
    assert (rates["refusal_rate"] == 0).all()
    assert rates["p_value"].isna().all()


def test_score_distribution_flags_ceiling() -> None:
    df = pd.DataFrame({"score": [95, 92, 90, 60]})
    dist = score_distribution(df)
    assert dist["share_at_or_above_ceiling"] == 0.75
    assert dist["n_distinct_scores"] == 4


def test_analyze_reports_each_model(biased_scores: pd.DataFrame) -> None:
    both = pd.concat([biased_scores, biased_scores.assign(model="other")])
    reports = analyze(both, n_boot=10)
    assert [r.model for r in reports] == ["mock", "other"]
    assert {"boot_ci_low", "boot_ci_high"} <= set(reports[0].effects.columns)


def test_subgroups_cover_every_prompt_and_job(biased_scores: pd.DataFrame) -> None:
    table = subgroup_effects(prepare(biased_scores))
    assert set(table["value"]) == set(biased_scores["prompt_id"]) | set(biased_scores["job_id"])
    assert len(table) == (3 + 4) * 4  # 7 subgroups x 4 effects


def test_planted_bias_holds_in_every_subgroup(biased_scores: pd.DataFrame) -> None:
    table = subgroup_effects(prepare(biased_scores))
    maghrebi = table[table["effect"] == "maghrebi"]
    assert maghrebi["significant"].all()
    # Each 95% interval misses the truth 5% of the time, so requiring all 7
    # to cover it would fail about 30% of the time (1 - 0.95**7). At least 5
    # of 7 happens with probability 0.996 when the method is correct.
    covers = (maghrebi["ci_low"] <= TRUE_MAGHREBI) & (maghrebi["ci_high"] >= TRUE_MAGHREBI)
    assert covers.sum() >= 5


def test_subgroup_holm_covers_all_tests(fair_scores: pd.DataFrame) -> None:
    table = subgroup_effects(prepare(fair_scores))
    assert (table["p_holm"] >= table["p_value"]).all()
    assert not table["significant"].any()


def test_subgroup_rejects_unknown_column(fair_scores: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Can only split"):
        subgroup_effects(prepare(fair_scores), by=("city",))
