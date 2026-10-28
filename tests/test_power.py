import numpy as np
import pytest

from llm_hiring_audit.analysis import estimate_noise, prepare
from llm_hiring_audit.power import Design, estimate_power, simulate_scores


def test_design_sizes() -> None:
    design = Design(profiles_per_job=5, names_per_cell=2, n_prompts=3, repetitions=2)
    assert design.n_cvs == 4 * 5 * 12 * 2
    assert design.n_queries == design.n_cvs * 3 * 2


def test_simulation_plants_the_effect() -> None:
    rng = np.random.default_rng(0)
    df = simulate_scores(Design(profiles_per_job=5), -8, "deprived", 1.0, 0.0, rng)
    means = df.groupby("address_area")["score"].mean()
    assert means["deprived"] - means["central"] == pytest.approx(-8, abs=1)


def test_noise_estimate_recovers_known_values() -> None:
    rng = np.random.default_rng(1)
    design = Design(profiles_per_job=10, repetitions=2)
    df = prepare(simulate_scores(design, 0, "female", 5.0, 3.0, rng))
    noise = estimate_noise(df)
    assert noise.residual_sd == pytest.approx(5.0, rel=0.1)
    assert noise.name_sd == pytest.approx(3.0, rel=0.35)


def test_no_effect_means_few_false_positives() -> None:
    p = estimate_power(Design(profiles_per_job=2), 0.0, 5.0, 1.0, n_sims=100)
    assert p <= 0.1  # close to (or below) alpha = 0.05


def test_large_effect_is_always_detected() -> None:
    p = estimate_power(Design(profiles_per_job=2), -10.0, 5.0, 1.0, n_sims=30)
    assert p == 1.0


def test_power_grows_with_sample_size() -> None:
    small = estimate_power(Design(profiles_per_job=1), -1.5, 5.0, 0.5, n_sims=60)
    large = estimate_power(Design(profiles_per_job=8), -1.5, 5.0, 0.5, n_sims=60)
    assert large > small + 0.3


def test_unknown_effect_name() -> None:
    with pytest.raises(ValueError):
        estimate_power(Design(profiles_per_job=1), 1.0, 1.0, 0.0, effect_name="age")
