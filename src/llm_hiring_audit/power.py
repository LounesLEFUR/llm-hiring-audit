"""Power analysis by simulation.

Question: with a given number of CVs, what is the probability of
detecting a true effect of `effect` points?

We simulate many fake audits with the same design and analysis as the
real one (profile fixed effects, name-clustered errors, Holm correction)
and count how often the effect comes out significant. Simulation avoids
closed-form formulas that would ignore clustering and multiple testing.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from llm_hiring_audit.analysis import TREATMENT_DUMMIES, estimate_effects, prepare
from llm_hiring_audit.generator import all_treatments

NAMES_PER_GROUP = 25  # 5 first names x 5 last names per gender x origin cell


@dataclass(frozen=True)
class Design:
    profiles_per_job: int
    n_jobs: int = 4
    names_per_cell: int = 2
    n_prompts: int = 3
    repetitions: int = 1

    @property
    def n_cvs(self) -> int:
        return self.n_jobs * self.profiles_per_job * len(all_treatments()) * self.names_per_cell

    @property
    def n_queries(self) -> int:
        return self.n_cvs * self.n_prompts * self.repetitions


def simulate_scores(
    design: Design,
    effect: float,
    effect_name: str,
    residual_sd: float,
    name_sd: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One fake audit in which only `effect_name` has a non-zero effect."""
    factor, level = TREATMENT_DUMMIES[effect_name]
    cells = pd.DataFrame([t.model_dump() for t in all_treatments()]).astype(str)
    group = pd.factorize(cells["gender"] + "-" + cells["origin"])[0]
    n_profiles = design.n_jobs * design.profiles_per_job
    k = design.names_per_cell

    # Each profile x cell draws k distinct names out of the cell's name pool.
    names = rng.random((n_profiles, len(cells), NAMES_PER_GROUP)).argsort(axis=2)[:, :, :k]
    shape = (n_profiles, len(cells), k, design.n_prompts, design.repetitions)
    profile, cell, slot, prompt, rep = (a.ravel() for a in np.indices(shape))
    name = names[profile, cell, slot]

    name_effects = rng.normal(0, name_sd, size=(group.max() + 1, NAMES_PER_GROUP))
    profile_effects = rng.normal(0, 10, size=n_profiles)
    treated = (cells[factor] == level).to_numpy()[cell]

    df: pd.DataFrame = cells.iloc[cell].reset_index(drop=True)
    df["model"] = "simulated"
    df["profile_id"] = profile.astype(str)
    df["prompt_id"] = prompt.astype(str)
    df["repetition"] = rep
    df["first_name"] = df["gender"] + "-" + df["origin"] + "-" + name.astype(str)
    df["last_name"] = ""
    df["score"] = (
        70
        + profile_effects[profile]
        + 2 * prompt
        + name_effects[group[cell], name]
        + effect * treated
        + rng.normal(0, residual_sd, size=len(df))
    )
    return df


def estimate_power(
    design: Design,
    effect: float,
    residual_sd: float,
    name_sd: float,
    effect_name: str = "maghrebi",
    n_sims: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> float:
    """Share of simulated audits in which `effect_name` is significant after Holm."""
    if effect_name not in TREATMENT_DUMMIES:
        raise ValueError(f"effect_name must be one of {list(TREATMENT_DUMMIES)}")
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        df = prepare(simulate_scores(design, effect, effect_name, residual_sd, name_sd, rng))
        effects = estimate_effects(df, alpha=alpha).set_index("effect")
        hits += bool(effects["significant"][effect_name])
    return hits / n_sims
