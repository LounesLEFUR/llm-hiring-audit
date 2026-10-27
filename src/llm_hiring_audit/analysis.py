"""Statistical analysis of audit scores.

Main specification (one model at a time):

    score = b1*female + b2*maghrebi + b3*subsaharan + b4*deprived
            + profile fixed effects + prompt fixed effects + error

- Profile fixed effects compare each CV only with the other versions of
  the same profile, so differences in qualifications cannot bias the b's.
- Standard errors are clustered by name: all scores of one name share
  that name's idiosyncratic effect, so they are not independent draws.
  Ignoring this would make confidence intervals too narrow.
- A cluster bootstrap over profiles gives a second, assumption-light
  confidence interval as a robustness check.
- p-values are Holm-corrected, since several effects are tested at once.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# Dummy variable -> (factor, level); reference levels are male, majority, central.
TREATMENT_DUMMIES: dict[str, tuple[str, str]] = {
    "female": ("gender", "female"),
    "maghrebi": ("origin", "maghrebi"),
    "subsaharan": ("origin", "subsaharan"),
    "deprived": ("address_area", "deprived"),
}
LABELS = {
    "female": "Female (vs male)",
    "maghrebi": "Maghrebi-sounding name (vs majority)",
    "subsaharan": "Sub-Saharan-sounding name (vs majority)",
    "deprived": "Deprived-area address (vs central Paris)",
}
FACTORS = ("gender", "origin", "address_area")


def prepare(scores: pd.DataFrame) -> pd.DataFrame:
    """Add treatment dummies and a name identifier; keep valid scores only."""
    df = scores.dropna(subset=["score"]).copy()
    for dummy, (factor, level) in TREATMENT_DUMMIES.items():
        df[dummy] = (df[factor] == level).astype(int)
    df["name"] = df["first_name"] + " " + df["last_name"]
    return df


def _formula(df: pd.DataFrame) -> str:
    rhs = list(TREATMENT_DUMMIES)
    if df["profile_id"].nunique() > 1:
        rhs.append("C(profile_id)")
    if df["prompt_id"].nunique() > 1:
        rhs.append("C(prompt_id)")
    return "score ~ " + " + ".join(rhs)


def estimate_effects(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Treatment effects in score points, with name-clustered inference.

    `df` must come from `prepare` and contain a single model.
    """
    names = pd.factorize(df["name"])[0]
    fit = smf.ols(_formula(df), data=df).fit(cov_type="cluster", cov_kwds={"groups": names})
    terms = list(TREATMENT_DUMMIES)
    ci = fit.conf_int(alpha=alpha).loc[terms]
    table = pd.DataFrame(
        {
            "effect": terms,
            "label": [LABELS[t] for t in terms],
            "estimate": fit.params[terms].to_numpy(),
            "std_error": fit.bse[terms].to_numpy(),
            "ci_low": ci[0].to_numpy(),
            "ci_high": ci[1].to_numpy(),
            "p_value": fit.pvalues[terms].to_numpy(),
        }
    )
    table["p_holm"] = multipletests(table["p_value"], alpha=alpha, method="holm")[1]
    table["significant"] = table["p_holm"] < alpha
    return table


def bootstrap_effects(
    df: pd.DataFrame, n_boot: int = 500, seed: int = 0, alpha: float = 0.05
) -> pd.DataFrame:
    """Percentile confidence intervals from a cluster bootstrap over profiles.

    Whole profiles (with all their versions) are resampled, which keeps the
    matched structure of the experiment intact inside each resample.
    """
    rng = np.random.default_rng(seed)
    profiles = [group for _, group in df.groupby("profile_id")]
    formula = _formula(df)
    terms = list(TREATMENT_DUMMIES)
    draws = np.empty((n_boot, len(terms)))
    for b in range(n_boot):
        picked = rng.integers(len(profiles), size=len(profiles))
        # A profile drawn twice must count as two distinct fixed effects.
        sample = pd.concat(profiles[i].assign(profile_id=str(k)) for k, i in enumerate(picked))
        draws[b] = smf.ols(formula, data=sample).fit().params[terms].to_numpy()
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2], axis=0)
    return pd.DataFrame({"effect": terms, "boot_ci_low": low, "boot_ci_high": high})


def refusal_rates(scores: pd.DataFrame) -> pd.DataFrame:
    """Share of answers without a valid score, by factor level, with a chi-square test.

    A model that refuses more often for some names discriminates too, and
    dropping refusals silently would hide it.
    """
    rows = []
    missing = scores["score"].isna()
    for factor in FACTORS:
        table = pd.crosstab(scores[factor], missing)
        p = np.nan
        if table.shape[0] > 1 and table.shape[1] == 2:  # test only if some refusals
            p = float(chi2_contingency(table)[1])
        for level, rate in missing.groupby(scores[factor]).mean().items():
            rows.append({"factor": factor, "level": level, "refusal_rate": rate, "p_value": p})
    return pd.DataFrame(rows)


def score_distribution(df: pd.DataFrame, ceiling: int = 90) -> dict[str, float]:
    """Summary statistics, including how many scores sit near the top of the scale.

    If most scores are at or above `ceiling`, the scale is compressed and
    real differences between groups can be hidden (ceiling effect).
    """
    s = df["score"]
    return {
        "n": float(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "share_at_or_above_ceiling": float((s >= ceiling).mean()),
        "n_distinct_scores": float(s.nunique()),
    }


@dataclass
class AuditReport:
    model: str
    effects: pd.DataFrame
    refusals: pd.DataFrame
    distribution: dict[str, float]


def analyze(scores: pd.DataFrame, n_boot: int = 500, seed: int = 0) -> list[AuditReport]:
    """Run the full analysis separately for each model in the scores file."""
    reports = []
    for model, raw in scores.groupby("model"):
        df = prepare(raw)
        effects = estimate_effects(df)
        if n_boot > 0:
            effects = effects.merge(bootstrap_effects(df, n_boot=n_boot, seed=seed), on="effect")
        reports.append(
            AuditReport(
                model=str(model),
                effects=effects,
                refusals=refusal_rates(raw),
                distribution=score_distribution(df),
            )
        )
    return reports
