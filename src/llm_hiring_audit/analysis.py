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


SUBGROUP_COLUMNS = ("prompt_id", "job_id")


def subgroup_effects(
    df: pd.DataFrame, by: tuple[str, ...] = SUBGROUP_COLUMNS, alpha: float = 0.05
) -> pd.DataFrame:
    """Re-estimate the effects within each prompt wording and each job.

    Checks that a pooled result is not driven by a single prompt or job.
    The Holm correction is applied across *all* subgroup tests at once:
    with 7 subgroups x 4 effects = 28 tests, correcting within each
    subgroup separately would let several false positives through.
    """
    unknown = set(by) - set(SUBGROUP_COLUMNS)
    if unknown:
        raise ValueError(f"Can only split by {SUBGROUP_COLUMNS}, got {sorted(unknown)}")
    tables = []
    for column in by:
        for value, sub in df.groupby(column):
            table = estimate_effects(sub, alpha=alpha).drop(columns=["p_holm", "significant"])
            tables.append(table.assign(subgroup=column, value=str(value)))
    result = pd.concat(tables, ignore_index=True)
    result["p_holm"] = multipletests(result["p_value"], alpha=alpha, method="holm")[1]
    result["significant"] = result["p_holm"] < alpha
    columns = ["subgroup", "value", "effect", "estimate", "ci_low", "ci_high"]
    return result[[*columns, "p_value", "p_holm", "significant"]]


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


@dataclass(frozen=True)
class NoiseEstimate:
    residual_sd: float  # query-to-query noise
    name_sd: float  # spread of the names' own effects


def estimate_noise(df: pd.DataFrame) -> NoiseEstimate:
    """Split the unexplained variance into a name part and a residual part.

    One-way ANOVA (method of moments) on the regression residuals: the
    variance of the names' mean residuals is name_var + residual_var / n,
    where n is the average number of scores per name. These two numbers
    are the inputs of the power analysis.
    """
    fit = smf.ols(_formula(df), data=df).fit()
    total_var = float(fit.scale)
    by_name = fit.resid.groupby(df["name"])
    n = float(by_name.size().mean())
    if n <= 1:
        return NoiseEstimate(residual_sd=total_var**0.5, name_sd=0.0)
    name_var = (float(by_name.mean().var()) - total_var / n) / (1 - 1 / n)
    name_var = min(max(name_var, 0.0), total_var)
    return NoiseEstimate(residual_sd=(total_var - name_var) ** 0.5, name_sd=name_var**0.5)


@dataclass
class AuditReport:
    model: str
    effects: pd.DataFrame
    refusals: pd.DataFrame
    distribution: dict[str, float]
    noise: NoiseEstimate


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
                noise=estimate_noise(df),
            )
        )
    return reports
