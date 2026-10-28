# llm-hiring-audit

**Do large language models discriminate when they screen French CVs?**

This project measures it with the method labour economists use to measure discrimination
by human recruiters: the **correspondence audit study** (Bertrand & Mullainathan, 2004;
in France, the DARES / ISM Corum testing campaigns). We send a model CVs that are
identical except for the candidate's name and address, and test whether its scores differ.

> Status: work in progress. The full pipeline (generation, scoring, analysis) works and is
> validated on simulated data; audits of real models are next.

## Why it matters

Using AI to recruit or select candidates is classified as **high-risk** by the EU AI Act
(Regulation (EU) 2024/1689, **Annex III, point 4(a)**). Providers of such systems must
examine their data for possible biases (Article 10(2)(f)), and French law prohibits
discrimination in hiring based on origin, sex or place of residence (Code du travail,
art. L1132-1). This tool turns those requirements into a reproducible, quantitative test.

## Experimental design

Each **base profile** (experience, education, skills) is crossed with every cell of a
full factorial design:

| Factor | Levels |
|---|---|
| Gender (signalled by first name) | female, male |
| Perceived origin (signalled by name) | majority French, Maghrebi, sub-Saharan African |
| Address | central Paris, deprived neighbourhood (QPV) |

That gives 12 versions of every profile, with several different names per cell so the
analysis can separate the effect of a *group* from the effect of one particular *name*.

Design guarantees, each enforced by a unit test:

- **Qualifications are held constant**: CVs of the same profile differ only in name,
  address and email.
- **Gender-neutral wording**: French job titles are epicene (*comptable*,
  *gestionnaire de paie*), so grammatical gender cannot leak the treatment.
- **Balanced design**: every cell has the same number of CVs.
- **Reproducibility**: the same seed always produces the same CVs.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run llm-hiring-audit generate --profiles-per-job 5 --out data/cvs.jsonl
```

This writes 480 CVs (4 jobs × 5 profiles × 12 cells × 2 names), one JSON record per line.

Then ask a model to score them:

```bash
# Local model via Ollama
uv run llm-hiring-audit score --backend ollama --model mistral --cvs data/cvs.jsonl --out data/scores.jsonl

# Mistral API (set MISTRAL_API_KEY first)
uv run llm-hiring-audit score --backend mistral --model mistral-small-latest --cvs data/cvs.jsonl --out data/scores.jsonl

# Fake model with a planted bias, to validate the analysis
uv run llm-hiring-audit score --backend mock --mock-penalty Sevran=-4 --cvs data/cvs.jsonl --out data/scores_mock.jsonl
```

Each CV is scored on a 1–100 scale with three prompt wordings (rating, interview, hiring)
and several repetitions, because LLM answers are sensitive to wording and vary between
calls. Queries are shuffled so that model drift during a long run cannot line up with a
demographic group. Runs can be interrupted and resumed, and answers that contain no valid
score are kept (`score: null`) so the refusal rate can be analysed too.

## Statistical analysis

```bash
uv run llm-hiring-audit analyze --scores data/scores.jsonl --out-dir results/
```

For each model, the effect of each signal is estimated in score points with

$$
\text{score}_{i} = \beta_1\,\text{female}_i + \beta_2\,\text{maghrebi}_i + \beta_3\,\text{subsaharan}_i
+ \beta_4\,\text{deprived}_i + \alpha_{\text{profile}(i)} + \gamma_{\text{prompt}(i)} + \varepsilon_i
$$

- **Profile fixed effects** ($\alpha$) compare each CV only with the other versions of the
  same profile, so differences in qualifications cannot contaminate the estimates.
- **Prompt fixed effects** ($\gamma$) absorb level differences between the three wordings.
- **Standard errors clustered by name**: all scores of one name share that name's own
  effect, so treating them as independent would make confidence intervals too narrow.
- **Cluster bootstrap over profiles** gives a second confidence interval that does not rely
  on the regression's distributional assumptions.
- **Holm correction** controls the family-wise error rate across the effects tested together.
- **Refusal rates** by group are tested with a chi-square test, since refusing to score some
  candidates more often is itself a form of unequal treatment.
- **Ceiling check**: the share of scores at 90 or above, since a compressed scale can hide
  real differences.

### Validation on a model with a known bias

Before trusting the analysis on real models, it is run on a mock model with planted
penalties: −6 points for Maghrebi surnames and −4 points for one of the three deprived-area
addresses (so about −1.33 on average for the deprived area). 480 CVs × 3 prompts:

| Effect | Truth | Estimate [95% CI] | Significant (Holm) |
|---|---|---|---|
| Maghrebi-sounding name | −6 | −5.52 [−6.19, −4.85] | yes |
| Deprived-area address | ≈ −1.33 | −1.60 [−2.18, −1.03] | yes |
| Female | 0 | −0.35 [−0.90, 0.19] | no |
| Sub-Saharan-sounding name | 0 | −0.02 [−0.69, 0.65] | no |

Both planted effects are detected with the truth inside the interval, and no effect is
found where none was planted. These properties are checked automatically in
`tests/test_analysis.py`.

## Power analysis

How many CVs does an audit need? `power` simulates many audits with the same design and
analysis as the real one and reports how often a true effect is detected:

```bash
uv run llm-hiring-audit power --residual-sd 5.92 --name-sd 2.65 --effect 2
```

The noise inputs come from `analyze` on a 48-CV pilot run of each model. Power to detect a
2-point gap (100 simulations each):

| Profiles per job | CVs | open-mistral-nemo (sd 4.8, name sd 0.8) | Mistral 7B via Ollama (sd 5.9, name sd 2.7) |
|---|---|---|---|
| 2 | 192 | 89% | 43% |
| 5 | 480 | 100% | 56% |
| 10 | 960 | 100% | 79% |
| 20 | 1,920 | 100% | 78% |

For Mistral 7B, power stops increasing after about 10 profiles per job, and drawing 4 names
per cell instead of 2 only raises it to about 85%. When the model reacts strongly to
individual names, the number of *distinct names* limits the audit, not the number of CVs.
With 5 first names per group, adding more CVs keeps reusing the same names. A larger name
pool is the way to gain power.

## Development

```bash
uv run pytest          # tests + coverage
uv run ruff check .    # lint
uv run mypy src tests  # strict type checking
```

## Roadmap

- [x] Matched CV generator (factorial design)
- [x] Model backends: local models (Ollama) and Mistral API, with mocked tests
- [x] Three prompt wordings
- [x] Analysis: fixed-effects regression with name-clustered errors, cluster bootstrap,
      Holm correction, refusal rates, ceiling check
- [x] Validation: recover a planted bias from the mock model
- [x] Power analysis: how many CVs are needed to detect a given effect
- [ ] Larger name pool (power is limited by the number of distinct names)
- [ ] Audits of real models (Mistral 7B local, open-mistral-nemo, ministral-8b)
- [ ] Results report and write-up

## Related work

- Bertrand & Mullainathan (2004), *Are Emily and Greg More Employable than Lakisha and
  Jamal?* The original correspondence study of hiring discrimination.
- Armstrong et al. (2024), [*The Silicon Ceiling: Auditing GPT's Race and Gender Biases in
  Hiring*](https://arxiv.org/abs/2405.04412). Inspired the three prompt wordings; they model
  name-level variation with a random effect, which this project handles by clustering.
- Wilson & Caliskan (2024), [*Gender, Race, and Intersectional Bias in Resume Screening via
  Language Model Retrieval*](https://arxiv.org/abs/2407.20371).
- Tamkin et al. (2023), [*Evaluating and Mitigating Discrimination in Language Model
  Decisions*](https://arxiv.org/abs/2312.03689).
- Human recruiters in France: the DARES / ISM Corum testing campaigns, and the
  [IPP policy brief](https://www.ipp.eu/en/news/discrimination-in-hiring-in-france-findings-and-courses-of-action)
  on discrimination in hiring.

Most existing LLM audits use English CVs and US names. This project focuses on French,
on the origin groups relevant to the French labour market, and on place of residence.

## Limitations

Names and addresses are *signals of perceived* group membership, not identities. The
results describe how a model reacts to these signals in this specific setting.
