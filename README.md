# llm-hiring-audit

**Do large language models discriminate when they screen French CVs?**

This project measures it with the method labour economists use to measure discrimination
by human recruiters: the **correspondence audit study** (Bertrand & Mullainathan, 2004;
in France, the DARES / ISM Corum testing campaigns). We send a model CVs that are
identical except for the candidate's name and address, and test whether its scores differ.

## Key results (October 2026)

960 matched CVs × 3 prompt wordings = 2,880 scores per model, on a 1–100 scale. Effects are
in score points relative to a male candidate with a majority French name living in central
Paris, with profile and prompt fixed effects, name-clustered 95% confidence intervals, and
Holm-corrected significance (\*).

| Effect | Mistral 7B (local, Ollama) | open-mistral-nemo (Mistral API) |
|---|---|---|
| Female | **+2.43** [1.94, 2.92]\* | **+1.21** [0.80, 1.62]\* |
| Maghrebi-sounding name | **−0.94** [−1.54, −0.35]\* | −0.16 [−0.69, 0.36] |
| Sub-Saharan-sounding name | **−2.02** [−2.66, −1.39]\* | +0.01 [−0.51, 0.54] |
| Deprived-area address | **−1.09** [−1.55, −0.63]\* | +0.21 [−0.22, 0.64] |
| Unparseable answers | 0.5% (no group difference) | 0.1% (no group difference) |

- **Two models from the same provider behave very differently.** Mistral 7B penalises
  candidates with sub-Saharan-sounding names (−2.0 points), Maghrebi-sounding names
  (−0.9) and a deprived-area address (−1.1). open-mistral-nemo shows none of these
  penalties, and its intervals rule out penalties larger than about 0.7 points.
- **Both models favour women**, by +2.4 and +1.2 points.
- **The main effects are not driven by one prompt or one job** (`llm-hiring-audit
  robustness`, Holm correction across all 28 subgroup tests). For Mistral 7B, the female
  premium is significant in all 7 subsets (3 prompt wordings, 4 jobs) and the sub-Saharan
  penalty in 6 of 7 (in accounting it is −1.9 points, p = 0.07 after correction). For
  open-mistral-nemo, the female premium is positive in all 7 subsets (+0.8 to +1.8) but
  significant in only 3: a 1-point effect is too small to confirm within each subset alone.
  One subset result runs against the pooled pattern (nemo, payroll jobs: deprived address
  −1.15, significant); with 28 tests it is treated as exploratory.
- **For Mistral 7B, the female premium goes mostly to women with majority French names**
  (`llm-hiring-audit intersectional`). Relative to a man with a majority name: women with
  a majority name +3.3, Maghrebi-sounding names +1.9, sub-Saharan-sounding names +0.5 (not
  significant); men with sub-Saharan-sounding names −1.2. The sub-Saharan penalty looks
  larger for women than for men (interaction −1.6 points), but this is borderline after
  correction (p = 0.054) and is reported as suggestive. For open-mistral-nemo the female
  premium is similar for every origin (+1.0 to +1.2) and no interaction is detected.
- Every result is confirmed by the cluster bootstrap (see
  [`results/2026-10/`](results/2026-10/)), which also contains the raw scores needed to
  reproduce the analysis.

These gaps are small on a 100-point scale, but they are systematic: a screening tool that
ranks or thresholds candidates turns a consistent 2-point gap into fewer shortlisted
candidates from the penalised group. Results apply to these model versions, this prompt
design and these four jobs. Models change, so audits should be re-run on each new version.

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
uv run llm-hiring-audit score --backend mistral --model open-mistral-nemo --cvs data/cvs.jsonl --out data/scores.jsonl

# Gemini API (set GEMINI_API_KEY first; the free tier allows too few requests per day
# for a full audit, so this needs a paid key)
uv run llm-hiring-audit score --backend gemini --model gemini-3.5-flash --cvs data/cvs.jsonl --out data/scores.jsonl

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
uv run llm-hiring-audit robustness --scores data/scores.jsonl --out-dir results/
```

`robustness` re-estimates every effect within each prompt wording and each job, with the
Holm correction applied across all subgroup tests together.

`intersectional` adds gender × origin interaction terms to the main model, to test whether
the origin effect differs between women and men, and reports the gap of each gender ×
origin group relative to a man with a majority French name.

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
| Maghrebi-sounding name | −6 | −5.57 [−6.24, −4.89] | yes |
| Deprived-area address | ≈ −1.33 | −1.58 [−2.17, −0.99] | yes |
| Female | 0 | 0.56 [−0.01, 1.12] | no |
| Sub-Saharan-sounding name | 0 | 0.23 [−0.45, 0.92] | no |

Both planted effects are detected with the truth inside the interval, and no effect is
found where none was planted. These properties are checked automatically in
`tests/test_analysis.py`.

## Power analysis

How many CVs does an audit need? `power` simulates many audits with the same design and
analysis as the real one and reports how often a true effect is detected:

```bash
uv run llm-hiring-audit power --residual-sd 5.92 --name-sd 2.65 --effect 2
```

The noise inputs come from `analyze` on a 48-CV pilot run of each model. open-mistral-nemo
was barely affected by individual names (residual sd 4.8, name sd 0.8) and reaches 100%
power from 480 CVs. Mistral 7B via Ollama reacted much more to individual names (residual sd
5.9, name sd 2.7), and the simulation showed that **the number of distinct names, not the
number of CVs, limited the audit**: a first name keeps its own quirk whatever CV it is put
on, so with 5 first names per group, adding CVs kept reusing the same few quirks.

Power to detect a 2-point gap for Mistral 7B (100 simulations each):

| Profiles per job | CVs | 5 first names per group | 12 first names per group |
|---|---|---|---|
| 5 | 480 | 64% | 76% |
| 10 | 960 | 62% | **90%** |
| 20 | 1,920 | 74% | 93% |

The name pool was therefore enlarged (see [Names](#names)), and the audits use 10 profiles
per job (960 CVs).

## Names

First names are the 12 most frequent names given in France to children born 1985–2000
(people aged about 25–40, a plausible candidate age) in each group, according to the
[INSEE first-names file](https://www.insee.fr/fr/statistiques/8595130), among names typical
of the group. Names widely given across groups (e.g. *Sarah*, *Inès*, *Myriam*, *Issa*) are
excluded, since they would weaken the origin signal. Every name was given at least 425
times in that period; `scripts/check_first_names.py` re-checks this against the INSEE file.
INSEE does not publish surnames, so the 10 surnames per group are common surnames associated
with each group.

## Development

```bash
uv run pytest          # tests + coverage
uv run ruff check .    # lint
uv run mypy src tests  # strict type checking
```

## Design decisions

The main choices behind the audit, the alternatives considered and the trade-offs accepted
are recorded in [docs/decisions.md](docs/decisions.md).

## Roadmap

- [x] Matched CV generator (factorial design)
- [x] Model backends: local models (Ollama), Mistral API and Gemini API, with mocked tests
- [x] Three prompt wordings
- [x] Analysis: fixed-effects regression with name-clustered errors, cluster bootstrap,
      Holm correction, refusal rates, ceiling check
- [x] Validation: recover a planted bias from the mock model
- [x] Power analysis: how many CVs are needed to detect a given effect
- [x] Larger name pool, checked against INSEE data
- [x] Audits of real models: Mistral 7B (local) and open-mistral-nemo (API)
- [x] Robustness by prompt and job (`robustness` command)
- [x] Intersectional effects (gender × origin)
- [x] Results write-up and policy brief
- [ ] More models: a widely deployed frontier model (e.g. GPT, Gemini, Claude) via a paid API

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

The audit covers two open-weight models from one provider, Mistral, because both could
be run for free. It shows that models from the same provider can behave differently, but
it does not tell how the frontier models most used in industry behave; the Gemini backend
and the rest of the pipeline are ready for that extension.
