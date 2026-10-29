# Methodology

How the audit works, in enough detail to check or reproduce it. The reasons behind each choice,
and the alternatives considered, are in [decisions.md](decisions.md).

- [Experimental design](#experimental-design)
- [Names](#names)
- [Scoring](#scoring)
- [Statistical analysis](#statistical-analysis)
- [Validation on a model with a known bias](#validation-on-a-model-with-a-known-bias)
- [Power analysis](#power-analysis)

## Experimental design

The audit follows the **correspondence method** used by labour economists to measure
discrimination by human recruiters (Bertrand & Mullainathan, 2004; in France, the DARES /
ISM Corum testing campaigns): CVs identical except for the candidate's name and address.

Each **base profile** (job, experience, education, skills, languages) is crossed with every
cell of a full factorial design:

| Factor | Levels | Signalled by |
|---|---|---|
| Gender | female, male | first name |
| Perceived origin | majority French, Maghrebi, sub-Saharan African | first name and surname |
| Address | central Paris, deprived neighbourhood (QPV) | street, postcode and city |

That gives 12 versions of every profile. Each version is generated with 2 different names, so the
analysis can separate the effect of a *group* from the effect of one particular *name*. The audits
use 4 jobs (accountant, data analyst, payroll officer, retail department manager) × 10 profiles per
job = 40 profiles, × 12 versions × 2 names = 960 CVs. Each CV is therefore compared with the 23
other versions of its profile.

Guarantees, each enforced by a unit test (`tests/test_generator.py`):

- **Qualifications are held constant**: versions of the same profile differ only in name,
  address and email.
- **Gender-neutral wording**: job titles and descriptions are epicene French (*comptable*,
  *gestionnaire de paie*), so grammatical gender cannot reveal the treatment a second time.
- **Balanced design**: every cell has the same number of CVs.
- **Reproducibility**: the same seed always produces the same CVs.

## Names

First names are the 12 most frequent names given in France to children born 1985–2000 (people aged
about 25–40, a plausible candidate age) in each gender × origin group, according to the
[INSEE first-names file](https://www.insee.fr/fr/statistiques/8595130), among names typical of the
group. Names widely given across groups (e.g. *Sarah*, *Inès*, *Myriam*, *Issa*) are excluded, since
they would weaken the origin signal. Every name was given at least 425 times in that period;
[`scripts/check_first_names.py`](../scripts/check_first_names.py) re-checks this against the INSEE
file:

```bash
uv run python scripts/check_first_names.py path/to/prenoms-2025-nat.csv
```

INSEE does not publish surnames, so the 10 surnames per origin group are common surnames associated
with each group, chosen by judgement. Lists: [`names.json`](../src/llm_hiring_audit/data/names.json).

## Scoring

Each CV is scored from 1 to 100 under three French prompt wordings
([`prompts.py`](../src/llm_hiring_audit/prompts.py)), adapted from Armstrong et al. (2024):

- **rating**: rate the CV for the job posting;
- **interview**: likelihood of inviting the candidate to an interview;
- **hiring**: likelihood of hiring the candidate.

The audits ask each question once per CV at temperature 0.7, with a recorded seed: 960 CVs × 3
wordings = 2,880 scores per model.

- **Shuffled order**: queries are sent in random order, so that drift of the model or API during a
  long run cannot line up with one group.
- **Resumable**: each answer is appended to disk as it arrives; rerunning the command skips what is
  already done.
- **Refusals are kept**: an answer without a usable score is stored with `score: null`, and refusal
  rates are compared between groups.
- **Strict parsing**: the score is the first integer from 1 to 100 in the answer, ignoring ranges
  such as "de 1 à 100"; an ambiguous answer ("72 ou 80") gets no score.

## Statistical analysis

For each model, the effect of each signal, in score points, is estimated by OLS:

$$
\text{score}_{i} = \beta_1\,\text{female}_i + \beta_2\,\text{maghrebi}_i + \beta_3\,\text{subsaharan}_i
+ \beta_4\,\text{deprived}_i + \alpha_{\text{profile}(i)} + \gamma_{\text{prompt}(i)} + \varepsilon_i
$$

The reference candidate is a man with a majority French name living in central Paris.

- **Profile fixed effects** ($\alpha$) compare each CV only with the other versions of the same
  profile, so differences in qualifications cannot contaminate the estimates.
- **Prompt fixed effects** ($\gamma$) absorb level differences between the three wordings.
- **Standard errors clustered by first name and by surname** (two-way): all scores of one first
  name share that name's own effect, whatever surname it is paired with, and likewise for surnames.
  Treating them as independent would make confidence intervals too narrow.
- **Cluster bootstrap over profiles** (500 resamples) gives a second confidence interval that does
  not rely on the regression's distributional assumptions.
- **Holm correction** controls the family-wise error rate across the effects reported together.
- **Refusal rates** by group are compared with a chi-square test.
- **Ceiling check**: the share of scores at 90 or above, since a compressed scale can hide real
  differences.

Two further commands extend the main model:

- `robustness` re-estimates every effect within each prompt wording and each job (7 subsets × 4
  effects = 28 tests), with the Holm correction applied across all 28 together.
- `intersectional` adds gender × origin interaction terms, to test whether the origin effect differs
  between women and men, and reports the gap of each gender × origin group.

## Validation on a model with a known bias

Before trusting the analysis on real models, it is run on a mock model with planted penalties:
−6 points for Maghrebi surnames and −4 points for one of the three deprived-area addresses (Sevran),
so about −1.33 on average for the deprived area. 480 CVs × 3 prompts:

| Effect | Truth | Estimate [95% CI] | Significant (Holm) |
|---|---|---|---|
| Maghrebi-sounding name | −6 | −5.57 [−6.27, −4.86] | yes |
| Deprived-area address | ≈ −1.33 | −1.58 [−2.28, −0.89] | yes |
| Female | 0 | 0.56 [0.06, 1.05] | no (p = 0.056 after Holm) |
| Sub-Saharan-sounding name | 0 | 0.23 [−0.29, 0.76] | no |

Both planted effects are detected with the truth inside the interval, and nothing is found where
nothing was planted after the Holm correction. The uncorrected interval for "female" just misses
zero: a single 95% interval misses the truth 5% of the time, and with only 30 surname clusters,
cluster-robust intervals may be slightly too narrow (see
[decision 10](decisions.md#10-ols-with-profile-fixed-effects-errors-clustered-by-name)).

The same checks run automatically in [`tests/test_analysis.py`](../tests/test_analysis.py). To
reproduce the table:

```bash
uv run llm-hiring-audit generate --profiles-per-job 5 --out data/cvs.jsonl
uv run llm-hiring-audit score --backend mock --repetitions 1 \
    --mock-penalty Benali=-6 --mock-penalty Haddad=-6 --mock-penalty Bouzid=-6 \
    --mock-penalty Belkacem=-6 --mock-penalty Mansouri=-6 --mock-penalty Cherif=-6 \
    --mock-penalty Hamidi=-6 --mock-penalty Amrani=-6 --mock-penalty Benamar=-6 \
    --mock-penalty Saïdi=-6 --mock-penalty Sevran=-4 \
    --cvs data/cvs.jsonl --out data/scores_mock.jsonl
uv run llm-hiring-audit analyze --scores data/scores_mock.jsonl
```

## Power analysis

How many CVs does an audit need? `power` simulates many audits with the same design and analysis as
the real one, and reports how often a true effect is detected:

```bash
uv run llm-hiring-audit power --residual-sd 5.92 --name-sd 2.65 --effect 2
```

The noise inputs come from `analyze` on a 48-CV pilot run of each model (data in
[`results/2026-10/pilot/`](../results/2026-10/pilot/)). Mistral NeMo was barely affected by
individual names (residual sd 4.8, name sd 0.8) and reaches 100% power from 480 CVs. Mistral 7B
reacted much more to individual names (residual sd 5.9, name sd 2.7), and the simulation showed that
**the number of distinct names, not the number of CVs, limited the audit**: a first name keeps its
own quirk whatever CV it is put on, so with 5 first names per group, adding CVs kept reusing the
same few quirks.

Power to detect a 2-point gap for Mistral 7B (100 simulations each):

| Profiles per job | CVs | 5 first names per group | 12 first names per group |
|---|---|---|---|
| 5 | 480 | 64% | 76% |
| 10 | 960 | 62% | **90%** |
| 20 | 1,920 | 74% | 93% |

The name pool was therefore enlarged to 12 first names per group, and the audits use 10 profiles per
job (960 CVs).
