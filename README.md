# llm-hiring-audit

[![CI](https://github.com/LounesLEFUR/llm-hiring-audit/actions/workflows/ci.yml/badge.svg)](https://github.com/LounesLEFUR/llm-hiring-audit/actions/workflows/ci.yml)

**Do large language models discriminate when they screen French CVs?**

This project measures it with the method labour economists use for human recruiters, the
**correspondence audit**: send a model CVs that are identical except for the candidate's name and
address, and test whether its scores differ. It turns a requirement of the EU AI Act, examining
hiring systems for discriminatory bias, into a reproducible, quantitative test.

## Key results

960 matched CVs × 3 prompt wordings = 2,880 scores per model, on a 1–100 scale (October 2026).
Effects are in score points relative to a man with a majority French name living in central
Paris, with 95% confidence intervals. \* = significant after Holm correction.

| Effect | Mistral 7B (local) | Mistral NeMo (API) |
|---|---|---|
| Sub-Saharan-sounding name | **−2.02** [−2.60, −1.44]\* | +0.01 [−0.54, 0.57] |
| Deprived-area address | **−1.09** [−1.60, −0.59]\* | +0.21 [−0.20, 0.62] |
| Maghrebi-sounding name | **−0.94** [−1.46, −0.43]\* | −0.16 [−0.68, 0.35] |
| Female | **+2.43** [1.89, 2.97]\* | **+1.21** [0.86, 1.56]\* |
| Answers without a usable score | 0.5%, no group difference | 0.1%, no group difference |

- **Two models from the same provider behave in opposite ways.** Mistral 7B penalises
  sub-Saharan-sounding names, Maghrebi-sounding names and deprived-area addresses. Mistral NeMo
  shows none of these penalties, and its intervals rule out any larger than about 0.7 points.
- **Both models favour women.** For Mistral 7B, the premium goes mostly to women with majority
  French names (+3.3 points); women with sub-Saharan-sounding names get +0.5 (not significant).
- **The largest effects hold in every prompt wording and every job**: for Mistral 7B, the
  sub-Saharan penalty and the female premium are significant in all 7 subsets. The smaller
  Maghrebi and address penalties are confirmed overall but not in every subset. A cluster bootstrap
  confirms every overall result.

The gaps are small on a 100-point scale but systematic: a tool that shortlists the top-scored
candidates turns a steady 2-point gap into fewer interviews for the penalised group. Results apply
to these model versions, prompts and four jobs, and should be re-run on each new model version.
Full tables and raw data: [`results/2026-10/`](results/2026-10/).

## Why it matters

AI used to filter job applications is **high-risk** under the EU AI Act (Regulation (EU)
2024/1689, Annex III, point 4(a)): providers must examine such systems for biases that could lead
to discrimination (Article 10). French law prohibits hiring discrimination based on origin, sex or
place of residence (Code du travail, art. L1132-1). Most published LLM audits use English CVs and
US names; this one uses French CVs, the origin groups relevant to the French labour market, and
place of residence.

## How it works

1. **Generate matched CVs.** 40 base profiles across 4 jobs, each in 12 versions (gender × 3
   perceived origins × 2 addresses) with 2 names each. First names come from INSEE birth data;
   all wording is gender-neutral French, so the name is the only gender signal.
2. **Score them.** Each CV is rated 1–100 under three wordings (rate, interview, hire), in random
   order. Unusable answers are kept and analysed as refusals.
3. **Estimate the effects.** OLS with profile and prompt fixed effects, standard errors clustered
   by first name and surname, a cluster bootstrap, and Holm correction for multiple tests.
4. **Check the method.** A mock model with planted biases verifies that the analysis recovers them,
   and a simulation fixed the sample size in advance (90% power to detect a 2-point gap).

Details: [docs/methodology.md](docs/methodology.md). Why each choice was made, and what it cost:
[docs/decisions.md](docs/decisions.md).

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/LounesLEFUR/llm-hiring-audit.git
cd llm-hiring-audit
uv sync

# Rebuild the published results from the committed raw scores (no model needed)
uv run llm-hiring-audit analyze --scores results/2026-10/raw/scores_ollama_mistral.jsonl

# Or run a new audit end to end, here with a local model via Ollama
uv run llm-hiring-audit generate --profiles-per-job 10 --seed 2026
uv run llm-hiring-audit score --backend ollama --model mistral --repetitions 1
uv run llm-hiring-audit analyze
```

Generated files go to `data/`, which is not tracked by git.

| Command | What it does |
|---|---|
| `generate` | Write matched CVs to a JSONL file |
| `score` | Have a model score every CV (`--backend ollama`, `mistral`, `gemini` or `mock`); resumable |
| `analyze` | Main effects, bootstrap, refusal rates and score distribution |
| `robustness` | Effects within each prompt wording and each job |
| `intersectional` | Gender × origin groups and interactions |
| `power` | Simulated power for a given design and noise level |

API backends read their key from `MISTRAL_API_KEY` or `GEMINI_API_KEY`. Run
`uv run llm-hiring-audit <command> --help` for every option.

## Project structure

```
llm-hiring-audit/
├── src/llm_hiring_audit/
│   ├── models.py        Data models: profile, treatment, identity, CV
│   ├── generator.py     Matched CV generation (factorial design)
│   ├── render.py        CV as French plain text
│   ├── prompts.py       The three scoring prompts
│   ├── backends.py      Ollama, Mistral, Gemini and mock model clients
│   ├── scoring.py       Shuffled, resumable scoring and score parsing
│   ├── analysis.py      Regression, bootstrap, Holm, robustness, intersectional
│   ├── power.py         Simulation-based power analysis
│   ├── cli.py           Command-line interface
│   └── data/            Job postings and name lists
├── tests/               Unit tests (no network: model calls are mocked)
├── scripts/             Check of first names against INSEE data
├── results/2026-10/     Raw scores and result tables of the October 2026 audit
└── docs/                Methodology and decision log
```

## Development

```bash
uv run pytest                     # tests and coverage
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests scripts     # strict type checking
```

The same checks run on every push (GitHub Actions).

## Limitations

- Names and addresses signal *perceived* group membership; they say nothing about real people.
- Two open-weight models from one provider, chosen because both could be run for free. The audit
  does not show how the frontier models most used in industry behave; the Gemini backend is ready
  for that extension.
- Four jobs in the Paris region, one answer per question at temperature 0.7. A 1–100 score is not
  a hiring decision: real tools may rank, filter or summarise instead.
- Origin effects rest on 10 surnames per group (30 clusters), so their intervals may be slightly
  too narrow ([decision 10](docs/decisions.md#10-ols-with-profile-fixed-effects-errors-clustered-by-name)).

## Related work

- Bertrand & Mullainathan (2004), [*Are Emily and Greg More Employable than Lakisha and
  Jamal?*](https://doi.org/10.1257/0002828042002561) The original correspondence study of hiring
  discrimination.
- Armstrong et al. (2024), [*The Silicon Ceiling: Auditing GPT's Race and Gender Biases in
  Hiring*](https://arxiv.org/abs/2405.04412). Source of the three prompt wordings.
- Wilson & Caliskan (2024), [*Gender, Race, and Intersectional Bias in Resume Screening via
  Language Model Retrieval*](https://arxiv.org/abs/2407.20371).
- Tamkin et al. (2023), [*Evaluating and Mitigating Discrimination in Language Model
  Decisions*](https://arxiv.org/abs/2312.03689).
- Human recruiters in France: the DARES / ISM Corum testing campaigns and the
  [IPP policy brief](https://www.ipp.eu/en/news/discrimination-in-hiring-in-france-findings-and-courses-of-action)
  on hiring discrimination.

## Author

Lounès Le Fur, MSc Artificial Intelligence and Society (ENS-PSL / Dauphine-PSL).
