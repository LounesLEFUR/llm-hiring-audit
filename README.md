# llm-hiring-audit

**Do large language models discriminate when they screen French CVs?**

This project measures it with the method labour economists use to measure discrimination
by human recruiters: the **correspondence audit study** (Bertrand & Mullainathan, 2004;
in France, the DARES / ISM Corum testing campaigns). We send a model CVs that are
identical except for the candidate's name and address, and test whether its scores differ.

> Status: work in progress. CV generation and model scoring are done; the statistical
> analysis is next.

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
- [ ] Analysis: mixed-effects regression, bootstrap confidence intervals, multiple-testing
      correction, power analysis
- [ ] Validation: recover a planted bias from the mock model
- [ ] Results report and write-up

## Related work

- Bertrand & Mullainathan (2004), *Are Emily and Greg More Employable than Lakisha and
  Jamal?* The original correspondence study of hiring discrimination.
- Armstrong et al. (2024), [*The Silicon Ceiling: Auditing GPT's Race and Gender Biases in
  Hiring*](https://arxiv.org/abs/2405.04412). Inspired the three prompt wordings and the
  mixed-effects model with a random effect for names.
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
