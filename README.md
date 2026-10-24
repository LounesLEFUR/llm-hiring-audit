# llm-hiring-audit

**Do large language models discriminate when they screen French CVs?**

This project measures it with the method labour economists use to measure discrimination
by human recruiters: the **correspondence audit study** (Bertrand & Mullainathan, 2004;
in France, the DARES / ISM Corum testing campaigns). We send a model CVs that are
identical except for the candidate's name and address, and test whether its scores differ.

> Status: work in progress. Phase 1 (CV generation) is done; model scoring and the
> statistical analysis are next.

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

## Development

```bash
uv run pytest          # tests + coverage
uv run ruff check .    # lint
uv run mypy src tests  # strict type checking
```

## Roadmap

- [x] Matched CV generator (factorial design)
- [ ] Model backends: local models (Ollama) and Mistral API, with mocked tests
- [ ] Analysis: fixed-effects regression, bootstrap confidence intervals, multiple-testing
      correction, power analysis
- [ ] Robustness across prompt wordings
- [ ] Results report and write-up

## Limitations

Names and addresses are *signals of perceived* group membership, not identities. The
results describe how a model reacts to these signals in this specific setting.
