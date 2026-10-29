# Audit results, October 2026

Everything needed to check or rerun the analysis of the October 2026 audit without scoring the CVs
again (the scoring took about 5 hours for Mistral 7B and 26 minutes for Mistral NeMo).

## Models

| Model id in the files | Model | Access | Version |
|---|---|---|---|
| `ollama/mistral` | Mistral 7B Instruct | local, Ollama 0.40.11 | `mistral:latest`, digest `6577803aa9a0` |
| `mistral/open-mistral-nemo` | Mistral NeMo | Mistral API, free tier | `open-mistral-nemo`, as served on 2026-10-28 |

## Files

| Folder | File | Contents |
|---|---|---|
| `raw/` | `cvs.jsonl` | The 960 audit CVs: design variables (profile, job, gender, origin, address, name) and the CV text |
| | `scores_<model>.jsonl` | 2,880 answers per model: CV id, prompt, raw answer, parsed score (`null` when unusable) |
| `tables/` | `effects_<model>.csv` | Main effects with 95% CIs, Holm-corrected p-values and bootstrap CIs |
| | `refusals_<model>.csv` | Share of answers without a usable score, by group, with a chi-square test |
| | `subgroups_<model>.csv` | Effects within each prompt wording and each job, Holm-corrected across all 28 tests |
| | `intersectional_<model>.csv` | Gap of each gender × origin group, and gender × origin interactions |
| `pilot/` | `cvs.jsonl`, `scores_<model>.jsonl` | The 48-CV pilot whose noise estimates fed the power analysis (Mistral NeMo pilot: 2 repetitions) |

## Reproduce

From the repository root:

```bash
# Re-create the CVs (identical to raw/cvs.jsonl)
uv run llm-hiring-audit generate --profiles-per-job 10 --seed 2026 --out data/cvs.jsonl

# Rebuild every table from the raw scores
for model in ollama_mistral mistral_open-mistral-nemo; do
  scores=results/2026-10/raw/scores_$model.jsonl
  uv run llm-hiring-audit analyze --scores $scores --n-boot 500 --out-dir results/2026-10/tables
  uv run llm-hiring-audit robustness --scores $scores --out-dir results/2026-10/tables
  uv run llm-hiring-audit intersectional --scores $scores --out-dir results/2026-10/tables
done
```

The scores were produced with:

```bash
uv run llm-hiring-audit score --backend ollama --model mistral --repetitions 1 \
    --cvs data/cvs.jsonl --out results/2026-10/raw/scores_ollama_mistral.jsonl
uv run llm-hiring-audit score --backend mistral --model open-mistral-nemo --repetitions 1 \
    --cvs data/cvs.jsonl --out results/2026-10/raw/scores_mistral_open-mistral-nemo.jsonl
```

Rescoring would not give identical answers: models are sampled at temperature 0.7, and an API
model can change behind the same name.

The tables use two-way clustering by first name and surname, adopted on 2026-10-29 (see
[decision 10](../../docs/decisions.md#10-ols-with-profile-fixed-effects-errors-clustered-by-name)).
