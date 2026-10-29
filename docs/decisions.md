# Decision log

The main choices behind this audit: what was decided, what else was considered, and why.
Each entry ends with the trade-off accepted, because every choice gave something up.

| # | Decision | In one line |
|---|---|---|
| 1 | [Correspondence audit](#1-measure-bias-with-a-correspondence-audit) | Measure what the model does, not what it says |
| 2 | [Full factorial design within each profile](#2-a-full-factorial-design-within-each-profile) | Every CV exists in all 12 versions |
| 3 | [Names and address as the only signals](#3-names-and-address-as-the-only-signals) | Gender-neutral French wording, enforced by a test |
| 4 | [Names from INSEE birth data](#4-first-names-taken-from-insee-birth-data) | A documented rule instead of personal judgement |
| 5 | [A 1-100 score under three wordings](#5-a-1100-score-asked-three-ways) | No single prompt is trusted alone |
| 6 | [Temperature 0.7, one answer per question](#6-temperature-07-and-one-answer-per-question) | Typical behaviour; extra repetitions don't buy power |
| 7 | [Two free Mistral models](#7-two-mistral-models-run-for-free) | Feasible at zero cost; frontier models left out |
| 8 | [Shuffled, resumable runs that keep refusals](#8-shuffled-resumable-runs-that-keep-refusals) | Long runs must not bias or lose data |
| 9 | [Strict score parsing](#9-strict-score-parsing) | An unclear answer gets no score rather than a guess |
| 10 | [OLS with fixed effects, clustered by name](#10-ols-with-profile-fixed-effects-errors-clustered-by-name) | Simple, assumption-light estimation; revised to two-way clustering |
| 11 | [Holm correction](#11-holm-correction-for-multiple-testing) | Control false alarms across all tests reported together |
| 12 | [Sample size from a simulation](#12-sample-size-chosen-by-simulation-before-the-audit) | Decide how many CVs before spending hours of compute |
| 13 | [Validate on a planted bias first](#13-validate-the-pipeline-on-a-planted-bias-first) | Prove the method finds a known truth |
| 14 | [Report effect sizes with intervals](#14-report-effect-sizes-with-intervals-not-passfail) | A null result needs an interval to mean anything |
| 15 | [Engineering choices](#15-engineering-choices) | Typed Python, swappable backends, offline tests, committed raw data |

---

## 1. Measure bias with a correspondence audit

**Decision.** Send the model CVs that are identical except for the candidate's name and
address, and compare its scores, as labour economists do with human recruiters
(Bertrand & Mullainathan, 2004; the DARES / ISM Corum testing campaigns in France).

**Alternatives.**
- *Embedding association tests (WEAT-style).* They measure associations inside the model,
  which are a weak guide to what it actually decides.
- *Asking the model whether it is biased.* Measures what it says, not what it does.
- *Real CVs.* Qualifications would differ between groups and confound the comparison.

**Why.** It measures the behaviour that matters for the AI Act use case (scoring
applicants), and it is a method economists and regulators already trust.

**Trade-off.** Synthetic CVs are less realistic than real ones.

## 2. A full factorial design within each profile

**Decision.** Each base profile is copied into all 12 combinations of gender (2) × perceived
origin (3) × address (2), each with 2 different names: 40 profiles (10 per job), 960 CVs.

**Alternatives.** Matched pairs (one comparison at a time); random assignment of one version
per profile.

**Why.** Every profile serves as its own control for every comparison, and the effects of the
three signals (and their combinations) can be estimated together from one dataset.

**Trade-off.** More queries per profile, so fewer distinct profiles for a given budget.

## 3. Names and address as the only signals

**Decision.** Gender and origin are signalled only by the name; residence only by the address.
All job titles and descriptions use epicene French (*comptable*, *gestionnaire de paie*).
A unit test fails if a gendered form appears.

**Alternatives.** Stating gender or origin explicitly; leaving grammatical gender free.

**Why.** Explicit labels are not how discrimination happens in practice, and French grammar
would otherwise reveal gender a second time (*employé* / *employée*). The test exists because
the first draft of the job data contained masculine forms.

**Trade-off.** A narrower vocabulary for the CVs.

## 4. First names taken from INSEE birth data

**Decision.** For each gender × origin group, the 12 most frequent first names given in France
to children born 1985-2000 (INSEE first-names file), excluding names widely shared across
groups (Sarah, Inès, Issa...). 10 surnames per origin group.

**Alternatives.** Names chosen by intuition; a small fixed list (the first version used 5).

**Why.** A documented, checkable rule (`scripts/check_first_names.py` re-verifies every name)
and enough distinct names for the results to describe groups rather than a few names
(see decision 12).

**Trade-off.** Surnames are chosen by judgement, since INSEE publishes no surname statistics.
Excluding ambiguous names makes the signal clearer but less representative of everyone.

## 5. A 1-100 score, asked three ways

**Decision.** The model rates each CV from 1 to 100 under three wordings: rate the CV, likelihood
of an interview, likelihood of hiring. Prompts are in French. Adapted from Armstrong et al.
(2024), *The Silicon Ceiling*.

**Alternatives.** A yes/no shortlist decision; ranking several CVs at once.

**Why.** A score detects small gaps that a yes/no decision would hide, and LLMs are sensitive to
wording, so three prompts guard against a result that depends on one phrasing. Ranking CVs side
by side would add order effects.

**Trade-off.** A score is not a hiring decision; real tools may rank or filter instead.

## 6. Temperature 0.7 and one answer per question

**Decision.** Sample at temperature 0.7 with a recorded seed, and ask each question once per CV.

**Alternatives.** Temperature 0 (most likely answer only); several repetitions per question.

**Why.** Sampling reflects how the model behaves across typical answers rather than a single
most-likely one. Repetitions were dropped because the power analysis showed that precision was
limited by the number of distinct names, not by answer-to-answer noise.

**Trade-off.** Results describe behaviour at this temperature. Some newer models do not let you
set it, so a future audit of them would use the provider's default.

## 7. Two Mistral models, run for free

**Decision.** Mistral 7B Instruct run locally with Ollama, and Mistral NeMo through Mistral's
free API tier.

**Alternatives.** A paid frontier model (GPT, Gemini, Claude). Gemini's free tier was tried, but
it allows only 20 requests per day per model.

**Why.** Zero cost, and both models are relevant to a European deployment. Two models from one
provider is also a fair test of the claim "same provider, same behaviour".

**Trade-off.** The audit says nothing about the frontier models most used in industry. The
Gemini backend is built and tested for that extension.

## 8. Shuffled, resumable runs that keep refusals

**Decision.** Queries are sent in random order; each answer is appended to disk immediately and
a rerun skips what is done; an answer without a usable score is kept with `score = None`.

**Alternatives.** Run in file order; keep results in memory; drop refusals.

**Why.** A model or API that drifts during a 5-hour run must not line up with one group;
a crash must not lose hours of work; and a model that refuses some groups more often
discriminates too, which dropping refusals would hide (refusal rates are tested by group).

**Trade-off.** Slightly more code than a simple loop.

## 9. Strict score parsing

**Decision.** Take the first integer from 1 to 100 in the answer, ignoring ranges ("de 1 à 100",
"entre 60 et 70"); an ambiguous answer ("72 ou 80") gets no score.

**Alternatives.** Take the first number found; ask the model for structured JSON output.

**Why.** Testing on real outputs showed that "first number" would read "sur une échelle de 1 à
100, je donne 72" as 1, which would silently corrupt results. Structured output is not supported
the same way by every backend.

**Trade-off.** Some answers a human could interpret are recorded as missing; refusal rates show
how many.

## 10. OLS with profile fixed effects, errors clustered by name

**Decision.** For each model, regress the score on dummies for female, Maghrebi, sub-Saharan
and deprived address, with profile and prompt fixed effects. Standard errors are clustered by
first name and by surname (see "Revised" below). A cluster bootstrap over profiles (500 resamples) gives a second interval.

**Alternatives.** A mixed model with a random effect per name (as in *The Silicon Ceiling*);
plain OLS errors.

**Why.** Fixed effects compare each CV only with the other versions of the same profile and need
no assumption about how profiles are distributed. Scores given to one name share that name's
quirk, so they are not independent; plain errors would be too narrow.

**Trade-off.** The score is treated as a continuous number, although it is bounded and bunches
near the top (18% of Mistral 7B scores are 90 or above).

**Revised (two-way clustering).** The first version clustered by full name. But a first name
appears with several surnames, so its quirk is shared across those pairings, and clustering by
full name misses that. Errors are now clustered two ways, by first name and by surname. This was
decided on the structure of the data before looking at how it changed the results, and adopted
whatever it showed. A test checks that two-way intervals are wider when names have strong
quirks.

On the real audits, standard errors moved by about ±15%, in both directions, and no pooled
conclusion changed. Within subgroups, the intervals mostly narrowed: the Mistral 7B sub-Saharan
penalty became significant in 7 of 7 subsets (from 6) and the NeMo female premium in 6 of 7
(from 3). Subgroup counts are therefore reported with both versions. The full-name version remains available (`clustering="full_name"`) for comparison.

**Remaining caveat.** Origin is constant within a surname, so origin effects are estimated from
only 10 surnames per group (30 surname clusters). Cluster-robust errors can be unreliable with so
few clusters; a wild cluster bootstrap would be the stricter check.

## 11. Holm correction for multiple testing

**Decision.** p-values are Holm-corrected across all tests reported together: the 4 main effects,
or all 28 subgroup tests at once in the robustness check.

**Alternatives.** No correction; Bonferroni; Benjamini-Hochberg (controls the false discovery rate).

**Why.** Each effect is presented as an individual claim, so the chance of any false alarm should
be controlled (family-wise error rate). Holm does this and is never less powerful than
Bonferroni. A first robustness check corrected within each subgroup separately and overstated
two claims; correcting across all 28 tests fixed it.

**Trade-off.** Less power for small effects in subgroups: Mistral NeMo's female premium is positive
in all 7 subgroups but significant in only 6 (only 3 under the earlier full-name clustering).

## 12. Sample size chosen by simulation before the audit

**Decision.** A 48-CV pilot measured each model's noise; a simulation of the full design and
analysis then estimated the chance of detecting a 2-point gap for each sample size.

**Alternatives.** A textbook power formula; picking a "large enough" number.

**Why.** Formulas ignore name clustering and multiple-testing correction; simulation runs the real
analysis. It also showed that with 5 first names per group, power stalled between 62% and 74%
whatever the sample size, which led to decision 4. With 12 names, 960 CVs give about 90%.

**Trade-off.** Noise estimates from a small pilot are rough. The simulation's first version
treated each full name as independent and was too optimistic; it now models first-name and
surname effects separately.

## 13. Validate the pipeline on a planted bias first

**Decision.** A mock model adds known penalties (−6 for Maghrebi surnames, −4 for one suburb) to
noisy scores; tests check the analysis recovers them and flags nothing when no bias is planted.

**Alternatives.** Trust the code and go straight to real models.

**Why.** A method that cannot find a known effect cannot be trusted on an unknown one. These checks
run automatically in the test suite.

**Trade-off.** The mock is simpler than a real model, so passing it is necessary, not sufficient.

## 14. Report effect sizes with intervals, not pass/fail

**Decision.** Every result is an effect in score points with a 95% confidence interval.

**Alternatives.** Report only whether each effect is significant.

**Why.** "No significant effect" means little on its own. The interval is what makes Mistral NeMo's
null result informative: it rules out origin or address penalties larger than about 0.7 points.

**Trade-off.** Results take more effort to read than a pass/fail label.

## 15. Engineering choices

**Decision.** Python 3.12 with strict type checking (mypy), pydantic data models, a Typer
command-line interface, one interface shared by all model backends, HTTP calls mocked in tests,
continuous integration on GitHub Actions, and raw audit data committed in `results/`.

**Why.** Every claim in the README can be reproduced from a tested command; tests cost nothing and
need no network; a new provider is one small class; and anyone can rerun the analysis without the
5 hours of scoring.

**Trade-off.** More setup than a notebook, which would have been faster for a one-off study.
