# Interpretation: scope30-v1, seed 13 single-seed gap analysis (issue #14)

**What this is.** A written reading of the four committed analysis outputs in
[`scope30-v1-seed13/`](scope30-v1-seed13/): [`summary.md`](scope30-v1-seed13/summary.md),
[`analysis.json`](scope30-v1-seed13/analysis.json), [`f1_by_condition.png`](scope30-v1-seed13/f1_by_condition.png)
and [`gap.png`](scope30-v1-seed13/gap.png). It adds no new numbers beyond simple conversions of differences into
test-set example counts. It was drafted with model assistance (Claude Opus 5.5) for the lead's review. It is not an
independent human analysis.

**Provenance.**
- **How the outputs were made:** written once by `python -m ws3_evaluation.analyze_scope30` from a clean checkout of
  the PR #34 merge `c7761ec944d86e866ce1103ed2b90ecf57b3ac8c`. The operator ran `run-issue14-analysis.bat WRITE` on
  1 November 2026.
- **What they were made from:** the twelve-key grid that the lead accepted on issue #13, produced at `1354118` with
  `grid_complete.json` sha256 `3b12bbb878d8191e387e3abc0f77fc02d9f081b98f0216f9b64dbb144bf40032`. They also use the
  freeze `scope30-v1-seed13-freeze-v1` (`dd2e52ee…`) and frozen MiniLM `1110a243…` with LogisticRegression.
- **Committed byte for byte.** `tests/test_analysis_outputs_scope30.py` pins their SHA-256s and regenerates
  `summary.md` and every gap, comparison and cost from `analysis.json`.

## How to read these numbers

- **One seed, one run per key.** There is no standard deviation, interval, significance test or stability estimate,
  and none of the statements below claims one. A different seed (a different real sample, generated corpus or
  selection) could change any of these differences, including their sign.
- **Test-set scale.**
  - TREC has 500 test questions, so one question is 0.002 accuracy. Its `abbreviation` class has only 9 test items
    but carries 1/6 of macro-F1, so a single abbreviation item can move macro-F1 by roughly 0.01–0.02.
  - Emotion has 2,000 test texts, so one text is 0.0005 accuracy. Its `surprise` class has 66 items.
- **Absolute level.** All Emotion conditions score between 0.34 and 0.41 macro-F1, and 0.37–0.44 accuracy against
  a 0.35 majority-class (`joy`) share. TREC scores 0.60–0.66 macro-F1. Emotion differences therefore sit on a weak
  base for this frozen-embedding, 30-per-label classifier.

## The four planned comparisons

### 1. Synthetic-only versus real-only gap across tasks (C2, C3 vs same-task C1)

| | TREC | Emotion |
|---|---|---|
| C2 zero-shot vs C1 | +4.12% relative (+0.0247 macro-F1; +8 test questions correct) | −6.80% (−0.0278; −23 test texts) |
| C3 few-shot vs C1 | +6.69% (+0.0402; +18 questions) | −5.90% (−0.0241; −27 texts) |

- **Observed.** On TREC, both synthetic-only training sets scored above 180 real examples. On Emotion, both
  scored below. The Emotion-minus-TREC difference in relative gap is −10.92 percentage points for C2 and
  −12.59 for C3.
- **Consistent with the planned expectation.** The plan expected a larger negative synthetic-versus-real gap on
  the more subjective task, and that is the direction seen here, in both settings.
- **What this does not show:**
  - **Not the cause.** It does not show that subjectivity causes the difference. The tasks also differ in label
    semantics, text length and style, test-set size (500 vs 2,000) and label balance.
  - **Not a robust result.** Real data being *worse* than synthetic data on TREC is a single-sample observation.
  - **Untested explanations.** Possible explanations include more prototypical generated questions per answer type,
    the 180-row real sample, and the 9-item abbreviation class. None of them was tested here.

### 2. Few-shot versus zero-shot synthetic data (C3 − C2)

- **TREC:** +0.0155 macro-F1 and +0.0200 accuracy (+10 questions).
- **Emotion:** +0.0037 macro-F1 but −0.0020 accuracy (−4 texts). The two metrics disagree in sign, and the
  differences are a handful of test texts.
- **Reading.** On TREC, C3 was higher than C2 on both metrics. On Emotion, C3 was higher on macro-F1 and lower on
  accuracy. With one seed, the reliability of these directions is unknown, and no equivalence threshold was
  defined, so no claim of equivalence is made.
- **Cost.** Few-shot used real labelled demonstrations that zero-shot did not: 59 distinct real demonstrations
  for TREC and 54 for Emotion. These are disclosed separately from the 180 training rows.

### 3. Real plus few-shot mix versus real-only (C4 − C1)

- **TREC:** +0.0629 macro-F1 and +0.0580 accuracy (+29 questions). This is the largest within-task difference
  in the grid. C4 has the same 180 training rows as C1, but only 90 of them are real, plus up to 59 real
  demonstrations behind the 90 synthetic rows.
- **Emotion:** −0.0049 macro-F1 and +0.0005 accuracy (+1 text). The two metrics disagree in sign.
- **Reading.** Replacing half of the real rows with few-shot synthetic rows went with higher macro-F1 and accuracy
  on TREC. On Emotion it went with lower macro-F1 and higher accuracy. With one seed, the reliability of these
  directions is unknown, and they cannot be separated from sampling variation.

### 4. AugGPT-augmented versus real sources only (C5b − C5a)

- **TREC:** +0.0277 macro-F1 and +0.0300 accuracy (+15 questions).
- **Emotion:** +0.0051 macro-F1 and −0.0025 accuracy (−5 texts). Again the metrics disagree in sign, and the
  difference is a few texts.
- **Confounded with size.** C5b has 420 training rows and C5a has 60. Both use the same 60 real sources, so no
  extra real labels are needed. The difference mixes AugGPT content with simply having more rows.
- **Known quality flags.** The retained AugGPT data carries disclosed issues:
  - about 13 likely problematic TREC paraphrases out of 360;
  - all 12 Emotion paraphrases of two sources change their meaning.

  Both are listed in the freeze manifest and in issue #11.
- **Reading.** C5b was higher than C5a on both metrics on TREC. On Emotion it was higher on macro-F1 and lower on
  accuracy. With one seed, the reliability of these directions is unknown, and the size difference means they are
  not attributable to AugGPT content alone.

## Overall

Observed directions on this single seed:
- **TREC:** C2, C3 and C4 each scored higher than C1, and C5b higher than C5a, on both macro-F1 and accuracy.
- **Emotion:** C2 and C3 scored lower than C1 on both metrics. C4 versus C1 and C5b versus C5a went in opposite
  directions on the two metrics.
- **Size of the Emotion differences:** every Emotion difference from its reference was at most 0.0278 macro-F1 in
  absolute value.

With one seed, the reliability of every one of these directions is unknown. No equivalence or meaningfulness
threshold was defined, so none of them is described as a gain, a loss or no change.

These are exploratory, single-run observations for scope30-v1 (ADR 0012). They do not establish:
- statistical significance, stability or generalisation;
- that subjectivity is the cause of the cross-task difference;
- synthetic-data quality beyond the disclosed model-assisted review.

## Limitations

The full list is in [`summary.md`](scope30-v1-seed13/summary.md#limitations). In brief:
- **Single seed and run.** One seed and one run per key.
- **Small data.** 30 examples per label (10 sources per label for C5).
- **Unequal C5 sizes.** C5a and C5b differ in training size by design.
- **Task confounding.** The two tasks differ in more than subjectivity.
- **Separate labelling costs.** Demonstration and source costs are separate from training counts.
- **Model-assisted review.** Synthetic-data quality was screened by model-assisted and heuristic review, not
  independent humans. Flagged rows were retained as disclosed.
- **One disclosed duplicate.** One official TREC train/test duplicate (`trec.train.05262 = trec.test.00378`) is in
  C1 and C4, by lead decision on issue #13.
