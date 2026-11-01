## 6. Limitations and Conclusions

*Generation workstream owner / section author: Harshita (@harshitadjainbdes24-byte); implementation and integration performed by Kausthuba Belagod (@kausthubabelagodbdes24-oss) on her behalf.*

### Core Findings Grounded in the Four Accepted Comparisons

Our evaluation of the twelve-key `scope30-v1` grid on seed 13 (`docs/analysis/scope30-v1-seed13/`) yielded empirical observations across all four planned experimental comparisons, covering 1,440 total selected synthetic examples (720 selected zero/few-shot texts across four artifacts and 720 AugGPT paraphrases across two artifacts):

#### 1. Synthetic-versus-Real Gap Across Tasks (C2, C3 vs C1)

| Benchmark | Condition | C1 Real Macro-F1 | Condition Macro-F1 | Absolute Difference | Relative Gap | Accuracy | Test Items Shift |
|---|---|---:|---:|---:|---:|---:|---|
| **TREC** | C2 Zero-shot | 0.6009 | 0.6257 | +0.0247 | **+4.12%** | 0.6680 (+0.0160) | +8 questions |
| **TREC** | C3 Few-shot | 0.6009 | 0.6411 | +0.0402 | **+6.69%** | 0.6880 (+0.0360) | +18 questions |
| **Emotion** | C2 Zero-shot | 0.4082 | 0.3804 | -0.0278 | **-6.80%** | 0.4305 (-0.0115) | -23 texts |
| **Emotion** | C3 Few-shot | 0.4082 | 0.3841 | -0.0241 | **-5.90%** | 0.4285 (-0.0135) | -27 texts |

- **Observed Directions:** On TREC, both synthetic-only training sets scored above the 180 real examples (C1: 0.6009 macro-F1, 0.6520 accuracy; C2: 0.6257 macro-F1, 0.6680 accuracy; C3: 0.6411 macro-F1, 0.6880 accuracy). On Emotion, both synthetic-only training sets scored below the real baseline (C1: 0.4082 macro-F1, 0.4420 accuracy; C2: 0.3804 macro-F1, 0.4305 accuracy; C3: 0.3841 macro-F1, 0.4285 accuracy).
- **Cross-Task Gap Difference:** The difference in relative gap (Emotion minus TREC) is **-10.92 percentage points** for zero-shot (C2) and **-12.59 percentage points** for few-shot (C3). This observed numerical direction is consistent with the planned expectation of a larger negative synthetic-versus-real gap on the more subjective task.
- **Contrary Outcome on TREC:** The observation that synthetic-only training scored above real training on TREC is contrary to the pattern observed on Emotion. This is a single-sample observation on one seed; its reliability is unknown, and it does not establish a robust or general property of synthetic text. Untested explanations (such as prototypicality of generated questions, the specific 180-row real sample, or the 9-item abbreviation class) were not evaluated here.

#### 2. Few-Shot versus Zero-Shot Synthetic Data (C3 − C2)

- **TREC:** C3 was higher than C2 on both metrics: **+0.0155 macro-F1** and **+0.0200 accuracy** (+10 test questions correct).
- **Emotion:** C3 was higher than C2 on macro-F1 (**+0.0037 macro-F1**) but lower on accuracy (**-0.0020 accuracy**, -4 test texts). The two evaluation metrics disagree in sign.
- **Reading:** With one seed, the reliability of these observed directions is unknown. Because no equivalence or meaningfulness threshold was defined in the study design, no claim of equivalence or practical equality is made.
- **Demonstration Costs:** Few-shot generation utilized real labeled demonstrations that zero-shot did not: **59** distinct real pool demonstrations for TREC and **54** for Emotion. These demonstration costs are disclosed separately from the 180 classifier training rows.

#### 3. Real + Few-Shot Mix versus Real-Only (C4 − C1)

- **TREC:** Replacing half of the real training rows with few-shot synthetic rows went with higher metric scores: **0.6638 macro-F1** and **0.7100 accuracy**, differing from C1 by **+0.0629 macro-F1** and **+0.0580 accuracy** (+29 questions correct).
- **Emotion:** C4 scored **0.4033 macro-F1** and **0.4425 accuracy**, differing from C1 by **-0.0049 macro-F1** and **+0.0005 accuracy** (+1 text correct). The two metrics disagree in sign.
- **Reading:** On TREC, C4 scored higher than C1 on both metrics. On Emotion, C4 scored lower on macro-F1 and higher on accuracy. With one seed, the reliability of these directions is unknown, and they cannot be separated from sampling variation.

#### 4. AugGPT Paraphrase Augmentation versus Low-Resource Real (C5b − C5a)

- **TREC:** C5b scored **0.6251 macro-F1** and **0.6700 accuracy**, differing from C5a (60 real sources; **0.5974 macro-F1**, **0.6400 accuracy**) by **+0.0277 macro-F1** and **+0.0300 accuracy** (+15 questions correct).
- **Emotion:** C5b scored **0.3432 macro-F1** and **0.3705 accuracy**, differing from C5a (60 real sources; **0.3381 macro-F1**, **0.3730 accuracy**) by **+0.0051 macro-F1** and **-0.0025 accuracy** (-5 texts correct). The two metrics disagree in sign.
- **Confounding with Size:** C5b has 420 training rows while C5a has 60. Both use the identical 60 real source texts. Because C5b provides a $7\times$ expansion in training rows without requiring additional human labels, the observed metric differences mix AugGPT paraphrase phrasing with the effect of larger sample size.
- **Reading:** C5b scored higher than C5a on both metrics on TREC, while on Emotion C5b scored higher on macro-F1 and lower on accuracy. With one seed, the reliability of these directions is unknown, and they are not attributable to AugGPT content alone.

---

### Overall Observed Directions

Observed metric directions on this single seed:
- **TREC:** C2, C3, and C4 each scored higher than C1, and C5b scored higher than C5a, on both macro-F1 and accuracy.
- **Emotion:** C2 and C3 scored lower than C1 on both metrics. C4 versus C1 and C5b versus C5a went in opposite directions on the two metrics.
- **Magnitude on Emotion:** Every Emotion difference from its corresponding reference condition was at most 0.0278 macro-F1 in absolute value.

With one seed, the reliability of these observed directions is unknown. No equivalence or meaningfulness threshold was defined in the experiment, so none of these differences is described as an improvement, an equivalence, or an absence of change.

---

### Scientific Boundaries and Retained Limitations

Every finding reported above is subject to strict scientific boundaries:

1. **Single-Seed and Single-Run Constraints**:
   All twelve condition results derive from a single seed (`seed 13`). We do not report standard deviations, error bars, or statistical confidence intervals, because no estimate of sampling variance exists for a single run. Differences of a few hundredths of macro-F1 or accuracy may reflect single-sample variation; a different random seed (a different real sample, generated corpus, or selection) could alter the magnitude or sign of any observed difference. We make **no claims of statistical significance, reproducibility across seeds, or stability**.

2. **Task Confounding (Non-Causal Comparison)**:
   TREC and Emotion differ across multiple dimensions beyond task subjectivity:
   - *Label Semantics*: TREC's six classes represent syntactic question types (abbreviation, entity, description, human, location, numeric), whereas Emotion's six classes represent psychological affective states (sadness, joy, love, anger, fear, surprise).
   - *Text Length and Style*: TREC consists of short factual questions, whereas Emotion consists of informal first-person tweets. (The 20- and 30-word limits in `config.yaml` apply to generated text, not to the real data.)
   - *Test Set Scale and Balance*: TREC has 500 test questions (with an `abbreviation` class of only 9 items, so a single abbreviation item can move macro-F1 by roughly 0.01–0.02, per the Issue #14 interpretation), whereas Emotion has 2,000 test texts (with majority class `joy` accounting for 35% of the split).
   - *Baseline Performance Level*: Baseline real-data performance on Emotion is low (0.4082 macro-F1; 0.4420 accuracy vs a 0.35 majority-class share), whereas TREC baseline performance sits at 0.6009 macro-F1.
   Consequently, **cross-task metric differences cannot be causally attributed to task subjectivity alone**.

3. **Separate Labeling and Demonstration Budgets**:
   Equal classifier training counts (180 rows in C1–C4) do not represent equal human labeling costs:
   - C1 uses 180 real labeled rows.
   - C2 uses 0 real labeled rows.
   - C3 uses 180 synthetic rows generated from **59** (TREC) and **54** (Emotion) real labeled demonstrations.
   - C4 uses 90 real rows plus 90 synthetic rows drawing on up to 59/54 real demonstrations.
   - C5b re-uses the 60 real sources from C5a to produce 360 paraphrases without additional human labeling.

4. **Model-Assisted Quality Review Boundaries**:
   Content audits relied on model-assisted screening (Claude Opus 5.5) and heuristic regex scans:
   - The initial gate sampled 10 rows in each of six task/setting batches (60 rows total) and did not model-review all 1,440 rows.
   - Subsequent evidence included TREC round 2 (10 fresh rows), targeted rechecks of round-1 IDs, the supplementary Emotion check (12 rows across two sources), and heuristic scope scans across all 360 TREC AugGPT paraphrases.
   - This process is **disclosed as model-assisted and heuristic screening, not independent human review, multi-annotator consensus, or a population error-rate estimate**.

5. **Disclosed Artifact Quality Flaws (Option A Freeze)**:
   Under lead Option A ([docs/freeze/README.md](../freeze/README.md)), the seed-13 corpora were frozen and accepted with known, disclosed defects:
   - **Emotion AugGPT**: All 12 paraphrases of two long sources (`emotion.train.07790` and `emotion.train.15223`) change the core propositional meaning (reversing agent relations or shifting personal attribution).
   - **TREC AugGPT (v2)**: Retains ~13 likely problematic rows (corrupted entity names like *"Hot and Wheels"*, label-word leaks like *"Kathie a human"*, one reversed relation, and ~3–4 rows shifting expected answer types).
   - **TREC Train/Test Duplicate**: Official benchmark duplicate `trec.train.05262 = trec.test.00378` (abbreviation) is present in the real training data and retained in C1 and C4 under a lead "pin + disclose" decision.
   - **Cross-Setting Overlap**: 18 identical normalized TREC texts appear in both selected zero-shot and few-shot corpora, reflecting limited lexical variety in question generation.

---

### Conclusion

In this exploratory single-seed comparison (`scope30-v1`), compact local LLM generation (`qwen3:4b`) produced synthetic training data that exhibited contrary numerical directions across the two evaluated tasks:
- On the objective TREC question classification task, synthetic-only training sets (C2, C3) and mixed/augmented sets (C4, C5b) scored higher than their respective real references (C1, C5a) on both macro-F1 and accuracy on seed 13.
- On the subjective Emotion benchmark, synthetic-only training sets (C2, C3) scored lower than the real reference (C1) on both macro-F1 and accuracy, while C4 versus C1 and C5b versus C5a exhibited opposite directions across the two metrics.

While the observed direction of the cross-task relative gap difference (-10.92 percentage points for C2, -12.59 for C3) aligns with the hypothesis that synthetic data exhibits larger negative gaps on subjective classification tasks relative to real data (Li et al., 2023), these single-seed observations do not establish causal attribution to task subjectivity, nor do they establish stability across random seeds or model architectures. With one seed, the reliability of all observed directions remains unknown, and no equivalence or meaningfulness threshold was defined. Additional seeds, larger generator models and controls for linguistic confounds were not part of this study and remain future work.
