# SynthText

Language for the project's comparison of real and LLM-generated classifier training data.

## Language

**Two-task comparison**:
A comparison of synthetic-data performance gaps on TREC and Emotion under the agreed experimental conditions. Differences can be consistent with the subjectivity hypothesis but do not establish subjectivity as their cause.
_Avoid_: causal test of subjectivity, tasks differing only in subjectivity

**Subjectivity hypothesis**:
The prediction that synthetic-only training has a more negative performance gap on Emotion than on TREC. It is a prediction to assess against results, including results that contradict it.
_Avoid_: guaranteed result, proven subjectivity effect

**Generation replicate**:
A separately generated synthetic dataset used for one experimental repetition. Reusing or reshuffling a single generated dataset does not constitute another generation replicate.
_Avoid_: independent dataset when only the row order changed

**Experimental repetition**:
One seed-identified comparison across the six training conditions, using that repetition's separately generated zero-shot, few-shot and AugGPT data and its matched real-data samples. Repetitions share the fixed held-out real test set.
_Avoid_: additional test split, independent test set per seed

**In-repository evaluation**:
Evaluation conducted directly within the repository using reproducible artifacts and scripts, with evaluation evidence and findings documented in `docs/REPORT.md`. Supersedes independent human annotation and external report sharing (ADR 0012) under the sole-operator execution model.
_Avoid_: claiming independent human consensus when evaluations are performed in-repo by the operator or agent

**Independent annotation submission (historical / superseded)**:
Originally, one annotator's judgments on the common 60-item set, completed without seeing other annotators' responses or the gold answers (ADR 0005). Superseded by ADR 0012, which removes the independent-human annotation requirement and external distribution gates.

**Equal classifier training-set size**:
The same number of examples supplied to the classifier in C1-C4 for a given task and repetition. This excludes labeled examples used only as generation demonstrations and does not imply equal human-labeling cost.
_Avoid_: equal labeling budget, label-free few-shot generation

**Paired relative gap**:
The difference between a synthetic-only condition's macro-F1 and its matching repetition's real-only C1 macro-F1, divided by that C1 macro-F1. Each task and synthetic-only condition is summarized by the mean and standard deviation of its three repetition-level gaps.
_Avoid_: relative gap computed only after averaging baseline and synthetic scores
