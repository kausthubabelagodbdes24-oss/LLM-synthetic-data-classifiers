"""Tests for Issue #16: Data, classifier, and training methods specification and report sections.

Verifies that docs/report/methods_data_classifiers.md, docs/report/README.md, and CONTRACTS.md
strictly adhere to scope30-v1 specifications, match all accepted repository evidence (freeze manifest,
analysis outputs, and training harness constants), observe all scientific boundaries, and maintain
accurate model-identity hash terminology without contradictory leakage claims.
"""

import json
from pathlib import Path

from common import ROOT
from ws2_training import train_scope30 as ts
from ws3_evaluation import analyze_scope30 as an
from ws2_training import prepare_data

METHODS_PATH = ROOT / "docs" / "report" / "methods_data_classifiers.md"
README_PATH = ROOT / "docs" / "report" / "README.md"
CONTRACTS_PATH = ROOT / "docs" / "agents" / "CONTRACTS.md"
FREEZE_PATH = ROOT / "docs" / "freeze" / "scope30-v1-seed13.freeze.json"
ANALYSIS_PATH = ROOT / "docs" / "analysis" / "scope30-v1-seed13" / "analysis.json"
MINILM_IDENTITY_SHA256 = "36bb95116de1358079c1fab49ac634493101e484c99edbd882691fbf96859cad"


def _read_methods():
    assert METHODS_PATH.is_file(), f"Missing methods file at {METHODS_PATH}"
    return METHODS_PATH.read_text(encoding="utf-8")


def _read_contracts():
    assert CONTRACTS_PATH.is_file(), f"Missing contracts file at {CONTRACTS_PATH}"
    return CONTRACTS_PATH.read_text(encoding="utf-8")


def _read_analysis():
    assert ANALYSIS_PATH.is_file(), f"Missing analysis file at {ANALYSIS_PATH}"
    return json.loads(ANALYSIS_PATH.read_bytes().decode("utf-8"))


def _read_freeze():
    assert FREEZE_PATH.is_file(), f"Missing freeze file at {FREEZE_PATH}"
    return json.loads(FREEZE_PATH.read_bytes().decode("utf-8"))


# --- Requirement 1 & 2: Datasets, Label Sets, Canonical Splits, Separation ---

def test_methods_documents_trec_dataset_and_splits():
    text = _read_methods()
    assert "SetFit/TREC-QC" in text
    for label in ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]:
        assert label in text
    assert "5,452" in text or "5452" in text  # official train count
    assert "4,907" in text or "4907" in text  # 90% train
    assert "545" in text                     # 10% pool
    assert "500" in text                     # official test
    assert "76" in text                      # abbreviation training frequency (>30)
    # Canonical split hashes: the values enforced by prepare_data.CHECKSUMS (and recorded in the freeze).
    for name in ("train", "pool", "test", "lowres_seed13"):
        assert prepare_data.CHECKSUMS[f"trec/{name}.jsonl"]["sha256"] in text
    freeze = _read_freeze()["real_references"]
    for name in ("pool", "test", "lowres_seed13"):
        assert freeze[f"trec/{name}"]["sha256"] in text
    # Issue #17: hashes from an earlier revision that match no committed record must stay out.
    for unbacked in ("00bfe51d6dc988aaae09e5318db50b2849dff78363630f9a7aa97d19da050e88",
                     "08b68832a764d25725f0580a6b728ae5c76dbb945110eb96bc75c88ea3074ba3",
                     "cbf804ec675c9603f90b656793f6c1ff5b26719ab29f1b402cf8d0be4c0c45aa",
                     "50f24e933ef74465df9d4fc0298a44b82d9a3b680c6553fc27df8d4a9cf41d8e"):
        assert unbacked not in text
        assert unbacked not in _read_contracts()
    # Pool label counts come from the Issue #5 contract (not stored in committed artifacts).
    assert "`ABBR`: 10, `DESC`: 126, `ENTY`: 128, `HUM`: 124, `LOC`: 78, `NUM`: 79" in text
    assert "`DESC`: 118" not in text
    # lowres_seed13 is sampled from the pool (prepare_data.py), not from train.jsonl.
    assert "was drawn from `pool.jsonl` using seed 13" in text
    assert "extracted from `train.jsonl`" not in text


def test_methods_documents_emotion_dataset_and_splits():
    text = _read_methods()
    assert "dair-ai/emotion" in text
    for label in ["sadness", "joy", "love", "anger", "fear", "surprise"]:
        assert label in text
    assert "16,000" in text or "16000" in text  # official train count
    assert "14,400" in text or "14400" in text  # 90% train
    assert "1,600" in text or "1600" in text    # 10% pool
    assert "2,000" in text or "2000" in text    # official test
    # Affective subsumption
    assert "annoyance" in text.lower() and "anger" in text
    assert "anxiety" in text.lower() and "fear" in text
    # Canonical split hashes
    assert "4c6e92df9649661ab1c6fefe06f83206382322e6997fcb70ca53d22d302fac57" in text
    assert "dde4702b5f5168f5fad977ba6b659f9304ae02a3cb37c18281e208271f847d40" in text
    assert "d3b7176e8ab6be052151511a3e9843fa633c5747d0dae73980a2e52a4fb6cd88" in text
    assert "7bec4826ca96fa056ec5defc13a051f1c2ac25ef0270164219af13e1471af27f" in text


def test_methods_documents_test_isolation_without_contradictory_zero_leakage():
    text = _read_methods()
    assert "seed = 0" in text or "seed 0" in text
    assert "500" in text and ("2,000" in text or "2000" in text)

    # Review Finding 1: Absolute "Zero Leakage Policy" claim must NOT appear
    assert "Zero Leakage Policy" not in text
    assert "zero leakage policy" not in text.lower()

    # Precise boundaries must be articulated
    assert "no official test row was directly sampled" in text.lower()
    assert "trec.train.05262" in text and "trec.test.00378" in text
    assert "pin + disclose" in text.lower() or "pin and disclose" in text.lower()
    assert "fail closed" in text.lower()


# --- Requirement 3 & 4: Exact scope30-v1 Six-Condition Design and Seed-13 Counts ---

def test_methods_documents_scope30_design_and_exact_counts():
    text = _read_methods()
    assert "scope30-v1" in text
    assert "seed 13" in text or "seed-13" in text
    assert "12 unique" in text or "12 result keys" in text or "12 unique result keys" in text

    # Condition counts
    assert "180" in text  # C1-C4 total
    assert "30 per label" in text or "30 examples per label" in text
    assert "C1" in text and "C2" in text and "C3" in text and "C4" in text
    assert "C5a" in text and "C5b" in text

    # C4: 15 real + 15 few-shot per label
    assert "15 real" in text and "15 few-shot" in text

    # C5a: 60 rows (10 per label)
    assert "60" in text and ("10 per label" in text or "10 real" in text)

    # C5b: 420 rows (60 real + 360 AugGPT paraphrases)
    assert "420" in text
    assert "360" in text
    assert "6 AugGPT paraphrases" in text or "6 paraphrases per source" in text

    # Total generated texts: 720 zero/few-shot + 720 AugGPT = 1,440
    assert "1,440" in text or "1440" in text


# --- Requirement 5: Deterministic Selection, Freeze, and Provenance ---

def test_methods_documents_deterministic_selection_and_freeze():
    text = _read_methods()
    # ADR 0012 rank hashing rule
    assert "scope30-v1|13|" in text
    assert "SHA-256" in text

    # Freeze SHA-256
    freeze_hash = ts.ACCEPTED_FREEZE_SHA256
    assert freeze_hash in text

    # Generator identity bound in freeze
    freeze = _read_freeze()
    assert freeze["accepted_model_identity"]["digest"] in text
    assert freeze["accepted_model_identity"]["configured_tag"] in text

    # Provenance record binding
    assert "records/" in text
    assert an.ACCEPTED_GRID_REVISION in text


# --- Requirement 6: Frozen MiniLM Identity and Correct Hash Terminology ---

def test_methods_documents_embedder_and_classifier_settings():
    text = _read_methods()
    contracts = _read_contracts()

    # Embedder name and revision
    assert ts.EMBEDDING_MODEL in text
    assert ts.APPROVED_MINILM_REVISION in text

    # Review Finding 2: 36bb9511... must be identified as identity_sha256 (snapshot file-hash mapping)
    minilm_hash = MINILM_IDENTITY_SHA256
    assert minilm_hash in text
    assert minilm_hash in contracts
    assert "identity_sha256" in text
    assert "snapshot file-hash mapping" in text.lower() or "snapshot mapping" in text.lower()
    assert "individual snapshot" in text.lower() or "individual snapshot-file" in text.lower()

    # Review Finding 2: "Weight Digest" / "Weights Digest" must NOT be used for 36bb9511...
    assert "weight digest" not in text.lower()
    assert "weights digest" not in text.lower()
    assert "weight digest" not in contracts.lower()
    assert "weights digest" not in contracts.lower()

    # Embedding configuration
    assert "384" in text
    assert "normalize_embeddings=True" in text or "L2" in text
    assert "frozen" in text.lower()
    assert "CPU" in text
    assert "64" in text  # batch size

    # Classifier
    assert "LogisticRegression" in text
    assert "C = 1.0" in text or "C=1.0" in text
    assert "max_iter = 2000" in text or "max_iter=2000" in text
    assert "random_state = 13" in text or "random_state=13" in text
    # Issue #17: only recorded classifier settings are stated; defaults and convergence are not recorded.
    assert "lbfgs" not in text.lower()
    assert "tol = 1e-4" not in text
    assert "multinomial" not in text.lower()
    assert "ensuring full numerical convergence" not in text
    assert "not recorded in the grid binding" in text
    # C4's real half is an independent seed-13 draw, not a prefix of the C1 sample.
    assert "first 15 real examples per label from the C1 sample" not in text
    assert "it is not constructed as a subset of the C1 sample" in text
    assert "| **TREC** | **C4** | 180 | 90 | 90 | ≤ 59 |" in text
    assert "| **Emotion** | **C4** | 180 | 90 | 90 | ≤ 54 |" in text


# --- Requirement 7: Training Harness Behavior ---

def test_methods_documents_training_harness_behavior():
    text = _read_methods()
    assert "train_scope30.py" in text
    assert "fail-closed" in text.lower() or "aborts" in text.lower()
    assert "results/scope30-v1-seed13-freeze-v1" in text
    assert "superseded/" in text
    assert "--replace" in text
    assert "--validate" in text
    assert an.ACCEPTED_GRID_COMPLETE_SHA256 in text


# --- Requirement 8: Separation of Input Counts and Real Labelling Costs ---

def test_methods_documents_cost_separation():
    text = _read_methods()
    analysis = _read_analysis()

    # Look up demonstration costs from analysis.json
    trec_c3_cost = next(c for c in analysis["costs"] if c["task"] == "trec" and c["condition"] == "C3")
    emo_c3_cost = next(c for c in analysis["costs"] if c["task"] == "emotion" and c["condition"] == "C3")

    # TREC C3 uses 59 distinct demonstrations, Emotion C3 uses 54
    assert str(trec_c3_cost["real_generation_demonstrations"]) in text
    assert str(emo_c3_cost["real_generation_demonstrations"]) in text

    # Check distinct cost disclosure
    assert "Input Parity" in text or "input parity" in text.lower() or "labelling investment" in text.lower()
    assert "59" in text and "54" in text


# --- Requirement 9: Metrics, Paired-Gap Calculation, No Repetition SD ---

def test_methods_documents_metrics_and_paired_gap_definitions():
    text = _read_methods()
    assert "Macro-F1" in text or "macro-F1" in text
    assert "Accuracy" in text or "accuracy" in text
    assert "Relative Gap" in text or "relative gap" in text.lower()
    assert "Absolute Gap" in text or "absolute gap" in text.lower()
    assert "undefined" in text.lower()
    assert "zero" in text.lower()


# --- Requirement 10: Reference to Accepted Twelve Results ---

def test_methods_cites_accepted_twelve_results_accurately():
    text = _read_methods()
    analysis = _read_analysis()

    # Verify each accepted result macro-F1 and accuracy appears in the document
    for r in analysis["results"]:
        f1_str = f"{r['macro_f1']:.4f}"
        acc_str = f"{r['accuracy']:.4f}"
        assert f1_str in text, f"Missing macro-F1 {f1_str} for {r['task']} {r['condition']}"
        assert acc_str in text, f"Missing accuracy {acc_str} for {r['task']} {r['condition']}"

    # Verify C2/C3 relative gaps appear in text
    assert "+4.12%" in text
    assert "+6.69%" in text
    assert "-6.80%" in text
    assert "-5.90%" in text


# --- Scientific Boundaries and Required Disclosures ---

def test_scientific_boundaries_and_disclosures_observed():
    text = _read_methods()

    # 1. No statistical significance or p-value claims
    assert "p-value" not in text.lower()
    assert "statistically significant" not in text.lower()
    assert "no cross-seed standard deviation" in text.lower() or "no standard deviation" in text.lower()

    # 2. Unequal C5a and C5b sizes explicitly disclosed
    assert "unequal" in text.lower()
    assert "60" in text and "420" in text
    assert ("7x" in text.lower() or "7×" in text or "7\\times" in text)

    # 3. Pinned TREC duplicate disclosed with limited scope
    assert "trec.train.05262" in text
    assert "trec.test.00378" in text
    assert "ba6af17c366034f8dca2abff920ba01cc773fc77eda6188a1127b8190b324d2e" in text
    assert "disclosed_test_text_overlap" in text
    assert "C1" in text and "C4" in text

    # 4. Non-causal task comparison
    assert "non-causal" in text.lower() or "cannot be causally attributed" in text.lower()

    # 5. Fake test fixture distinction preserved
    assert "fake" in text.lower() or "fixtures" in text.lower()


def test_contracts_register_contains_issue16_specification():
    contracts_text = _read_contracts()
    assert "## Data, classifier and training-method specifications (Issue #16, ADR 0012, ADR 0014)" in contracts_text
    assert "docs/report/methods_data_classifiers.md" in contracts_text
    assert ts.ACCEPTED_FREEZE_SHA256 in contracts_text
    assert ts.APPROVED_MINILM_REVISION in contracts_text
    assert MINILM_IDENTITY_SHA256 in contracts_text
    assert "identity_sha256" in contracts_text
    assert "weight digest" not in contracts_text.lower()
    assert "trec.train.05262" in contracts_text

    # Reconciled with Issue #15 (PR #38)
    assert "## Generation report sections and Appendix A (Issue #15, scope30-v1)" in contracts_text
    assert "section_2_related_work.md" in contracts_text
    assert "section_3a_generation_methods.md" in contracts_text
    assert "section_6_limitations_and_conclusions.md" in contracts_text
    assert "appendix_a_prompts_and_examples.md" in contracts_text

    # Review 5727002343: generation_report_sections.md must NOT appear in CONTRACTS.md
    assert "generation_report_sections.md" not in contracts_text

    # Review 5727002343: Forbidden inferential wording must NOT appear in Issue #15 contract
    issue15_section = contracts_text.split("## Generation report sections and Appendix A")[1].split("## Data, classifier")[0]
    for forbidden in ["outperformed", "underperformed", "gain", "loss", "matched or exceeded"]:
        assert forbidden not in issue15_section.lower(), f"Forbidden inferential word '{forbidden}' found in Issue #15 contract"

    # Review 5727002343: Accepted neutral interpretation must appear in Issue #15 contract
    assert "higher than their references at seed 13" in issue15_section
    assert "emotion c2/c3 values were lower than c1" in issue15_section.lower()
    assert "mixed metric directions" in issue15_section
    assert "reliability is unknown with one seed" in issue15_section
    assert "no equivalence or meaningfulness threshold was defined" in issue15_section
    assert "no causal, significance, stability or generalisation claim" in issue15_section


def test_report_readme_reconciled_with_pr38_and_sole_operator():
    assert README_PATH.is_file()
    readme_text = README_PATH.read_text(encoding="utf-8")

    # Review 5727002343: Actual canonical #15 deliverables from PR #38 (no consolidated duplicate)
    pr38_canonical_files = [
        "section_2_related_work.md",
        "section_3a_generation_methods.md",
        "section_6_limitations_and_conclusions.md",
        "appendix_a_prompts_and_examples.md",
        "methods_data_classifiers.md",
    ]
    for fn in pr38_canonical_files:
        assert fn in readme_text, f"Missing {fn} in docs/report/README.md"

    # Review 5727002343: generation_report_sections.md must NOT appear in README.md or methods
    assert "generation_report_sections.md" not in readme_text
    methods_text = _read_methods()
    assert "generation_report_sections.md" not in methods_text

    # No "planned" status
    assert "planned" not in readme_text.lower()

    # ADR 0015 restores the original accountable workstream ownership.
    assert "@harshitadjainbdes24-byte" in readme_text
    assert "@kausthubabelagodbdes24-oss" in readme_text
    assert "@hem4456" in readme_text
    assert "sole operator" not in readme_text.lower()

    # Shared assembly scaffold REPORT.md belongs to Issue #17
    assert "REPORT.md" in readme_text
    assert "17" in readme_text

    # ADR 0014 presentation boundary
    assert "ADR 0014" in readme_text
    assert "presentation" in readme_text.lower()
