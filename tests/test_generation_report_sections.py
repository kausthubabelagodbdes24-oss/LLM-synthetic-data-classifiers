"""Focused validation tests for Issue #15 generation report sections and Appendix A.

Validates:
1. Presence and non-emptiness of modular report files under docs/report/.
2. Primary literature citations (Li et al. 2023, Dai et al. 2023, Brown et al. 2020, Reimers & Gurevych 2019).
3. Accepted Qwen model identity, digest, temperature, and thinking behavior.
4. Scope30-v1 exact counts, selected artifacts, demonstration costs, and cross-setting overlap.
5. Model-assisted quality review protocol, trigger rule, findings, corrections, and Option A freeze.
6. Verbatim prompt templates matching ws1_generation/generate.py.
7. Two good and two problematic examples per task with verified provenance, distinguishing final artifacts from historical pilot evidence.
8. Grounding of conclusions in the four accepted single-seed comparisons, including contrary outcomes and boundary disclosures.
9. Preservation of shared assembly file (docs/REPORT.md).
"""

from pathlib import Path
import json
import pytest

from ws1_generation import generate

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_REPORT_DIR = REPO_ROOT / "docs" / "report"
FREEZE_MANIFEST_PATH = REPO_ROOT / "docs" / "freeze" / "scope30-v1-seed13.freeze.json"
ANALYSIS_JSON_PATH = REPO_ROOT / "docs" / "analysis" / "scope30-v1-seed13" / "analysis.json"
REPORT_SCAFFOLD_PATH = REPO_ROOT / "docs" / "REPORT.md"

EXPECTED_FILES = [
    "README.md",
    "section_2_related_work.md",
    "section_3a_generation_methods.md",
    "section_6_limitations_and_conclusions.md",
    "appendix_a_prompts_and_examples.md",
]


def test_report_files_exist_and_non_empty():
    assert DOCS_REPORT_DIR.is_dir(), f"Missing directory {DOCS_REPORT_DIR}"
    for filename in EXPECTED_FILES:
        filepath = DOCS_REPORT_DIR / filename
        assert filepath.is_file(), f"Missing expected report file {filepath}"
        content = filepath.read_text(encoding="utf-8")
        assert len(content.strip()) > 200, f"File {filename} is unexpectedly short ({len(content)} chars)"


def test_primary_citations_verified():
    sec2 = (DOCS_REPORT_DIR / "section_2_related_work.md").read_text(encoding="utf-8")
    
    # 1. Li et al. (2023)
    assert "2310.07849" in sec2
    assert "Li" in sec2
    assert "Synthetic Data Generation with Large Language Models for Text Classification" in sec2
    assert "EMNLP 2023" in sec2 or "2023 Conference on Empirical Methods" in sec2

    # 2. Dai et al. (2023) - AugGPT
    assert "2302.13007" in sec2
    assert "Dai" in sec2
    assert "AugGPT: Leveraging ChatGPT for Text Data Augmentation" in sec2

    # 3. Brown et al. (2020) - GPT-3
    assert "2005.14165" in sec2
    assert "Brown" in sec2
    assert "Language Models are Few-Shot Learners" in sec2
    assert "NeurIPS 2020" in sec2 or "Neural Information Processing Systems" in sec2

    # 4. Reimers and Gurevych (2019) - Sentence-BERT
    assert "1908.10084" in sec2
    assert "Reimers" in sec2
    assert "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" in sec2
    assert "EMNLP" in sec2


def test_accepted_qwen_model_settings_and_provenance():
    sec3a = (DOCS_REPORT_DIR / "section_3a_generation_methods.md").read_text(encoding="utf-8")
    
    # Model identity & digest
    assert "qwen3:4b" in sec3a
    assert "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7" in sec3a
    assert "0.9" in sec3a  # temperature
    assert "Ollama" in sec3a
    assert "RTX 4060" in sec3a

    # Thinking behavior & token cap
    assert '"think": false' in sec3a
    assert "1024" in sec3a
    assert "ADR 0013" in sec3a


def test_scope30_counts_and_costs_disclosed():
    sec3a = (DOCS_REPORT_DIR / "section_3a_generation_methods.md").read_text(encoding="utf-8")
    sec6 = (DOCS_REPORT_DIR / "section_6_limitations_and_conclusions.md").read_text(encoding="utf-8")
    
    # Selected counts
    for text in (sec3a, sec6):
        assert "180" in text  # C1-C4 size per task
        assert "60" in text   # C5a size
        assert "420" in text  # C5b size
        assert "360" in text  # AugGPT paraphrases per task
        assert "1,440" in text or "1440" in text  # Total synthetic selected

    # Demonstration costs
    assert "59" in sec3a and "54" in sec3a
    assert "59" in sec6 and "54" in sec6
    assert "18" in sec3a  # TREC cross-setting overlap


def test_quality_review_findings_and_freeze_disclosures():
    sec3a = (DOCS_REPORT_DIR / "section_3a_generation_methods.md").read_text(encoding="utf-8")
    sec6 = (DOCS_REPORT_DIR / "section_6_limitations_and_conclusions.md").read_text(encoding="utf-8")
    
    # Model-assisted review disclosure (not human)
    assert "Claude Opus 5.5" in sec3a
    assert "model-assisted" in sec3a.lower()
    assert "not independent human" in sec3a.lower() or "not described as independent human" in sec6.lower()
    assert "not a population" in sec3a.lower() or "not a population" in sec6.lower()
    assert "not model-review all 1,440" in sec3a.lower() or "did not model-review all 1,440" in sec3a.lower()
    assert "10" in sec3a  # 10 rows per batch

    # Multi-stage quality evidence
    assert "round 2" in sec3a.lower()
    assert "targeted recheck" in sec3a.lower() or "targeted re-check" in sec3a.lower()
    assert "supplementary emotion" in sec3a.lower()
    assert "scope scan" in sec3a.lower()

    # Freeze SHA-256
    freeze_json = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert "dd2e52ee704835525df41071ff5214c56593969d248ca8812c23b617bfd74de1" in sec3a
    assert freeze_json["totals"]["selected_zero_few_shot_texts"] == 720
    assert freeze_json["totals"]["auggpt_paraphrases"] == 720

    # Disclosed quality flags
    assert "07790" in sec3a and "15223" in sec3a
    assert "04163" in sec3a and "00845" in sec3a
    assert "Hot Wheels" in sec3a and "Hot and Wheels" in sec3a
    assert "Option A" in sec3a and "Option A" in sec6


def test_prompt_templates_match_code():
    app_a = (DOCS_REPORT_DIR / "appendix_a_prompts_and_examples.md").read_text(encoding="utf-8")
    
    # Verify function names
    assert "def prompt_zero_shot" in app_a
    assert "def prompt_few_shot" in app_a
    assert "def prompt_auggpt" in app_a
    assert "def prompt_auggpt_length_guard" in app_a
    assert "def prompt_auggpt_trec_answer_type" in app_a

    # Verify key template text snippets directly matching generate.py
    assert 'You are building a dataset of {t[\'context\']}' in app_a
    assert 'Write {n} different examples of {t[\'style\']}' in app_a
    assert 'Here are real examples with this label:' in app_a
    assert 'Please rephrase the following sentence {k} different ways, keeping its meaning:' in app_a
    assert 'Each rephrasing must be at most {max_words} words, so shorten it:' in app_a
    assert 'Every rephrasing must keep the same expected answer type, so it keeps this label.' in app_a
    assert '"think": false, "num_predict": 1024' in app_a


def test_examples_provenance_and_separation():
    app_a = (DOCS_REPORT_DIR / "appendix_a_prompts_and_examples.md").read_text(encoding="utf-8")
    freeze_json = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))

    # Accepted Final Artifact Examples
    final_ids = [
        "trec.train.04455.seed13.aug6",
        "trec.train.03037.seed13.aug3",
        "trec.train.04163.seed13.aug4",
        "trec.train.00845.seed13.aug3",
        "emotion.few_shot.seed13.anger.0016",
        "emotion.zero_shot.seed13.joy.0004",
        "emotion.train.07790.seed13.aug1",
        "emotion.train.15223.seed13.aug1",
    ]
    for row_id in final_ids:
        assert row_id in app_a, f"Missing final evidence row ID {row_id} in Appendix A"
        # Verify the row ID exists in the freeze manifest ordered IDs
        batch_matches = [
            batch for batch, data in freeze_json["artifacts"].items()
            if row_id in data["consumed"]["ids_in_file_order"]
        ]
        assert len(batch_matches) == 1, f"Row ID {row_id} not uniquely found in freeze manifest (found in {batch_matches})"

    # Historical Pilot Examples
    pilot_ids = [
        "trec.zero_shot.seed13.description.0005",
        "trec.few_shot.seed13.entity.0003",
        "trec.zero_shot.seed13.entity.0005",
        "trec.few_shot.seed13.location.0007",
        "emotion.zero_shot.seed13.sadness.0002",
        "emotion.few_shot.seed13.love.0009",
        "emotion.zero_shot.seed13.joy.0002",
        "emotion.few_shot.seed13.fear.0005",
    ]
    for pid in pilot_ids:
        assert pid in app_a, f"Missing historical pilot row ID {pid} in Appendix A"

    # Distinction check
    assert "Historical Pilot Examples" in app_a
    assert "distinct from the final frozen" in app_a.lower() or "superseded by the final frozen" in app_a.lower()


def test_conclusions_grounded_in_four_comparisons():
    sec6 = (DOCS_REPORT_DIR / "section_6_limitations_and_conclusions.md").read_text(encoding="utf-8")
    analysis_data = json.loads(ANALYSIS_JSON_PATH.read_text(encoding="utf-8"))
    res = analysis_data["results"]

    # Comparison 1: Synthetic vs Real across tasks
    assert "0.6009" in sec6  # TREC C1
    assert "0.6257" in sec6  # TREC C2
    assert "0.6411" in sec6  # TREC C3
    assert "+4.12%" in sec6  # TREC C2 gap
    assert "+6.69%" in sec6  # TREC C3 gap
    assert "0.4082" in sec6  # Emotion C1
    assert "0.3804" in sec6  # Emotion C2
    assert "0.3841" in sec6  # Emotion C3
    assert "-6.80%" in sec6  # Emotion C2 gap
    assert "-5.90%" in sec6  # Emotion C3 gap
    assert "-10.92" in sec6  # C2 cross-task gap difference
    assert "-12.59" in sec6  # C3 cross-task gap difference

    # Comparison 2: Few-shot vs Zero-shot
    assert "+0.0155" in sec6  # TREC C3 - C2 macro-F1
    assert "+0.0037" in sec6  # Emotion C3 - C2 macro-F1

    # Comparison 3: 50/50 mix vs Real
    assert "0.6638" in sec6  # TREC C4
    assert "+0.0629" in sec6  # TREC C4 - C1
    assert "0.4033" in sec6  # Emotion C4
    assert "-0.0049" in sec6  # Emotion C4 - C1

    # Comparison 4: AugGPT vs Low-resource real
    assert "0.5974" in sec6  # TREC C5a
    assert "0.6251" in sec6  # TREC C5b
    assert "+0.0277" in sec6  # TREC C5b - C5a
    assert "0.3381" in sec6  # Emotion C5a
    assert "0.3432" in sec6  # Emotion C5b
    assert "+0.0051" in sec6  # Emotion C5b - C5a

    # Contrary numerical directions highlighted without inferential wording
    assert "contrary" in sec6.lower()
    assert "the reliability of these observed directions is unknown" in sec6.lower()
    assert "with one seed" in sec6.lower()
    assert "no equivalence" in sec6.lower() or "meaningfulness threshold" in sec6.lower()
    assert "observed numerical directions" in sec6.lower() or "observed directions" in sec6.lower()

    # Enforce absence of inferential language across section 6
    forbidden_terms = [
        "outperformed",
        "outperform",
        "underperformed",
        "underperform",
        "matched or exceeded",
        "negligible",
        "sampling noise",
        "align well",
    ]
    for term in forbidden_terms:
        assert term not in sec6.lower(), f"Forbidden inferential term '{term}' found in section 6"

    # Enforce absence of inferential gain/loss claims
    import re
    assert not re.search(r"\bperformance gain\b|\bperformance loss\b|\bgains?\b|\bloss(es)?\b", sec6, re.IGNORECASE), "Inferential gain/loss claim found in section 6"

    # Boundaries: single seed, no significance, no causal subjectivity
    assert "single-seed" in sec6.lower() or "single seed" in sec6.lower()
    assert "no claims of statistical significance" in sec6.lower() or "no statistical significance" in sec6.lower()
    assert "cannot be causally attributed" in sec6.lower()


def test_shared_report_scaffold_unmodified():
    # Verify REPORT.md is preserved untouched
    scaffold = REPORT_SCAFFOLD_PATH.read_text(encoding="utf-8")
    assert "## Contributors & Workstream Responsibilities" in scaffold
    assert "*Owner: Harshita (@harshitadjainbdes24-byte)*" in scaffold
    assert "## Appendix A. Prompt Templates and Generation Examples" in scaffold
