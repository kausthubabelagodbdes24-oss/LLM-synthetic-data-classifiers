"""Issue #17: the assembled scope30-v1 report (docs/REPORT.md) and its PDF reconcile with accepted evidence.

Every check reads only committed artifacts: the accepted #14 analysis.json and charts, the #12 freeze manifest,
the #5 canonical split hashes and the #13 trainer constants. Nothing here generates, trains or analyses data.
Negative cases mutate an in-memory copy of the report and expect a specific problem.
"""

import json
import shutil
from pathlib import Path

import pytest

from ws3_evaluation import report_scope30 as rp


@pytest.fixture(scope="module")
def evidence():
    return rp.load_evidence()


@pytest.fixture(scope="module")
def report():
    return rp.REPORT_PATH.read_text(encoding="utf-8")


def problems_for(text, evidence):
    analysis, freeze = evidence
    return rp.validate_report(text, analysis, freeze)


def assert_flagged(text, evidence, needle):
    found = problems_for(text, evidence)
    assert any(needle in p for p in found), f"expected a problem containing {needle!r}, got {found}"


def test_committed_report_reconciles_with_accepted_evidence(report, evidence):
    assert problems_for(report, evidence) == []


def test_committed_pdf_matches_sidecar_source_and_values(evidence):
    analysis, _ = evidence
    assert rp.check_pdf(analysis) == []
    sidecar = json.loads(rp.PDF_SIDECAR_PATH.read_text(encoding="utf-8"))
    assert sidecar["report_sha256_lf"] == rp.sha256_lf(rp.REPORT_PATH)
    assert sidecar["pdf_sha256"] == rp.sha256_bytes(rp.PDF_PATH.read_bytes())
    assert sidecar["review_status"] == "not reviewed or accepted by the author"
    assert b"file://" not in rp.PDF_PATH.read_bytes()


def test_cli_check_passes(capsys):
    assert rp.main(["--check"]) == 0
    assert "PASS report reconciles" in capsys.readouterr().out


def test_table1_is_the_exact_twelve_accepted_results(evidence):
    analysis, _ = evidence
    rows = rp.expected_table1_rows(analysis)
    assert len(rows) == 12
    assert rows[0] == "| TREC | C1 Real | 180 | 0.6009 | 0.6520 |"
    assert rows[5] == "| TREC | C5b Sources + AugGPT | 420 | 0.6251 | 0.6700 |"
    assert rows[7] == "| Emotion | C2 Zero-shot | 180 | 0.3804 | 0.4305 |"
    assert rows[10] == "| Emotion | C5a Real sources | 60 | 0.3381 | 0.3730 |"


def test_table2_covers_all_four_planned_comparisons(evidence):
    analysis, _ = evidence
    rows = rp.expected_table2_rows(analysis)
    assert len(rows) == 10
    assert rows[0] == "| 1. C2 − C1 (zero-shot vs real) | TREC | +0.0247 | +4.12% | +0.0160 | +8 |"
    assert rows[3] == "| | Emotion | −0.0241 | −5.90% | −0.0135 | −27 |"
    assert rows[6] == "| 3. C4 − C1 (mix vs real) | TREC | +0.0629 | | +0.0580 | +29 |"
    assert rows[9] == "| | Emotion | +0.0051 | | −0.0025 | −5 |"


@pytest.mark.parametrize("old,new,needle", [
    ("| TREC | C3 Few-shot | 180 | 0.6411 | 0.6880 |", "| TREC | C3 Few-shot | 180 | 0.6412 | 0.6880 |", "Table 1 row"),
    ("| Emotion | C5b Sources + AugGPT | 420 | 0.3432 | 0.3705 |", "", "exactly twelve result rows"),
    ("| | Emotion | −0.0278 | −6.80% | −0.0115 | −23 |", "| | Emotion | −0.0278 | −6.81% | −0.0115 | −23 |",
     "Table 2 row"),
    ("−10.92 percentage", "−10.9 percentage", "10.92"),
    ("**59** distinct real demonstrations for TREC", "**60** distinct real demonstrations for TREC", "**59**"),
    ("| C5b | C5a sources + 6 AugGPT paraphrases each | 420 | 60 | 360 |",
     "| C5b | C5a sources + 6 AugGPT paraphrases each | 360 | 60 | 360 |", "420 | 60 | 360"),
    ("**1,440** generated texts", "**1,400** generated texts", "1,440"),
    ("| trec/auggpt (v2, promoted) | 2 | 1 | 3 | PASS |", "| trec/auggpt (v2, promoted) | 2 | 0 | 3 | PASS |",
     "trec/auggpt (v2, promoted)"),
    ("LogisticRegression(C=1.0, max_iter=2000, random_state=13)", "LogisticRegression(C=0.5, max_iter=2000, "
     "random_state=13)", "LogisticRegression"),
])
def test_altered_values_are_rejected(report, evidence, old, new, needle):
    assert old in report
    assert_flagged(report.replace(old, new, 1), evidence, needle)


def test_wrong_or_unbacked_identifiers_are_rejected(report, evidence):
    # The #16 module's TREC split hashes are not backed by prepare_data.CHECKSUMS or the freeze.
    wrong = "00bfe51d6dc988aaae09e5318db50b2849dff78363630f9a7aa97d19da050e88"
    canonical = "b587c84d26b361bbe6f4126aff136e9d737c6c349059e89d2a3ce79e226f4331"
    mutated = report.replace(canonical, wrong)
    assert_flagged(mutated, evidence, f"hash not backed by accepted evidence: {wrong}")
    assert_flagged(mutated, evidence, "canonical split hash for trec/train.jsonl")
    assert_flagged(report.replace("1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
                                  "1110a243fdf4706b3f48f1d95db1a4f5529b4d42"), evidence, "1110a243")
    assert_flagged(report.replace("359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7",
                                  "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae8"), evidence,
                   "359d7dd4")


@pytest.mark.parametrize("heading", list(rp.REQUIRED_HEADINGS))
def test_missing_sections_are_rejected(report, evidence, heading):
    assert_flagged(report.replace(heading, "## Removed", 1), evidence, heading)


def test_appendix_must_follow_references_after_a_page_break(report, evidence):
    assert_flagged(report.replace(rp.APPENDIX_BREAK, ""), evidence, "page break")
    body, refs, appendix = rp.split_parts(report)
    assert_flagged(body + appendix + "\n" + refs, evidence, "References")


def test_comparisons_figures_and_tables_must_be_discussed(report, evidence):
    assert_flagged(report.replace("**Comparison 3:", "**Mixed result:"), evidence, "comparison 3")
    assert_flagged(report.replace("**Figure 2**", "**The second chart**"), evidence, "Figure 2")


def test_charts_must_be_the_unchanged_accepted_charts(report, evidence, tmp_path):
    assert_flagged(report.replace("](analysis/scope30-v1-seed13/gap.png)", "](analysis/gap_v2.png)"), evidence,
                   "gap.png must be embedded exactly once")
    docs = tmp_path / "docs"
    charts = docs / "analysis" / "scope30-v1-seed13"
    charts.mkdir(parents=True)
    for name in rp.ACCEPTED_CHARTS:
        shutil.copyfile(rp.ANALYSIS_DIR / name, charts / name)
    analysis, freeze = evidence
    assert rp.validate_report(report, analysis, freeze, report_dir=docs) == []
    with open(charts / "gap.png", "ab") as fh:
        fh.write(b"\0")
    found = rp.validate_report(report, analysis, freeze, report_dir=docs)
    assert any("gap.png is missing or differs" in p for p in found)


@pytest.mark.parametrize("sentence,needle", [
    ("C3 is significantly better than C2 on TREC.", "significance"),
    ("The gap is statistically significant (p < 0.05).", "significance"),
    ("These directions are stable across seeds.", "stability"),
    ("We replicated the finding on Emotion.", "replication"),
    ("Subjectivity caused the larger Emotion gap.", "causation"),
    ("C3 and C2 are equivalent on Emotion.", "equivalence"),
    ("Two human annotators reached inter-annotator agreement on the sample.", "human agreement"),
    ("The corpora passed independent human review.", "independent human review"),
    ("Few-shot prompting improves Emotion accuracy.", "superiority"),
])
def test_forbidden_scientific_claims_are_rejected(report, evidence, sentence, needle):
    marker = "**Summary of directions.**"
    assert_flagged(report.replace(marker, f"{sentence} {marker}", 1), evidence, needle)


def test_negated_or_limited_claim_wording_is_allowed():
    assert rp.claim_problems("No significance, stability or replication claim is made.") == []
    assert rp.claim_problems("This does not show that subjectivity is the cause.") == []
    assert rp.claim_problems("Reliability is unknown, so no equivalence is claimed.") == []
    assert rp.claim_problems("`stable` in code and ```\nsignificant\n``` blocks are ignored.") == []


@pytest.mark.parametrize("old,new,needle", [
    (rp.AUTHOR_LINE, "**Author:** K. Belagod", "workstream author line"),
    (rp.REVIEW_STATUS, "The project lead reviewed and accepted this report.", "not yet reviewed"),
    ("Harshita (@harshitadjainbdes24-byte", "Harshita Jain (@harshitadjainbdes24-byte", "Harshita Jain"),
    ("## 1. Introduction", "*Planning length: ~0.5 pages*\n\n## 1. Introduction", "Planning length"),
    ("## 1. Introduction", "Comparison 1: Blocked pending C2/C3 runs.\n\n## 1. Introduction", "Blocked pending"),
    ("## 1. Introduction", "Krippendorff alpha was 0.6.\n\n## 1. Introduction", "Krippendorff"),
])
def test_identity_review_status_and_stale_scaffold_are_checked(report, evidence, old, new, needle):
    assert_flagged(report.replace(old, new, 1), evidence, needle)


def test_disclosures_must_remain(report, evidence):
    assert_flagged(report.replace("trec.test.00378", "a test item"), evidence, "trec.test.00378")
    assert_flagged(report.replace("C5b has seven times as many training rows as C5a", "C5b is larger"), evidence,
                   "seven times")
    assert_flagged(report.replace("model-assisted", "careful"), evidence, "model-assisted")


def test_citations_are_complete_and_consistent(report, evidence):
    assert_flagged(report.replace("pages 10443–10461", "pages 10443–10460"), evidence, "Li et al. (2023)")
    assert_flagged(report.replace("Reimers and Gurevych (2019) introduced", "Sentence-BERT was introduced"),
                   evidence, "Reimers and Gurevych (2019) is not used")
    assert_flagged(report.replace("Brown et al. (2020) showed", "Smith et al. (2021) and Brown et al. (2020) showed"),
                   evidence, "Smith et al. (2021)")


def test_analysis_and_freeze_pins_are_enforced(tmp_path):
    changed = tmp_path / "analysis.json"
    changed.write_bytes(rp.ANALYSIS_PATH.read_bytes().replace(b"0.652", b"0.653", 1))
    with pytest.raises(rp.ReportError, match="accepted #14 analysis"):
        rp.load_evidence(analysis_path=changed)
    freeze = tmp_path / "freeze.json"
    freeze.write_bytes(rp.FREEZE_PATH.read_bytes() + b" ")
    with pytest.raises(rp.ReportError, match="accepted #12 freeze"):
        rp.load_evidence(freeze_path=freeze)


def test_stale_or_altered_pdf_is_rejected(evidence, tmp_path):
    analysis, _ = evidence
    pdf, sidecar, md = tmp_path / "REPORT.pdf", tmp_path / "REPORT.pdf.json", tmp_path / "REPORT.md"
    shutil.copyfile(rp.PDF_PATH, pdf)
    shutil.copyfile(rp.PDF_SIDECAR_PATH, sidecar)
    md.write_text(rp.REPORT_PATH.read_text(encoding="utf-8") + "\nEdited after the PDF build.\n", encoding="utf-8")
    found = rp.check_pdf(analysis, pdf_path=pdf, sidecar_path=sidecar, report_path=md)
    assert any("stale" in p for p in found)
    shutil.copyfile(rp.REPORT_PATH, md)
    assert rp.check_pdf(analysis, pdf_path=pdf, sidecar_path=sidecar, report_path=md) == []
    with open(pdf, "ab") as fh:
        fh.write(b"\n%tampered\n")
    found = rp.check_pdf(analysis, pdf_path=pdf, sidecar_path=sidecar, report_path=md)
    assert any("does not match the SHA-256" in p for p in found)


def test_pdf_text_check_detects_missing_values_and_blank_pages(evidence):
    analysis, _ = evidence
    pages = rp.pdf_text_pages(rp.PDF_PATH)
    assert rp.check_pdf_text(pages, analysis) == []
    first = next(r for r in analysis["results"])
    value = f"{first['macro_f1']:.4f}"
    altered = [p.replace(value, "0.0000") for p in pages]
    assert any(value in problem for problem in rp.check_pdf_text(altered, analysis))
    assert any("blank page" in p for p in rp.check_pdf_text(pages + [" "], analysis))


def test_report_crlf_checkout_is_tolerated(report, evidence):
    assert problems_for(report.replace("\n", "\r\n"), evidence) == []


# --- PR #39 review fixes -------------------------------------------------------------------------------------

@pytest.mark.parametrize("sentence,needle", [
    ("The C3 − C2 difference is not significant.", "not significant"),
    ("No difference was statistically insignificant.", "insignificant"),
    ("This is the largest within-task difference in the grid.", "largest within-task difference in the grid"),
    ("C3/C4 consumed 59/54 demonstrations.", "C3/C4 consumed"),
])
def test_strict_wording_is_rejected_even_when_negated(report, evidence, sentence, needle):
    marker = "**Summary of directions.**"
    assert_flagged(report.replace(marker, f"{sentence} {marker}", 1), evidence, needle)


def test_significance_and_bounded_c4_cost_disclosures_are_required(report, evidence):
    assert_flagged(report.replace("Statistical significance was not assessed or claimed",
                                  "Significance was not discussed"), evidence, "Statistical significance")
    assert_flagged(report.replace("C4 on at most", "C4 on exactly"), evidence, "C4 on at most")


def test_largest_comparison_claim_matches_accepted_comparisons(report, evidence):
    analysis, freeze = evidence
    assert rp.largest_planned_difference(analysis) == ("mix_vs_real", "trec")
    altered = json.loads(json.dumps(analysis))
    for comp in altered["comparisons"]:
        if comp["id"] == "few_vs_zero":
            comp["per_task"]["emotion"]["macro_f1_diff"] = 0.5
    found = rp.validate_report(report, altered, freeze)
    assert any("'largest' claim does not match" in p for p in found)


def test_report_modules_cite_only_backed_hashes_and_no_strict_wording(evidence):
    import re
    analysis, freeze = evidence
    known = rp.known_hashes(analysis, freeze)
    for path in sorted((rp.DOCS / "report").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        unbacked = set(re.findall(r"\b[0-9a-f]{64}\b", text)) - known
        assert not unbacked, f"{path.name} cites hashes not backed by accepted evidence: {sorted(unbacked)}"
        if path.name == "README.md":
            continue                                   # quotes the retired claims as reconciliation notes
        for phrase in rp.STRICT_FORBIDDEN:
            assert phrase.lower() not in text.lower(), f"{path.name}: {phrase!r}"


def test_every_committed_pdf_page_renders_inside_the_margins():
    geometry = rp.render_page_geometry(rp.PDF_PATH)
    assert len(geometry) == json.loads(rp.PDF_SIDECAR_PATH.read_text(encoding="utf-8"))["pages"]
    assert rp.check_pdf_render(rp.PDF_PATH) == []
    for g in geometry:
        assert abs(g["left_mm"] - rp.LEFT_MARGIN_MM) <= rp.LEFT_EDGE_TOLERANCE_MM


def test_shifted_or_clipped_page_is_detected(tmp_path):
    from pypdf import PdfReader, PdfWriter, Transformation
    writer = PdfWriter(clone_from=PdfReader(str(rp.PDF_PATH)))
    writer.pages[7].add_transformation(Transformation().translate(tx=-40, ty=0))   # page 8, 14 mm left
    shifted = tmp_path / "shifted.pdf"
    with open(shifted, "wb") as fh:
        writer.write(fh)
    problems = rp.check_pdf_render(shifted)
    assert any(p.startswith("page 8 ") for p in problems), problems
    assert not any(p.startswith("page 7 ") for p in problems)


def test_pdf_footer_is_part_of_the_page_not_a_type3_overlay():
    from pypdf import PdfReader
    reader = PdfReader(str(rp.PDF_PATH))
    total = len(reader.pages)
    for number, page in enumerate(reader.pages, 1):
        assert f"SynthText scope30-v1 report — page {number} of {total}" in page.extract_text()
        fonts = page.get("/Resources", {}).get("/Font", {})
        for ref in fonts.values():
            assert ref.get_object().get("/Subtype") != "/Type3"
