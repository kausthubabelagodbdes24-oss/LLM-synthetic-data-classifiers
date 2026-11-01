"""Issue #14: the committed real analysis outputs are the exact files written at the PR #34 merge.

These checks need no real data or grid. They verify that the four files committed under
docs/analysis/scope30-v1-seed13/ are byte-for-byte the WRITE output (SHA-256 pinned here and recorded in
analysis.json), are bound to the accepted grid and a clean analysis checkout, and that every gap, comparison,
cost and the whole summary.md reproduce exactly from analysis.json with the merged analysis code.
"""

import json
from pathlib import Path

from common import ROOT
from ws3_evaluation import analyze_scope30 as an

OUT = ROOT / "docs" / "analysis" / "scope30-v1-seed13"
ANALYSIS_REVISION = "c7761ec944d86e866ce1103ed2b90ecf57b3ac8c"      # PR #34 merge that wrote the outputs
FILES_SHA256 = {
    "analysis.json": "d0486efdceb6cec23d47fc0a943f3307a65fe9230913c38f7f50af5dc26177f6",
    "f1_by_condition.png": "fc6e2f6acd9d9786d7ebb867cd232b0c757b2745e4eebf264e7a80745ba941e8",
    "gap.png": "5ed188eb1a1a853a6b812fcac218ee024d7986fcdd41266ab162f50a8d4971e3",
    "summary.md": "a95abe66758878a5b5e2bbb29eca8a203f4bb00d93443915a93236869d09b989",
}


def _doc():
    return json.loads((OUT / "analysis.json").read_bytes().decode("utf-8"))


def _sha(path):
    return an.ts._sha(Path(path).read_bytes())


def test_exactly_the_four_written_files_byte_for_byte():
    assert sorted(p.name for p in OUT.iterdir()) == sorted(FILES_SHA256)
    assert {n: _sha(OUT / n) for n in FILES_SHA256} == FILES_SHA256
    doc = _doc()
    assert doc["outputs_sha256"] == {n: s for n, s in FILES_SHA256.items() if n != "analysis.json"}
    assert (OUT / "analysis.json").read_bytes() == an.ts._canonical(doc)


def test_outputs_are_bound_to_the_accepted_grid_and_a_clean_merge_checkout():
    doc = _doc()
    assert (doc["analysis_kind"], doc["schema_version"]) == (an.ANALYSIS_KIND, an.SCHEMA_VERSION)
    assert doc["generation_mode"] == "real" and doc["real_run_evidence"] is True and doc["single_seed"] is True
    assert doc["analysis_code"] == {"revision": ANALYSIS_REVISION, "clean": True}
    grid = doc["grid"]
    assert grid["grid_complete_sha256"] == an.ACCEPTED_GRID_COMPLETE_SHA256
    assert grid["revision"] == an.ACCEPTED_GRID_REVISION and grid["decision"] == an.ACCEPTED_GRID_DECISION
    assert grid["freeze"]["sha256"] == an.ts.ACCEPTED_FREEZE_SHA256
    assert grid["classifier"]["embedding"]["revision"] == an.ts.APPROVED_MINILM_REVISION
    assert sorted(grid["records"]) == sorted(an.ts.key_name(*k) for k in an.ts.all_keys())


def test_every_calculation_and_the_summary_reproduce_from_analysis_json():
    doc = _doc()
    table = {(r["task"], r["condition"]): {k: r[k] for k in ("macro_f1", "accuracy", "n_train", "composition")}
             for r in doc["results"]}
    assert len(table) == 12
    assert all(table[(t, c)]["composition"] == an.EXPECTED_COMPOSITION[c] for t, c in table)
    gaps = an.gaps(table)
    comparisons = an.comparisons(table, gaps)
    demos = {(t, s): next(r["real_generation_demonstrations"] for r in doc["costs"]
                          if r["task"] == t and r["condition"] == ("C3" if s == "few_shot" else "C2"))
             for t in an.TASKS for s in ("zero_shot", "few_shot")}
    costs = an.costs(table, demos)
    roundtrip = json.loads(json.dumps({"gaps": gaps, "comparisons": comparisons, "costs": costs}))
    assert roundtrip == {"gaps": doc["gaps"], "comparisons": doc["comparisons"], "costs": doc["costs"]}
    assert doc["limitations"] == list(an.LIMITATIONS)
    prov = {"generation_mode": doc["generation_mode"], "real_run_evidence": doc["real_run_evidence"],
            "grid": doc["grid"]}
    result = {"table": table, "gaps": gaps, "comparisons": comparisons, "costs": costs}
    assert an.summary_markdown(result, prov).encode("utf-8") == (OUT / "summary.md").read_bytes()


def test_no_uncertainty_or_significance_is_reported():
    doc = _doc()
    assert doc["uncertainty"] == "none computed (one seed, one run per key)"
    for r in doc["results"]:
        assert set(r) == {"task", "condition", "macro_f1", "accuracy", "n_train", "composition"}
    text = (OUT / "summary.md").read_text(encoding="utf-8")
    assert "±" not in text and "p-value" not in text.lower()
