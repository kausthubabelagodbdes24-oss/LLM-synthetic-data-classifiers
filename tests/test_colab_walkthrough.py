import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "SynthText_scope30_walkthrough.ipynb"


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_colab_notebook_is_valid_and_output_free():
    nb = _notebook()
    assert (nb["nbformat"], nb["nbformat_minor"]) == (4, 5)
    assert nb["metadata"]["colab"]["name"] == NOTEBOOK.name
    assert len(nb["cells"]) >= 15
    for cell in nb["cells"]:
        assert cell["cell_type"] in {"markdown", "code"}
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))


def test_colab_notebook_keeps_fake_and_real_evidence_separate():
    nb = _notebook()
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "FAKE DEMONSTRATION — NOT EXPERIMENTAL EVIDENCE" in text
    assert "ACCEPTED REAL RESULTS — READ ONLY" in text
    assert "SYNTHTEXT_GENERATION_MODE': 'fake'" in text
    assert "SYNTHTEXT_EMBEDDER': 'hashing'" in text
    assert "data/sample" in text and "results/sample" in text
    assert "docs/analysis/scope30-v1-seed13" in text
    assert "grid_complete_sha256" in text and "outputs_sha256" in text
    assert "without Ollama" in text


def test_colab_notebook_has_generation_training_analysis_and_claim_limits():
    text = "\n".join("".join(c["source"]) for c in _notebook()["cells"])
    for setting in ("zero_shot", "few_shot", "auggpt"):
        assert setting in text
    assert "ws2_training.train" in text
    assert "ws3_evaluation.analyze" in text
    assert "f1_by_condition.png" in text and "gap.png" in text
    assert "reliability is unknown with one seed" in text
    assert "show subjectivity caused the task difference" in text
    assert "no standard deviation, confidence interval, significance test" in text.lower()


def test_presentation_is_explicitly_outside_github_scope():
    files = [
        ROOT / "README.md",
        ROOT / "docs" / "PROJECT_SPEC.md",
        ROOT / "docs" / "SCOPE_MIGRATION.md",
        ROOT / "docs" / "adr" / "0014-presentation-separation-colab.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "final professor-facing presentation" in combined
    assert "outside" in combined
    assert "not tracked" in combined or "must not request" in combined
