"""Checks for the professor-facing documentation path."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "CONTEXT.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "PROJECT_SPEC.md",
    ROOT / "docs" / "REPRODUCIBILITY.md",
)


def test_public_documentation_links_resolve():
    pattern = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
    missing = []
    for document in PUBLIC_DOCS:
        for target in pattern.findall(document.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (document.parent / target).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not missing, "Broken public-documentation links: " + ", ".join(missing)


def test_public_entry_points_describe_the_finished_study_not_internal_rescoping():
    text = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (ROOT / "README.md", ROOT / "docs" / "PROJECT_SPEC.md")
    )
    for stale in (
        "100 per label",
        "uniform 100",
        "scope migration",
        "supersedes",
        "awaiting lead review",
        "not yet reviewed",
        "still need",
        "sole operator",
    ):
        assert stale not in text
    assert "repository development is complete" in text
    assert "completed and accepted" in text


def test_readme_ci_badge_tracks_main_pushes_explicitly():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        "actions/workflows/ci.yml/badge.svg?branch=main&event=push" in readme
    )


def test_readme_results_match_the_accepted_analysis():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    analysis = json.loads(
        (ROOT / "docs" / "analysis" / "scope30-v1-seed13" / "analysis.json").read_text(encoding="utf-8")
    )
    for row in analysis["results"]:
        assert f"{row['macro_f1']:.4f}" in readme
        assert f"{row['accuracy']:.4f}" in readme


def test_public_documents_preserve_ownership_and_claim_boundaries():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    spec = (ROOT / "docs" / "PROJECT_SPEC.md").read_text(encoding="utf-8")
    combined = readme + "\n" + spec
    for owner in ("Harshita", "Kausthuba Belagod", "Hementh"):
        assert owner in combined
    assert "does not isolate subjectivity as the cause" in spec
    assert "no standard deviation, confidence interval, significance test" in spec
    assert "final professor-facing presentation" in spec
    assert "outside the repository" in spec


def test_public_credits_are_human_team_members_only_and_mailmap_corrects_old_email():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    team = readme.split("## Team", 1)[1].split("## Status and scope", 1)[0]
    for owner in ("Harshita", "Kausthuba Belagod", "Hementh"):
        assert owner in team
    for non_team_credit in ("swiftsol", "claude", "anthropic"):
        assert non_team_credit not in team.lower()

    mailmap = (ROOT / ".mailmap").read_text(encoding="utf-8")
    assert "Kausthuba Belagod <belgod@gmail.com>" in mailmap
    assert "Claude Opus 5.5 <noreply@anthropic.com>" in mailmap
    assert "kausthubabelagodbdes24@rvu.edu.in" in mailmap
