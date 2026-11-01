"""Issue #17: check and build the scope30-v1 exploratory report (docs/REPORT.md) and its PDF.

    python -m ws3_evaluation.report_scope30 --check            # read-only: report, charts and committed PDF
    python -m ws3_evaluation.report_scope30 --build-pdf OUT    # write a PDF of docs/REPORT.md to OUT
    python -m ws3_evaluation.report_scope30 --write-pdf        # rebuild docs/REPORT.pdf and its sidecar

``--check`` never writes. It reconciles the report against the committed, accepted evidence only:
- the #14 ``analysis.json`` (results, gaps, comparisons, costs), pinned by SHA-256;
- the #12 freeze manifest (counts, demonstrations, artifact hashes);
- ``prepare_data.CHECKSUMS`` (real split hashes) and the #13 trainer constants (MiniLM, freeze, duplicate);
- the two #14 charts, which must be referenced unchanged.

It also rejects missing sections, unbacked hashes, stale scaffold text and unqualified scientific claims
(significance, stability, replication, causation, equivalence, human agreement). Finally, it verifies that
``docs/REPORT.pdf`` matches its sidecar and that the sidecar was built from the current ``REPORT.md``.

Building needs the optional ``requirements-report.txt`` packages (``markdown``, ``playwright`` with Chromium,
``pypdf``). Checking needs only the standard library, plus ``pypdf`` when the PDF text is checked.
It never runs generation, training, analysis, freezing or Ollama.
"""

import argparse
import base64
import hashlib
import html
import json
import re
import sys
import tempfile
from pathlib import Path

from common import ROOT

DOCS = ROOT / "docs"
REPORT_PATH = DOCS / "REPORT.md"
PDF_PATH = DOCS / "REPORT.pdf"
PDF_SIDECAR_PATH = DOCS / "REPORT.pdf.json"
ANALYSIS_DIR = DOCS / "analysis" / "scope30-v1-seed13"
ANALYSIS_PATH = ANALYSIS_DIR / "analysis.json"
FREEZE_PATH = DOCS / "freeze" / "scope30-v1-seed13.freeze.json"

# Accepted evidence pins (#12, #13, #14). A changed input is a different report, not a silent update.
ACCEPTED_ANALYSIS_SHA256 = "d0486efdceb6cec23d47fc0a943f3307a65fe9230913c38f7f50af5dc26177f6"
ACCEPTED_CHARTS = {
    "f1_by_condition.png": "fc6e2f6acd9d9786d7ebb867cd232b0c757b2745e4eebf264e7a80745ba941e8",
    "gap.png": "5ed188eb1a1a853a6b812fcac218ee024d7986fcdd41266ab162f50a8d4971e3",
}
ACCEPTED_FREEZE_SHA256 = "dd2e52ee704835525df41071ff5214c56593969d248ca8812c23b617bfd74de1"
SIDECAR_SCHEMA = 1
REPOSITORY_DOCS_URL = "https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers/blob/main/docs/"

TASK_NAMES = {"trec": "TREC", "emotion": "Emotion"}
TEST_SIZES = {"trec": 500, "emotion": 2000}
CONDITION_NAMES = {
    "C1": "C1 Real", "C2": "C2 Zero-shot", "C3": "C3 Few-shot", "C4": "C4 Real + few-shot",
    "C5a": "C5a Real sources", "C5b": "C5b Sources + AugGPT",
}
REQUIRED_HEADINGS = (
    "## 1. Introduction",
    "## 2. Related Work",
    "## 3. Tasks, Data and Generation",
    "## 4. Classifier and Experimental Conditions",
    "## 5. Results",
    "## 6. Limitations and Conclusions",
    "## References",
    "## Appendix A.",
)
APPENDIX_BREAK = '<div class="appendix-break"></div>'
AUTHOR_LINE = "**Author:** Kausthuba Belagod (@kausthubabelagodbdes24-oss)"
REVIEW_STATUS = "The author has not yet reviewed or accepted this report or its PDF."
# Surnames that appear only in the superseded scaffold and are not backed by an authoritative record.
UNVERIFIED_NAMES = ("Harshita Jain", "Hementh M ", "Hementh M.", "Hementh M (")
STALE_SCAFFOLD = (
    "Planning length", "*Owner:", "Blocked pending", "will be evaluated once", "Upstream Dependencies",
    "annotation_key.csv", "Krippendorff", "TODO", "TBD", "combined slides", "speaker notes",
)
# Scientific-claim terms that may appear only in a sentence that negates or limits them.
CLAIM_PATTERNS = {
    "significance": r"\bsignifican(?:t|tly|ce)\b|\bp\s*[<=]\s*0?\.\d",
    "confidence interval": r"confidence interval",
    "stability": r"\bstab(?:le|ility)\b|\brobust(?:ly|ness)?\b",
    "replication": r"\breplicat\w*|\breproduced across seeds",
    "causation": r"\bcaus(?:e|es|ed|al|ally|ation)\b",
    "equivalence/meaningfulness": r"\bequivalen\w*|\bmeaningful\w*|\bpractically equal",
    "human agreement": r"inter-annotator|annotator agreement|human agreement|human consensus|\balpha\b",
    "independent human review": r"independent[- ]human|independent human",
    "superiority": r"\boutperform\w*|\bsuperior\w*|\bimprove(?:s|d|ment)?\b|\bdegrad\w*",
}
NEGATION = re.compile(
    r"\b(?:no|not|nor|never|cannot|without|neither|unknown|none|isn't|doesn't|did not|does not|is not|"
    r"are not|was not|were not|rather than|instead of)\b", re.I)
CITATIONS = {
    "Li et al. (2023)": ("Li, Z., Zhu, H., Lu, Z., and Yin, M. (2023)", "pages 10443–10461",
                         "10.18653/v1/2023.emnlp-main.647"),
    "Dai et al. (2023)": ("Dai, H., Liu, Z., Liao, W.,", "arXiv:2302.13007"),
    "Brown et al. (2020)": ("Brown, T. B., Mann, B., Ryder, N.,", "pages 1877–1901", "arXiv:2005.14165"),
    "Reimers and Gurevych (2019)": ("Reimers, N. and Gurevych, I. (2019)", "pages 3982–3992",
                                    "10.18653/v1/D19-1410"),
}


class ReportError(Exception):
    pass


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_lf(path):
    """SHA-256 of a text file with CRLF normalized to LF, so Windows checkouts agree."""
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def _minus(x):
    return "−" if x < 0 else "+"


def fmt_signed(x, places=4):
    return f"{_minus(x)}{abs(x):.{places}f}"


def fmt_pct(x):
    return f"{_minus(x)}{abs(100 * x):.2f}%"


def load_evidence(analysis_path=ANALYSIS_PATH, freeze_path=FREEZE_PATH, pinned=True):
    analysis_bytes = Path(analysis_path).read_bytes()
    if pinned and sha256_bytes(analysis_bytes) != ACCEPTED_ANALYSIS_SHA256:
        raise ReportError(f"{analysis_path} is not the accepted #14 analysis ({ACCEPTED_ANALYSIS_SHA256[:12]}…)")
    if pinned and sha256_lf(freeze_path) != ACCEPTED_FREEZE_SHA256:
        raise ReportError(f"{freeze_path} is not the accepted #12 freeze ({ACCEPTED_FREEZE_SHA256[:12]}…)")
    analysis = json.loads(analysis_bytes.decode("utf-8"))
    freeze = json.loads(Path(freeze_path).read_bytes().decode("utf-8"))
    if analysis.get("generation_mode") != "real" or analysis.get("real_run_evidence") is not True:
        raise ReportError("analysis.json is not real-run evidence")
    if analysis["grid"]["freeze"]["sha256"] != ACCEPTED_FREEZE_SHA256:
        raise ReportError("analysis.json is not bound to the accepted freeze")
    return analysis, freeze


def expected_table1_rows(analysis):
    """Table 1 lines, in canonical order, from the full-precision accepted results."""
    by_key = {(r["task"], r["condition"]): r for r in analysis["results"]}
    if len(by_key) != 12 or len(analysis["results"]) != 12:
        raise ReportError("analysis.json does not hold exactly twelve result keys")
    rows = []
    for task in ("trec", "emotion"):
        for cond in CONDITION_NAMES:
            r = by_key[(task, cond)]
            rows.append(f"| {TASK_NAMES[task]} | {CONDITION_NAMES[cond]} | {r['n_train']} | "
                        f"{r['macro_f1']:.4f} | {r['accuracy']:.4f} |")
    return rows


def _first_cell(label):
    return f"| {label} | " if label else "| | "


def expected_table2_rows(analysis):
    """Table 2 lines (the four planned comparisons) from the accepted gaps and comparisons."""
    results = {(r["task"], r["condition"]): r for r in analysis["results"]}
    gaps = {(g["task"], g["condition"]): g for g in analysis["gaps"]}
    comps = {c["id"]: c for c in analysis["comparisons"]}
    rows = []
    labels = {"C2": "1. C2 − C1 (zero-shot vs real)", "C3": "1. C3 − C1 (few-shot vs real)"}
    for cond in ("C2", "C3"):
        for i, task in enumerate(("trec", "emotion")):
            g = gaps[(task, cond)]
            if g["status"] != "defined":
                raise ReportError(f"{task} {cond}: undefined gap needs a labelled row, not a number")
            acc = results[(task, cond)]["accuracy"] - results[(task, "C1")]["accuracy"]
            rows.append(f"{_first_cell(labels[cond] if i == 0 else '')}{TASK_NAMES[task]} | {fmt_signed(g['absolute_gap'])} | "
                        f"{fmt_pct(g['relative_gap'])} | {fmt_signed(acc)} | "
                        f"{_minus(acc)}{abs(round(acc * TEST_SIZES[task]))} |")
    others = (("few_vs_zero", "2. C3 − C2 (few-shot vs zero-shot)"),
              ("mix_vs_real", "3. C4 − C1 (mix vs real)"),
              ("auggpt_vs_sources", "4. C5b − C5a (AugGPT, 420 vs 60 rows)"))
    for cid, label in others:
        for i, task in enumerate(("trec", "emotion")):
            v = comps[cid]["per_task"][task]
            acc = v["accuracy_diff"]
            rows.append(f"{_first_cell(label if i == 0 else '')}{TASK_NAMES[task]} | {fmt_signed(v['macro_f1_diff'])} | | "
                        f"{fmt_signed(acc)} | {_minus(acc)}{abs(round(acc * TEST_SIZES[task]))} |")
    return rows


def expected_phrases(analysis, freeze):
    """Exact values that the prose must state, each derived from the accepted evidence."""
    comps = {c["id"]: c for c in analysis["comparisons"]}
    c2, c3 = comps["cross_task_gap_C2"], comps["cross_task_gap_C3"]
    phrases = [
        f"Emotion minus TREC is {fmt_signed(100 * c2['emotion_minus_trec_relative'], 2)} percentage points of "
        f"relative gap for C2 ({fmt_signed(c2['emotion_minus_trec_absolute'])} absolute macro-F1) and "
        f"{fmt_signed(100 * c3['emotion_minus_trec_relative'], 2)} points for C3 "
        f"({fmt_signed(c3['emotion_minus_trec_absolute'])})",
        f"The relative gap was {abs(100 * c2['emotion_minus_trec_relative']):.2f} (zero-shot) and "
        f"{abs(100 * c3['emotion_minus_trec_relative']):.2f} (few-shot) percentage points more negative on Emotion",
    ]
    demos = {k: v.get("selected_distinct_demonstrations") for k, v in freeze["artifacts"].items()}
    costs = {(c["task"], c["condition"]): c for c in analysis["costs"]}
    for task in ("trec", "emotion"):
        if costs[(task, "C3")]["real_generation_demonstrations"] != demos[f"{task}/few_shot"]:
            raise ReportError(f"{task}: analysis and freeze disagree on few-shot demonstrations")
    trec_d, emo_d = demos["trec/few_shot"], demos["emotion/few_shot"]
    phrases += [f"**{trec_d}** distinct real demonstrations for TREC and **{emo_d}** for Emotion",
                f"| {trec_d} / {emo_d} |",
                f"at most {trec_d} / {emo_d}"]
    totals = freeze["totals"]
    total = totals["selected_zero_few_shot_texts"] + totals["auggpt_paraphrases"]
    phrases += [f"**{total:,}** generated texts", f"{totals['selected_zero_few_shot_texts']} texts in total",
                f"{totals['auggpt_paraphrases']} in total"]
    for task in ("trec", "emotion"):
        for cond, n in (("C1", 180), ("C2", 180), ("C3", 180), ("C4", 180), ("C5a", 60), ("C5b", 420)):
            got = costs[(task, cond)]
            if got["n_train"] != n:
                raise ReportError(f"{task} {cond}: accepted n_train {got['n_train']} != {n}")
    phrases += ["| C1 | 30 real per label, seed-13 sample of `train` | 180 | 180 | 0 |",
                "| C4 | 15 real + 15 few-shot per label | 180 | 90 | 90 |",
                "| C5a | 10 `lowres_seed13` sources per label | 60 | 60 | 0 |",
                "| C5b | C5a sources + 6 AugGPT paraphrases each | 420 | 60 | 360 |"]
    clf = analysis["grid"]["classifier"]["classifier"]
    phrases.append(f"LogisticRegression(C={clf['C']}, max_iter={clf['max_iter']}, random_state={clf['random_state']})")
    emb = analysis["grid"]["classifier"]["embedding"]
    phrases += [emb["name"], emb["revision"], emb["identity_sha256"], f"batch size {emb['batch_size']}"]
    model = freeze["accepted_model_identity"]
    phrases += [f"`{model['configured_tag']}`", model["digest"]]
    phrases += [analysis["grid"]["grid_complete_sha256"], analysis["grid"]["runs_csv_sha256"],
                analysis["grid"]["revision"], analysis["grid"]["freeze"]["sha256"],
                analysis["analysis_code"]["revision"], ACCEPTED_ANALYSIS_SHA256]
    phrases += list(analysis["outputs_sha256"][k] for k in ACCEPTED_CHARTS)
    for key, batch in freeze["issue11_evidence"]["latest_round_by_batch"].items():
        phrases.append(f"| {key}{' (v2, promoted)' if key == 'trec/auggpt' else ''} | {batch['latest_round']} | "
                       f"{batch['clear_errors']} | {batch['ambiguous']} | {batch['outcome']} |")
    return phrases


def known_hashes(analysis, freeze):
    """Every 64-hex value the report may cite: those present in accepted evidence or pinned in code."""
    from ws2_training import prepare_data, train_scope30
    blobs = [json.dumps(analysis), json.dumps(freeze), json.dumps(prepare_data.CHECKSUMS),
             Path(train_scope30.__file__).read_text(encoding="utf-8"),
             ACCEPTED_ANALYSIS_SHA256, " ".join(ACCEPTED_CHARTS.values())]
    known = set()
    for b in blobs:
        known |= set(re.findall(r"\b[0-9a-f]{64}\b", b))
    return known


def known_revisions(analysis, freeze):
    from ws2_training import train_scope30
    blobs = [json.dumps(analysis), json.dumps(freeze), Path(train_scope30.__file__).read_text(encoding="utf-8")]
    return set().union(*(set(re.findall(r"\b[0-9a-f]{40}\b", b)) for b in blobs))


def split_parts(text):
    """(main body up to References, references, appendix)."""
    i, j = text.find("## References"), text.find("## Appendix A.")
    if i < 0 or j < 0 or j < i:
        raise ReportError("report must contain '## References' followed by '## Appendix A.'")
    return text[:i], text[i:j], text[j:]


def _prose_sentences(text):
    """Sentences outside code fences, code spans, links, image references and table hash cells."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    for block in re.split(r"\n\s*\n|\n(?=\s*(?:[-*]|\d+\.|\|)\s)", text):
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z*(])", block):
            yield " ".join(sentence.split())


def claim_problems(text):
    problems = []
    for sentence in _prose_sentences(text):
        for name, pattern in CLAIM_PATTERNS.items():
            if re.search(pattern, sentence, re.I) and not NEGATION.search(sentence):
                problems.append(f"unqualified {name} wording: {sentence[:160]!r}")
    return problems


def validate_report(text, analysis, freeze, report_dir=DOCS):
    """Return a list of problems; an empty list means the report reconciles with the accepted evidence."""
    problems = []
    if "\r" in text:
        text = text.replace("\r\n", "\n")
    # Structure and separation.
    positions = []
    for heading in REQUIRED_HEADINGS:
        found = [m.start() for m in re.finditer("^" + re.escape(heading), text, re.M)]
        if len(found) != 1:
            problems.append(f"heading {heading!r} must appear exactly once (found {len(found)})")
        positions.append(found[0] if found else -1)
    if all(p >= 0 for p in positions) and positions != sorted(positions):
        problems.append("sections are out of order")
    if APPENDIX_BREAK not in text or (text.find(APPENDIX_BREAK) > text.find("## Appendix A.") >= 0):
        problems.append("Appendix A must start after an explicit page break")
    try:
        body, refs, appendix = split_parts(text)
    except ReportError as exc:
        return problems + [str(exc)]
    # Author and review status.
    if AUTHOR_LINE not in text:
        problems.append("author line must name Kausthuba Belagod (@kausthubabelagodbdes24-oss)")
    if REVIEW_STATUS not in text:
        problems.append("report must state that the author has not yet reviewed or accepted it")
    for name in UNVERIFIED_NAMES:
        if name in text:
            problems.append(f"unverified author name {name.strip()!r} (not in an authoritative record)")
    for stale in STALE_SCAFFOLD:
        if stale in text:
            problems.append(f"stale scaffold/out-of-scope text: {stale!r}")
    # Exact tables and values.
    for row in expected_table1_rows(analysis):
        if row not in body:
            problems.append(f"Table 1 row missing or altered: {row}")
    table1 = re.findall(r"^\| (?:TREC|Emotion) \| C\d[ab]? [^|]+\| \d+ \| \d\.\d{4} \| \d\.\d{4} \|$", body, re.M)
    if len(table1) != 12:
        problems.append(f"Table 1 must have exactly twelve result rows (found {len(table1)})")
    for row in expected_table2_rows(analysis):
        if row not in body:
            problems.append(f"Table 2 row missing or altered: {row}")
    for phrase in expected_phrases(analysis, freeze):
        if phrase not in text:
            problems.append(f"required value missing or altered: {phrase!r}")
    for n in (1, 2, 3, 4):
        if f"**Comparison {n}:" not in body:
            problems.append(f"planned comparison {n} is not discussed")
    for label in ("Table 1", "Table 2", "Figure 1", "Figure 2"):
        if body.count(label) < 2:
            problems.append(f"{label} must be captioned and referenced from the text")
    # Charts: both accepted charts, unchanged, embedded from their committed location.
    for name, digest in ACCEPTED_CHARTS.items():
        ref = f"](analysis/scope30-v1-seed13/{name})"
        if body.count(ref) != 1:
            problems.append(f"chart {name} must be embedded exactly once from analysis/scope30-v1-seed13/")
        path = Path(report_dir) / "analysis" / "scope30-v1-seed13" / name
        if not path.is_file() or sha256_bytes(path.read_bytes()) != digest:
            problems.append(f"chart {name} is missing or differs from the accepted #14 chart")
        if analysis["outputs_sha256"].get(name) != digest:
            problems.append(f"chart {name} is not the chart recorded in analysis.json")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    if sorted(images) != sorted(f"analysis/scope30-v1-seed13/{n}" for n in ACCEPTED_CHARTS):
        problems.append(f"unexpected images in the report: {images}")
    # Identifiers: every cited hash/revision must come from accepted evidence or pinned code.
    hashes = known_hashes(analysis, freeze)
    for h in sorted(set(re.findall(r"\b[0-9a-f]{64}\b", text)) - hashes):
        problems.append(f"hash not backed by accepted evidence: {h}")
    from ws2_training import prepare_data
    for key in ("trec/train.jsonl", "trec/pool.jsonl", "trec/test.jsonl", "trec/lowres_seed13.jsonl",
                "emotion/train.jsonl", "emotion/pool.jsonl", "emotion/test.jsonl", "emotion/lowres_seed13.jsonl"):
        if prepare_data.CHECKSUMS[key]["sha256"] not in text:
            problems.append(f"canonical split hash for {key} is missing")
    revisions = known_revisions(analysis, freeze)
    for rev in sorted(set(re.findall(r"(?<![0-9a-f])[0-9a-f]{40}(?![0-9a-f])", text)) - revisions - hashes):
        if not any(rev in h for h in hashes):
            problems.append(f"revision not backed by accepted evidence: {rev}")
    # Scientific wording.
    problems += claim_problems(body + "\n" + appendix)
    for required in ("model-assisted", "Option A", "trec.train.05262", "trec.test.00378", "seed 13",
                     "not independent human annotation", "C5b has seven times as many training rows as C5a",
                     "No equivalence or meaningfulness threshold was defined"):
        if required not in text:
            problems.append(f"required disclosure missing: {required!r}")
    # Citations: every in-text citation resolves to a complete reference, and vice versa.
    for cite, needles in CITATIONS.items():
        if cite not in body:
            problems.append(f"citation {cite} is not used in the text")
        if any(n not in refs for n in needles):
            problems.append(f"reference for {cite} is incomplete or inconsistent")
    entries = [e for e in refs.split("\n\n")[1:] if e.strip() and not e.startswith(("Datasets:", "<"))]
    if len(entries) != len(CITATIONS):
        problems.append(f"references must list exactly {len(CITATIONS)} cited works (found {len(entries)})")
    for author, year in re.findall(r"\b([A-Z][a-z]+(?: et al\.| and [A-Z][a-z]+)) \((\d{4})\)", body):
        if f"{author} ({year})" not in CITATIONS:
            problems.append(f"in-text citation without a reference: {author} ({year})")
    return problems


# --------------------------------------------------------------------------------------- PDF build/check

CSS = """
@page { size: A4; margin: 17mm 17mm 18mm 17mm; }
html { font-family: "DejaVu Serif", Cambria, Georgia, "Times New Roman", serif; font-size: 10pt;
       line-height: 1.38; color: #111; }
body { margin: 0; }
h1 { font-size: 15pt; line-height: 1.25; margin: 0 0 8pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 5pt; border-bottom: 0.6pt solid #999; padding-bottom: 2pt;
     break-after: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 4pt; break-after: avoid; }
h4 { font-size: 10pt; margin: 8pt 0 3pt; break-after: avoid; }
h5 { font-size: 9.5pt; margin: 6pt 0 2pt; break-after: avoid; }
p, li { text-align: left; }
p { margin: 0 0 5pt; } ul, ol { margin: 0 0 5pt; padding-left: 16pt; } li { margin: 0 0 2pt; }
code { font-family: "DejaVu Sans Mono", Consolas, "Courier New", monospace; font-size: 8.2pt;
       overflow-wrap: anywhere; word-break: break-word; }
pre { background: #f5f5f5; border: 0.5pt solid #ccc; padding: 4pt; white-space: pre-wrap;
      overflow-wrap: anywhere; font-size: 6.9pt; line-height: 1.28; }
pre code { font-size: 6.9pt; }   /* the longest template line (113 characters) fits without wrapping */
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt; font-size: 8.6pt; break-inside: auto; }
tr { break-inside: avoid; }
th, td { border: 0.5pt solid #888; padding: 2.2pt 4pt; vertical-align: top; text-align: left; }
th { background: #eee; }
td code, th code { font-size: 7.4pt; word-break: normal; overflow-wrap: anywhere; }
img { display: block; max-width: 100%; max-height: 68mm; width: auto; height: auto; margin: 6pt auto 2pt;
      break-inside: avoid; }
p:has(> img) { text-align: center; break-inside: avoid; }
.appendix-break { break-before: page; }
p.caption { break-after: avoid; }
"""


def render_html(text, report_dir=DOCS):
    import markdown
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"], output_format="html5")

    def inline_image(match):
        src = match.group(1)
        path = (Path(report_dir) / src).resolve()
        data = base64.b64encode(path.read_bytes()).decode("ascii")      # exact PNG bytes, never re-encoded
        return f'src="data:image/png;base64,{data}"'

    body = re.sub(r'src="(analysis/scope30-v1-seed13/[a-z_0-9]+\.png)"', inline_image, body)
    body = re.sub(r"<p><strong>(Table \d+\.)", r'<p class="caption"><strong>\1', body)   # keep captions with tables
    # Relative repository links become absolute GitHub links, so the PDF never points at the builder's disk.
    body = re.sub(r'href="(?![a-z]+:|#)([^"]+)"', lambda m: f'href="{REPOSITORY_DOCS_URL}{m.group(1)}"', body)
    return ("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{html.escape('SynthText scope30-v1 exploratory report')}</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


FIXED_PDF_DATE = b"D:20261101000000+00'00'"


def _normalize_pdf_dates(pdf):
    """Replace Chromium's build timestamps with a fixed, equal-length date so rebuilds are comparable."""
    return re.sub(rb"D:\d{14}(?:[+\-Z]\d{2}'\d{2}'|Z|[+\-]\d{2}'\d{2})?",
                  lambda m: FIXED_PDF_DATE[:len(m.group(0))].ljust(len(m.group(0)), b"'"), pdf)


def build_pdf(out_path, report_path=REPORT_PATH):
    from playwright.sync_api import sync_playwright
    text = Path(report_path).read_text(encoding="utf-8")
    page_html = render_html(text, Path(report_path).parent)
    with tempfile.TemporaryDirectory() as tmp:
        html_path = Path(tmp) / "report.html"
        html_path.write_text(page_html, encoding="utf-8")
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.as_uri(), wait_until="load")
            pdf = page.pdf(format="A4", print_background=True, display_header_footer=True,
                           header_template="<span></span>",
                           footer_template=('<div style="font-size:8px;width:100%;text-align:center;'
                                            'color:#555;">SynthText scope30-v1 report — page '
                                            '<span class="pageNumber"></span> of <span class="totalPages">'
                                            '</span></div>'),
                           margin={"top": "17mm", "bottom": "18mm", "left": "17mm", "right": "17mm"})
            browser.close()
    pdf = _normalize_pdf_dates(pdf)
    Path(out_path).write_bytes(pdf)
    return pdf


def pdf_text_pages(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def _flat(s):
    return " ".join(s.replace("−", "-").replace("\u00ad", "").split())


def check_pdf_text(pages, analysis):
    """The PDF must show every Table 1/Table 2 value and the key identifiers of the source Markdown."""
    problems = []
    flat = _flat(" ".join(pages))
    for r in analysis["results"]:
        for value in (f"{r['macro_f1']:.4f}", f"{r['accuracy']:.4f}"):
            if value not in flat:
                problems.append(f"PDF is missing {r['task']} {r['condition']} value {value}")
    for g in analysis["gaps"]:
        for value in (_flat(fmt_pct(g["relative_gap"])), _flat(fmt_signed(g["absolute_gap"]))):
            if value not in flat:
                problems.append(f"PDF is missing gap value {value}")
    for needle in ("1. Introduction", "6. Limitations and Conclusions", "References", "Appendix A.",
                   "Figure 1", "Figure 2", "Table 1", "Table 2", "Kausthuba Belagod", "dd2e52ee"):
        if needle not in flat:
            problems.append(f"PDF is missing {needle!r}")
    if any(not p.strip() for p in pages):
        problems.append("PDF has a blank page")
    if "\ufffd" in flat:
        problems.append("PDF text contains replacement characters (missing glyphs)")
    order = [flat.find(h) for h in ("1. Introduction", "2. Related Work", "3. Tasks, Data and Generation",
                                    "4. Classifier and Experimental Conditions", "5. Results",
                                    "6. Limitations and Conclusions", "Appendix A.")]
    if -1 not in order and order != sorted(order):
        problems.append("PDF sections are out of order")
    return problems


def sidecar_for(pdf_bytes, report_path, pages):
    import platform
    from importlib import metadata

    def version(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    return {
        "schema_version": SIDECAR_SCHEMA,
        "report": "docs/REPORT.md",
        "report_sha256_lf": sha256_lf(report_path),
        "pdf": "docs/REPORT.pdf",
        "pdf_sha256": sha256_bytes(pdf_bytes),
        "pdf_bytes": len(pdf_bytes),
        "pages": pages,
        "builder": "python -m ws3_evaluation.report_scope30 --write-pdf",
        "environment": {"python": platform.python_version(), "os": platform.system(),
                        "markdown": version("markdown"), "playwright": version("playwright"),
                        "pypdf": version("pypdf")},
        "review_status": "not reviewed or accepted by the author",
    }


def check_pdf(analysis, pdf_path=PDF_PATH, sidecar_path=PDF_SIDECAR_PATH, report_path=REPORT_PATH,
              require_text=True):
    problems = []
    if not Path(pdf_path).is_file() or not Path(sidecar_path).is_file():
        return [f"missing {pdf_path.name} or its sidecar {sidecar_path.name}"]
    sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
    data = Path(pdf_path).read_bytes()
    if sidecar.get("schema_version") != SIDECAR_SCHEMA:
        problems.append("PDF sidecar schema mismatch")
    if sha256_bytes(data) != sidecar.get("pdf_sha256") or len(data) != sidecar.get("pdf_bytes"):
        problems.append("REPORT.pdf does not match the SHA-256/size recorded in REPORT.pdf.json")
    if sha256_lf(report_path) != sidecar.get("report_sha256_lf"):
        problems.append("REPORT.pdf is stale: REPORT.md changed after the PDF was built (rebuild it)")
    if not data.startswith(b"%PDF-"):
        problems.append("REPORT.pdf is not a PDF")
    if b"file://" in data:
        problems.append("REPORT.pdf links to a local file:// path")
    try:
        pages = pdf_text_pages(pdf_path)
    except ImportError:
        if require_text:
            problems.append("pypdf is required to check the PDF text (pip install -r requirements-report.txt)")
        return problems
    if len(pages) != sidecar.get("pages"):
        problems.append(f"PDF has {len(pages)} pages but the sidecar records {sidecar.get('pages')}")
    problems += check_pdf_text(pages, analysis)
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="read-only reconciliation of report, charts and PDF")
    mode.add_argument("--build-pdf", metavar="OUT", help="build a PDF of docs/REPORT.md at OUT (no sidecar)")
    mode.add_argument("--write-pdf", action="store_true", help="rebuild docs/REPORT.pdf and REPORT.pdf.json")
    ap.add_argument("--skip-pdf-text", action="store_true", help="with --check: do not require pypdf")
    args = ap.parse_args(argv)
    try:
        analysis, freeze = load_evidence()
        problems = validate_report(REPORT_PATH.read_text(encoding="utf-8"), analysis, freeze)
        if args.check:
            problems += check_pdf(analysis, require_text=not args.skip_pdf_text)
            for p in problems:
                print(f"FAIL {p}")
            if problems:
                print(f"FAIL report check: {len(problems)} problem(s)")
                return 1
            sidecar = json.loads(PDF_SIDECAR_PATH.read_text(encoding="utf-8"))
            print(f"PASS report reconciles with accepted evidence (analysis {ACCEPTED_ANALYSIS_SHA256[:12]}…, "
                  f"freeze {ACCEPTED_FREEZE_SHA256[:12]}…)")
            print(f"PASS REPORT.pdf sha256 {sidecar['pdf_sha256']} ({sidecar['pages']} pages) built from "
                  f"REPORT.md sha256(LF) {sidecar['report_sha256_lf']}")
            return 0
        if problems:
            for p in problems:
                print(f"FAIL {p}")
            print("FAIL refusing to build a PDF from a report that does not reconcile")
            return 1
        out = Path(args.build_pdf) if args.build_pdf else PDF_PATH
        pdf = build_pdf(out)
        pages = pdf_text_pages(out)
        text_problems = check_pdf_text(pages, analysis)
        for p in text_problems:
            print(f"FAIL {p}")
        if text_problems:
            return 1
        if args.write_pdf:
            sidecar = sidecar_for(pdf, REPORT_PATH, len(pages))
            PDF_SIDECAR_PATH.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                        newline="\n")
        print(f"WROTE {out} sha256 {sha256_bytes(pdf)} ({len(pages)} pages)")
        if not args.write_pdf and PDF_SIDECAR_PATH.is_file():
            recorded = json.loads(PDF_SIDECAR_PATH.read_text(encoding="utf-8"))["pdf_sha256"]
            same = "IDENTICAL to" if recorded == sha256_bytes(pdf) else "DIFFERENT from"
            print(f"NOTE rebuilt PDF is {same} the committed REPORT.pdf ({recorded[:12]}…); fonts and Chromium "
                  "versions differ between machines, so compare page renders rather than bytes across platforms")
        return 0
    except ReportError as exc:
        print(f"FAIL {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
