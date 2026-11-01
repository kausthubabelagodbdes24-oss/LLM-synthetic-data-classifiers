# Reproducibility guide

This guide provides a clean route through the committed project without requiring private datasets, local model files, or accepted generated corpora.

## Fastest route: Google Colab

Open [the walkthrough in Google Colab](https://colab.research.google.com/github/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers/blob/main/notebooks/SynthText_scope30_walkthrough.ipynb).

The notebook keeps two evidence lanes separate:

1. **Fake demonstration:** committed fixtures exercise zero-shot, few-shot, and AugGPT generation followed by hashing-based training and analysis.
2. **Accepted real results, read only:** committed tables, provenance, limitations, and charts are hash-verified and displayed without retraining.

Fake outputs demonstrate software behavior only. They are not research results.

## Local verification

Requirements:

- Git
- Python 3.11

### Windows PowerShell

```powershell
git clone https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers.git
cd LLM-synthetic-data-classifiers
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest
```

### macOS or Linux

```bash
git clone https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers.git
cd LLM-synthetic-data-classifiers
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-test.txt
.venv/bin/python -m pytest
```

CI runs the same test suite on Python 3.11 under Ubuntu and Windows.

## Verify the accepted report

Install the report dependencies, then run the read-only reconciliation check:

```bash
python -m pip install -r requirements-report.txt
python -m ws3_evaluation.report_scope30 --check
```

This confirms that the report, PDF sidecar, accepted analysis, figures, freeze identity, and reported values agree.

## Safe local pipeline demonstration

The committed `data/sample/real/` fixtures can exercise the legacy demonstration pipeline without Ollama or private data. The Colab notebook is the recommended interface because it sets the safe environment explicitly and labels all outputs.

The demonstration uses:

- `SYNTHTEXT_GENERATION_MODE=fake`
- `SYNTHTEXT_EMBEDDER=hashing`
- `data/sample/` fixture paths
- `results/sample/` ignored output paths

It runs all three generation modes, classifier training, and analysis. Its scores have no scientific meaning.

## Accepted experiment provenance

The real experiment used local resources that are intentionally not published:

- downloaded real datasets;
- the approved local Qwen/Ollama model identity;
- generated corpora and sidecars;
- the pinned MiniLM model cache;
- freeze-bound training results.

The repository instead commits the accepted manifests, hashes, selected IDs, model identities, review evidence, result summaries, and verification code. See:

- [Final specification](PROJECT_SPEC.md)
- [Freeze overview](freeze/README.md)
- [Analysis summary](analysis/scope30-v1-seed13/summary.md)
- [Technical provenance archive](archive/TECHNICAL_PROVENANCE.md)

The real pipeline must not be reconstructed by substituting a different generator or silently changing frozen inputs. Such a run would be a new experiment, not a reproduction of the accepted study.
