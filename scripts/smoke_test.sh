#!/usr/bin/env bash
# Run the whole pipeline on the FAKE sample data with the fake LLM and the
# offline hashing embedder. Takes under a minute, needs no GPU and no internet.
# Use it to check the plumbing after a change; never report its numbers.
set -euo pipefail
cd "$(dirname "$0")/.."

export SYNTHTEXT_DATA=data/sample
export SYNTHTEXT_RESULTS=results/sample
export SYNTHTEXT_FIGURES=results/sample/figures
export SYNTHTEXT_EMBEDDER=hashing

rm -rf results/sample data/sample/synthetic
python scripts/make_sample_data.py

for setting in zero_shot few_shot; do
  python -m workstreams.ws1_generation.generate --task all --setting "$setting" --per-label 8 --dry-run
  python -m workstreams.ws1_generation.filter --task all --setting "$setting"
done
python -m workstreams.ws1_generation.generate --task all --setting auggpt --dry-run
python -m workstreams.ws1_generation.filter --task all --setting auggpt

python -m workstreams.ws2_training.run_matrix --task all --n-train 30

python -m workstreams.ws3_evaluation.metrics --task all
python -m workstreams.ws3_evaluation.subjectivity make-sheet
python scripts/fake_annotations.py   # fills the sheet + a fake survey so scoring can run
python -m workstreams.ws3_evaluation.subjectivity score --sheet results/sample/annotation/annotation_filled.csv
python -m workstreams.ws3_evaluation.subjectivity survey --csv results/sample/annotation/survey.csv
python -m workstreams.ws3_evaluation.analyze

echo "Smoke test passed. Outputs in results/sample/"
