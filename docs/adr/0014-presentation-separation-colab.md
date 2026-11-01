# ADR 0014: Separate the professor-facing presentation and provide a Colab walkthrough

- **Status:** Accepted
- **Date:** 1 November 2026
- **Decision owner:** Kausthuba

## Context

The repository already produces the development evidence needed to explain the experiment: methods, provenance, a reviewed report, comparison tables, charts and a single-seed interpretation. Earlier planning also placed the final presentation deck and submission logistics inside GitHub issues. The owner wants the professor-facing presentation to remain separate from development while retaining an easy, executable way to demonstrate how generation, training and analysis work.

## Decision

1. The final presentation deck, speaker narrative and submission logistics are private owner deliverables outside this repository and GitHub issue tracker. Issues must not request, attach, review or claim completion of those slides.
2. Repository development continues to produce reviewed reports, appendices, comparison tables, charts and machine-readable provenance.
3. `notebooks/SynthText_scope30_walkthrough.ipynb` is the supported Google Colab-compatible walkthrough.
4. The notebook separates two evidence lanes:
   - a safe fake-fixture generation/training demonstration using the repository's fake generator and hashing embedder;
   - read-only display of the committed, accepted real analysis artifacts.
5. The notebook must not run Ollama, regenerate/promote/freeze real data, train the accepted real grid, or describe fake outputs as experimental results. The exact Qwen/Ollama run remains evidenced by its recorded local provenance; it is not reproducible from public Colab inputs.

## Consequences

- Issues #15 and #16 cover methods, limitations and report material without slide deliverables.
- Issue #17 covers the reviewed report/PDF.
- Issue #18 covers the Colab walkthrough and its documentation/tests.
- Reports, tables and charts may be reused by the owner when preparing a presentation elsewhere, but repository completion does not prove presentation or submission.
- Release tags remain separately authorized.
