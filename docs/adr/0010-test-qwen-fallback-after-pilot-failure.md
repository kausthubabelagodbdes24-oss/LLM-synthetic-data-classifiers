# Test the documented Qwen fallback after the Llama quality screen

On 30 October 2026, after the initial Llama pilot and both permitted prompt-refinement rounds still showed human-reviewed quality problems, Kausthuba explicitly approved: "Approve the qwen3:4b fallback pilot". Kausthuba is the sole operator across the separate workspaces.

Activate the fallback already proposed in the specification for an isolated pilot only. Use `qwen3:4b` with the existing final focused prompts, six-label definitions, temperature 0.9, three real few-shot demonstrations, and seed 13. Verify the actual model digest, JSON compatibility and measured timing, generate 10 accepted rows per label for both tasks/settings (240 total), and obtain a new human quality inspection. Do not treat a larger model or software test success as evidence of semantic acceptance.

This approval does not authorize further prompt refinement, a silent generation-option change, full generation, training on an unaccepted corpus, a relaxed quality gate or a final 100-versus-60 target decision. If default thinking or other runtime behavior prevents compatibility or completion, record the actual failure and seek a specific decision before altering the experimental settings.

Preserve all three Llama pilot rounds, provenance, submitted judgments and appendix selections. Store the Qwen pilot separately under local `.ollama/pilot-qwen3-4b/data`; copy only checksum-verified real inputs, never Llama-generated rows. Model identity changes must continue to fail closed across resume. The branch configuration selects the candidate for this pilot; it does not establish acceptance on main.

If the fallback pilot is accepted later, one approved generator must be used consistently for both tasks, all three synthetic settings and all three repetitions. Earlier Llama rows remain failed-pilot evidence, not Qwen training data. Keep issue #6 and dependent work blocked until the relevant acceptance decision is recorded.
