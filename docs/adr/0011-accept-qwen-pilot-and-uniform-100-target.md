# Accept Qwen with disclosed pilot limitations and use 100 examples per label

On 31 October 2026, after reviewing the Qwen pilot, Kausthuba explicitly selected: "Accept Qwen; use 100 per label (recommended)". This supersedes ADR 0010's pilot-only limitation for generator selection and target choice, while preserving its model-isolation and historical-evidence requirements.

Use the accepted `qwen3:4b` generator consistently for both tasks, all three synthetic settings and all three repetitions (13, 42, 7). The observed model digest is `359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7`; temperature remains 0.9 and Qwen's default thinking behavior was retained. Keep the final focused prompts; this decision does not approve another refinement or hidden generation-option change.

The uniform C1–C4 target is 100 examples per label (600 per task/condition/repetition). C4 uses 50 real plus 50 few-shot examples per label. C5a remains 60 real examples; C5b remains 420 examples after six valid paraphrases of each of those exact 60 sources. Do not silently mix targets or re-use generated rows between repetitions.

The pilot produced 240 real texts, exactly ten per label in four artifacts. The 40-item human review recorded 30 Good, 6 Ambiguous and 4 Bad, with one Bad item per group. Accept the pilot with these disclosed limitations; do not claim flawless text or a population error estimate. Preserve reviewed text and judgments unchanged, including the eight selected appendix examples. The later exact-completeness, 180-item content audit, investigation and freeze requirements remain in force.

Accepted Qwen pilot rows may count toward full targets only in their own task, setting and seed-13 artifact, using the existing fingerprint and sidecar. The accepted local data root is `.ollama/pilot-qwen3-4b/data` under the lead's project folder; producer/consumer commands must explicitly select it with `SYNTHTEXT_DATA`. Earlier Llama outputs and all four human submissions remain historical evidence and are not Qwen training data.

The measured Qwen pilot took 1490.374461 seconds over 24 calls. Linear extrapolation estimates about 12 hours of remaining zero/few-shot work at 100 per label. AugGPT's additional 2,160 paraphrases remain untimed and must be measured/monitored against the 1 November 2026, 18:00 IST freeze. Estimates are not guaranteed completion times. Any deadline-driven target revision requires an explicit coordinated decision before affected full generation and matched baseline reruns where necessary.

This decision satisfies the human pilot/target requirements of issue #6, subject to passing integration review and CI. After merging its implementation, issues #7, #9 and #10 may proceed under their own acceptance criteria; none is already implemented or complete merely because the prerequisite was accepted. Independent human annotation under the sole-operator arrangement remains a separate unresolved scientific decision.
