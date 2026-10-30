# Instructions for every SynthText implementation agent

**Current lead instruction, 30 October 2026:** Kausthuba (@kausthubabelagodbdes24-oss) is the sole implementing operator for generation, training and evaluation. Keep the workspaces separate; historical teammate ownership below does not require waiting for those teammates to execute tickets. Guide Kausthuba through physical setup, real runs and human quality review. This operating change does not turn one person's repeated judgments into independent annotators; requirements involving genuinely independent humans still need an explicit scientific decision.

This repository is the central project record for every provider. Explicitly read [the shared protocol](docs/agents/IMPLEMENTATION_PROTOCOL.md) before editing, then use [the issue prompt](docs/agents/ISSUE_PROMPT.md) to launch tasks. Claude and Gemini entry files point to these same rules, without separate policies.

Read live issues and blockers, current main, README, specification, glossary, relevant decisions, contracts, code and tests. Planned requirements are not proof of implemented behavior. Do not rely on another chat or invent missing decisions.

One issue, one active implementer, one isolated checkout and unique branch. Respect CODEOWNERS and workstream ownership. Shared helpers/configuration and cross-workstream interfaces require lead approval and the required group coordination. No force-push, history rewrite, branch deletion, release tagging or post-freeze regeneration without explicit approval.

Never commit real data, secrets, environments or unreviewed outputs; never fabricate annotations, results, approvals or test success. Human annotations must remain independent. Follow the approved repetition, exact-count, audit, freeze and paired-gap requirements.

Run relevant behavior checks and pipeline tests, open a scoped PR into main, and verify CI at its current head. Leave merging to the lead or an explicitly authorized integrator; use merge commits. Leave issues open while any acceptance criterion remains unmet. Persist handoffs and consequential decisions in GitHub.

Owners: generation @harshitadjainbdes24-byte; training and lead @kausthubabelagodbdes24-oss; evaluation @hem4456.
