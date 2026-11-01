# Instructions for every SynthText implementation agent

**Current lead-approved scope (31 October 2026):** 30 examples/label, seed 13 only, both tasks and all six conditions; twelve result keys. Independent-human annotation, distribution and external report-sharing gates are removed. Read [ADR 0012](docs/adr/0012-eod-exploratory-30-seed13.md) and scope migration before resuming old work. Conflicting earlier requirements are historical.

Accountable workstream ownership follows the original project plan: generation is owned by Harshita (@harshitadjainbdes24-byte), training and project lead by Kausthuba (@kausthubabelagodbdes24-oss), and evaluation by Hementh (@hem4456). Keep workspaces separate. Kausthuba may implement or integrate work on another owner's behalf, but that does not transfer accountable ownership. See [ADR 0015](docs/adr/0015-restore-original-workstream-ownership.md) and the [migration](docs/SCOPE_MIGRATION.md).

This repository is the central project record for every provider. Explicitly read [the shared protocol](docs/agents/IMPLEMENTATION_PROTOCOL.md) before editing, then use [the issue prompt](docs/agents/ISSUE_PROMPT.md) to launch tasks. Claude and Gemini entry files point to these same rules, without separate policies.

Read live issues and blockers, current main, README, specification, glossary, relevant decisions, contracts, code and tests. Planned requirements are not proof of implemented behavior. Do not rely on another chat or invent missing decisions.

One issue, one active implementer, one isolated checkout and unique branch. Respect CODEOWNERS and workstream ownership. Shared helpers/configuration and cross-workstream interfaces require lead approval and the required group coordination. No force-push, history rewrite, branch deletion, release tagging or post-freeze regeneration without explicit approval.

Never commit real data, secrets, environments or unreviewed outputs; never fabricate annotations, results, approvals or test success. Never describe model-assisted judgments as independent humans or human agreement. Follow current exact-count/source, quality, freeze and single-seed paired-gap rules.

Professor-facing presentation slides and submission logistics are intentionally outside the repository and GitHub issue scope. Do not create, attach, review or track the final slide deck in this project. Repository work may produce reviewed reports, comparison tables, charts and a reproducibility notebook that the owner can cite separately.

Run relevant behavior checks and pipeline tests, open a scoped PR into main, and verify CI at its current head. Leave merging to the lead or an explicitly authorized integrator; use merge commits. Leave issues open while any acceptance criterion remains unmet. Persist handoffs and consequential decisions in GitHub.

Owners: generation @harshitadjainbdes24-byte; training and project lead @kausthubabelagodbdes24-oss; evaluation @hem4456. Authorized implementation on an owner's behalf must retain both the accountable owner and the actual implementing account in the handoff.
