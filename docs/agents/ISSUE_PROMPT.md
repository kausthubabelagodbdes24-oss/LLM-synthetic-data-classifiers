# Reusable task prompt

Replace ISSUE_NUMBER below with the GitHub issue number and paste the prompt into any model. Give it authenticated repo access or a complete checkout, never credentials in chat. No other project conversation is required.

---

Implement issue ISSUE_NUMBER in repository https://github.com/kausthubabelagodbdes24-oss/LLM-synthetic-data-classifiers.

GitHub is the central project context. Fetch the full live issue/comments, linked PRs, blockers and current main. Explicitly read AGENTS.md and docs/agents/IMPLEMENTATION_PROTOCOL.md even if your provider does not auto-load them. Then read README.md, docs/PROJECT_SPEC.md, CONTEXT.md, relevant docs/adr/ decisions, docs/agents/CONTRACTS.md, CODEOWNERS, affected code/tests and CI. Record main's SHA. Planned requirements are not evidence that legacy code implements them.

Check existing claims/PRs and the accountable human owner. Post a short context receipt with your scope, branch, decisions, blockers and verification plan. Do not silently take someone else's prerequisite. If blocked, state why and limit work to independent investigation. If repo access is unavailable, request access or a complete current snapshot and state verification limitations; never invent context.

Use an isolated checkout/worktree and unique issue branch from current main. Preserve local changes and concurrent work. Implement the smallest complete acceptance-criteria slice under the shared protocol. Coordinate shared helpers/settings and producer/consumer interfaces before changing them. Record accepted contracts and decisions in GitHub so other agents agree. Ask the lead to resolve conflicting requirements and scientific unknowns rather than guessing.

Read ADR0012 and docs/SCOPE_MIGRATION.md: 30/label, seed13 only, both tasks, six conditions, twelve result keys. Follow exact-count/source checks, disclosed model-assisted quality review, freeze and single-seed paired gaps. Independent-human annotation and distribution/report-sharing are removed. Preserve originals; use derived selection manifests and fresh scope-specific result storage. Do not fabricate human judgments, results, approvals or test outcomes. Do not commit real data, secrets, environments or unrelated changes. No unapproved shared-setting changes, force-pushes, history rewrites, branch deletion, release tags or frozen-data regeneration.

The final professor-facing presentation deck and submission logistics are private owner work outside GitHub. Issues may deliver reports, tables, charts and the Colab walkthrough, but must not request, attach or track final slides.

Run meaningful behavior tests and required fake-data pipeline checks using Python 3.11. Distinguish fake tests from real model/data evidence. Review the full diff, update affected durable docs and open a PR into main using its template. State evidence per acceptance criterion, test commands/results, contract effects and remaining human/operational requirements. Verify current-head CI. Leave the PR open for the lead or an explicitly authorized integrator; do not merge or close an incomplete issue yourself.

Finish with issue/PR links, branch/commit, completed criteria with evidence, tests actually run, blockers and an actionable next-owner handoff. Persist that handoff on GitHub so the next model needs no private chat context.

---
