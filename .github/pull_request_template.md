## Issue and owner

Related to #

Accountable human/workstream; base main SHA; relevant decisions:

## Delivered behavior and acceptance evidence

Map every acceptance criterion to evidence. List unmet human/operational criteria explicitly.

## Verification

Commands actually run, environment, outcomes and limitations. Separate fake checks from real results. Check CI at the current PR head.

## Contracts and handoff

Producer/consumer changes, contract-register updates and required owner/lead approvals (or none). Next owner's actionable handoff:

## Integration checks

- [ ] Scoped changes only; no secrets, real data, environments or unrelated outputs.
- [ ] Relevant tests pass; unrun tests and outstanding criteria are disclosed.
- [ ] Shared-file approvals and affected-owner coordination are recorded.
- [ ] Current main reconciled without rewriting shared history.
- [ ] Affected specification, decisions, glossary and contracts updated.

Lead or explicitly authorized integrator merges with a merge commit after review and current-head CI. Do not close incomplete issues.
