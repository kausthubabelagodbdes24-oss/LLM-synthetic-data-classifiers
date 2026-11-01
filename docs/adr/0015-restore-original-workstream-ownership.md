# ADR 0015: Restore original workstream ownership

Status: accepted by project lead Kausthuba Belagod on 1 November 2026.

## Decision

The repository and GitHub tracker use the original accountable workstream ownership:

- Data generation and generation-report material: Harshita (`@harshitadjainbdes24-byte`).
- Data preparation, training and project lead: Kausthuba (`@kausthubabelagodbdes24-oss`).
- Evaluation, analysis and report assembly: Hementh (`@hem4456`).

Kausthuba implemented and integrated some generation and evaluation tasks on the relevant owner's behalf during the compressed execution window. Repository handoffs must distinguish this actual implementation provenance from accountable ownership. They must not describe Kausthuba as the sole owner or sole operator of all three workstreams.

Accepted immutable experiment records remain unchanged when they record who actually ran a model session, made a lead decision or wrote an artifact. Those records are historical provenance, not current ownership declarations. Current governance documents, report attribution, CODEOWNERS and issue assignees follow the mapping above.

## Consequences

- Generation issues are assigned to Harshita, training/lead issues to Kausthuba, and evaluation issues to Hementh.
- Cross-workstream project specification remains assigned to Kausthuba as project lead.
- Reproducibility/report tooling is assigned to Hementh when its primary deliverable is evaluation or report presentation of accepted results.
- An AI or human acting through Kausthuba's account may continue an authorized task, but the issue and PR must name both the accountable owner and the actual implementing account.
