# Generation Pilot Report and Target Decision

- **Owner**: @harshitadjainbdes24-byte (Generation)
- **Workstream**: WS1 - Generation
- **Last updated**: 30 October 2026
- **Status**: **Pending owner-run Ollama evidence. Issue #6 is not accepted or merge-ready on this report alone.**

## Evidence boundary

The implementation has automated fake-model coverage for repetition isolation, malformed replies, connection failures, resume compatibility, and the pipeline smoke seam. Fake output is stored under `data/synthetic-fixtures/`, is marked `generation_mode: fake`, and must not be reported as experimental evidence.

No real Ollama smoke or pilot evidence is committed at this revision. Therefore this report does **not** claim:

- a completed 240-row real pilot;
- inspection of about 40 real model outputs;
- measured Ollama throughput;
- an approved choice of 100 or 60 examples per label;
- approval to unblock dependent experiments.

The examples and runtime calculations in the previous draft were illustrative. They were not outputs or measurements from an identified Ollama run and have been removed.

## Owner-run evidence required for Issue #6

### 1. Real Ollama smoke

Run:

```powershell
python -m ws1_generation.generate --smoke-ollama
```

Record the following from the machine that will generate the pilot:

| Field | Owner evidence |
| --- | --- |
| Date/time and operator | Pending |
| Ollama endpoint | Pending |
| Configured model tag | Pending |
| Resolved model name | Pending |
| Actual Ollama model digest/ID | Pending |
| Smoke completion result and elapsed time | Pending |

The configured tag alone is insufficient. Real-generation sidecars resolve and persist the actual Ollama digest before accepting or resuming rows.

### 2. Repetition-aware real pilot

After the smoke succeeds, run seed 13 for both tasks and both settings without `--dry-run`:

```powershell
python -m ws1_generation.generate --setting zero_shot --pilot --seed 13 --task all
python -m ws1_generation.generate --setting few_shot --pilot --seed 13 --task all
```

Expected only after successful completion: four real artifacts with 10 rows per label, 60 rows each, 240 rows total:

- `data/synthetic/trec/zero_shot_seed13.jsonl`
- `data/synthetic/trec/few_shot_seed13.jsonl`
- `data/synthetic/emotion/zero_shot_seed13.jsonl`
- `data/synthetic/emotion/few_shot_seed13.jsonl`

For each artifact, retain its `.provenance.json` sidecar and record:

| Evidence | Required value |
| --- | --- |
| Generation mode | `real` |
| Run identity | Sidecar session `run_id` |
| Model identity | Resolved name plus immutable digest |
| Prompt/options identity | Sidecar prompt and options hashes |
| Source identity | Pool/test hashes and row counts |
| Repetition | `13` |
| Actual outputs | Artifact path and accepted row IDs from the session |
| Counts | 10 accepted rows for every configured label |
| Timing | Session start, finish and actual elapsed seconds |
| Failures/rejections | Session statistics, preserved cumulatively |

Do not copy fake rows into these paths. A fake artifact cannot satisfy or resume a real target.

### 3. Manual inspection and prompt refinement

Select about 40 row IDs across both tasks and both settings from the real pilot. Record the selection rule or selected IDs so the inspection can be reproduced. Review label fit, dataset style, variation, length, copied demonstrations, malformed content, and unacceptable output.

Complete this table with actual evidence:

| Field | Owner evidence |
| --- | --- |
| Inspector and date | Pending |
| Selected row IDs or selection file | Pending |
| Findings by task and setting | Pending |
| Prompt refinement round 0 result | Pending |
| Refinement round 1, if needed | Pending |
| Refinement round 2, if needed | Pending; maximum two rounds |
| Two retained good examples per task | Pending |
| Two retained bad examples per task, with disposition | Pending |
| Approved pre-freeze reset, if any | Pending |

Illustrative text must not be presented as inspected model output. If prompt identity changes, existing rows are incompatible with resume; reset only the affected artifact before freeze, record approval and preserve the prior run evidence separately.

### 4. Measured timing and uniform target decision

The default plan remains 100 examples per label. The only approved fallback is 60 examples per label applied uniformly across both tasks and all three repetitions when actual pilot timing shows the default cannot complete before the agreed freeze. The decision is deadline-based and requires coordination with Kausthuba and the team before full generation.

A fixed 30-second automatic trigger is **not** the agreed policy.

Use the real sidecar session timings and observed call/rejection counts to estimate the remaining zero-shot, few-shot, and AugGPT work. Record:

| Decision evidence | Owner evidence |
| --- | --- |
| Pilot elapsed time and calls by artifact | Pending |
| Observed acceptance/retry rate | Pending |
| Remaining work at 100 per label | Pending |
| Remaining work at 60 per label | Pending |
| Time available to the agreed freeze | Pending |
| Recommended uniform target and rationale | Pending |
| Kausthuba/team approval and date | Pending |

Do not change shared configuration or begin dependent experiments on an unsupported recommendation.

## Dependency status

Keep issue #6 open until the owner-run evidence above is reviewed and the implementation is accepted. Keep #7, #9, and #10 blocked on issue #6 until then. Completing the later full AugGPT work is outside the acceptance scope of issue #6.
