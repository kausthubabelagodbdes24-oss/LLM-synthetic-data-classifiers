## Appendix A. Prompt Templates and Generation Examples

*Generation workstream owner / section author: Harshita (@harshitadjainbdes24-byte); implementation and integration performed by Kausthuba Belagod (@kausthubabelagodbdes24-oss) on her behalf.*

### Part 1: Actual Prompt Templates

The exact prompt generation functions from `ws1_generation/generate.py` are documented below. All prompts enforce valid JSON replies formatted as `{"examples": ["...", "..."]}`.

#### 1. Zero-Shot Prompt Template (`prompt_zero_shot`)

```python
NUDGES = [
    "Pick specific, uncommon subjects.",
    "Mix short and longer examples.",
    "Use a different sentence structure each time.",
    "Avoid repeating the same opening words."
]

def prompt_zero_shot(task, label, n, i):
    t = CFG["tasks"][task]
    if task == "trec":
        boundaries = {
            "abbreviation": "Ask explicitly for an acronym's expansion or a shortened form. "
                            "Do not ask for a general definition or explanation.",
            "entity": "The answer must name a non-human thing, object, product, animal or substance. "
                      "Do not ask for a definition, acronym expansion or chemical symbol.",
            "description": "Ask for an explanation, definition, reason or how something works. "
                           "Do not ask what an acronym stands for.",
            "human": "The answer must name a person, team, group or organisation. "
                     "Mentioning a person does not suffice when the requested answer is their work or creation.",
            "location": "The answer must be a geographic place such as a city, country, mountain or river. "
                        "Do not ask for a person, definition, count, age or acronym expansion.",
            "numeric": "The answer must be a number, count, date, age, duration or measurement. "
                       "Do not ask for a product name or a place name.",
        }[label] + " Ask open questions, not yes/no questions."
    else:
        boundaries = {
            "sadness": "Clearly express feeling sad, lonely, hurt or low in a concrete situation.",
            "joy": "Clearly express happiness, delight, contentment or excitement. "
                   "The dominant feeling should be happiness rather than affection toward someone.",
            "love": "Clearly express affection, tenderness or care toward someone or something. "
                    "A good event or feeling happy alone is not sufficient.",
            "anger": "Clearly express anger, irritation, frustration or annoyance about a situation. "
                     "Do not rely on a situation alone without expressing the writer's reaction.",
            "fear": "Clearly express feeling afraid, anxious, nervous or worried about a threat or uncertainty.",
            "surprise": "Clearly express being amazed, shocked or caught off guard by an unexpected event. "
                        "Mentioning a surprise party alone does not establish the writer's reaction.",
        }[label] + " Make this the clear dominant emotion; use varied situations and natural wording."
    return (f"You are building a dataset of {t['context']}.\n"
            f"Write {n} different examples of {t['style']}.\n"
            f'Every example must have the label "{label}", meaning: {t["labels"][label]}.\n'
            f"Requirements for this label: {boundaries}\n"
            f"At most {t['max_words']} words each. {NUDGES[i % len(NUDGES)]}\n"
            "Check each candidate against the requested label before returning it. "
            "Return only examples that clearly fit; do not include answers or explanations.\n"
            'Reply only with JSON: {"examples": ["...", "..."]}')
```

#### 2. Few-Shot Prompt Template (`prompt_few_shot`)

The few-shot prompt inherits the zero-shot task framing, label definition, and boundaries, inserting three real demonstrations from the isolated demonstration pool split:

```python
def prompt_few_shot(task, label, n, examples, i):
    shown = "\n".join(f"- {json.dumps(e)}" for e in examples)
    return (prompt_zero_shot(task, label, n, i).replace(
        "Reply only", f"Here are real examples with this label:\n{shown}\n"
        "Use these for style, while following the requested label's requirements above.\n"
        "Write NEW examples in the same style. Do not copy or paraphrase these.\nReply only"))
```

#### 3. AugGPT Paraphrasing Templates

The AugGPT pipeline utilizes three prompt variants depending on task and source length:

**Base AugGPT Template (`prompt_auggpt`):**
```python
def prompt_auggpt(text, k):
    return (f"Please rephrase the following sentence {k} different ways, keeping its meaning:\n"
            f"{json.dumps(text)}\n"
            'Reply only with JSON: {"examples": ["...", "..."]}')
```

**Length Guard Variant (`prompt_auggpt_length_guard`):**
Used when source length exceeds `max_words` (30 words in Emotion):
```python
def prompt_auggpt_length_guard(text, k, max_words):
    return prompt_auggpt(text, k).replace(
        "Reply only",
        f"Each rephrasing must be at most {max_words} words, so shorten it: keep the main "
        "meaning and the writer's feeling, and leave out minor details.\nReply only")
```

**Corrected TREC Answer-Type Template (`prompt_auggpt_trec_answer_type`, ADR 0013 v2):**
Replaces the base prompt for all TREC seed-13 AugGPT sources. Calls send `{"think": false, "num_predict": 1024}`:
```python
def prompt_auggpt_trec_answer_type(text, k, label, definition, max_words):
    label_rule = {
        "abbreviation": "If it asks what an abbreviation or acronym stands for, every rephrasing must still "
                        "ask what it stands for (its expansion), not for a definition, meaning or explanation "
                        "of the thing. If it asks for the abbreviation of something, every rephrasing must "
                        "still ask for that abbreviation.",
        "description": "Every rephrasing must remain a question asking for an explanation, definition, "
                       "reason or how something is done, never one whose answer is a name, person, place, "
                       "number or event.",
        "human": "If it asks who a named person, group or organisation is (for example \"Who is X ?\"), "
                 "every rephrasing must still ask who that same person, group or organisation is; it keeps "
                 "the human label, so do not turn it into a request for a definition, explanation or "
                 "biography. Otherwise every rephrasing must still expect a person, group of people, "
                 "organisation or title as its answer.",
    }.get(label, f"Every rephrasing must still expect this kind of answer: {definition}.")
    return (f"Please rephrase the following question {k} different ways:\n"
            f"{json.dumps(text)}\n"
            f'Its answer-type label is "{label}", meaning: {definition}.\n'
            "Every rephrasing must keep the same expected answer type, so it keeps this label. "
            f"{label_rule}\n"
            "Keep the same facts and relationships between them, the same entities and participants "
            "(who did what to whom), any negation, and the same intent.\n"
            "Each rephrasing must be a question or request for information. Do not answer the question, "
            "and do not add facts, names, dates or details that are not in it.\n"
            f"Each rephrasing must be at most {max_words} words; if the question is longer, word it more "
            "concisely without dropping any of its facts.\n"
            'Reply only with JSON: {"examples": ["...", "..."]}')
```

---

### Part 2: Preserved Generation Examples from Accepted Final Evidence

Below are two good and two problematic examples per task selected directly from the accepted final artifacts reviewed in Issue #11 and frozen under Issue #12 ([`docs/freeze/scope30-v1-seed13.freeze.json`](../freeze/scope30-v1-seed13.freeze.json)). Every item is identified by its artifact hash, generation fingerprint, row ID, source ID (where applicable), label, and model-assisted review assessment.

#### Objective Task: TREC Question Classification

##### 1. Good Example (Abbreviation Expansion Preserved)
- **Item ID:** `trec.train.04455.seed13.aug6`
- **Artifact:** `data/synthetic/trec/auggpt_seed13.jsonl` (ADR 0013 v2 promoted; SHA-256 `cf526bb948ef0e5227d54434ec897429bb0afe3f7a803579d34b374c5755ef07`)
- **Generation Fingerprint:** `c4d5308aa37a6ca8c8a820fc14e042376101bb2cddf352f25c5b933c2e3a4f56`
- **Source ID:** `trec.train.04455` (from `trec/lowres_seed13.jsonl`, SHA-256 `901c0596…`)
- **Source Text:** `What does `` B.Y.O.B. '' mean ?`
- **Intended Label:** `abbreviation`
- **Generated Text:** `"What is B.Y.O.B. expanded to?"`
- **Review Provenance:** Round 2 random audit (`docs/reviews/issue11/round2_judgments.json`, order 5).
- **Rubric Judgment:** `label_fit: fits`, `naturalness: natural`, `style_fit: natural`, `clear_error: false`, `ambiguous: false`. Successfully complies with ADR 0013 by requesting the acronym's expansion rather than drifting into a general definition request.

##### 2. Good Example (Geographic Location Question Preserved)
- **Item ID:** `trec.train.03037.seed13.aug3`
- **Artifact:** `data/synthetic/trec/auggpt_seed13.jsonl` (ADR 0013 v2 promoted; SHA-256 `cf526bb9…`)
- **Generation Fingerprint:** `c4d5308aa37a6ca8c8a820fc14e042376101bb2cddf352f25c5b933c2e3a4f56`
- **Source ID:** `trec.train.03037` (from `trec/lowres_seed13.jsonl`)
- **Source Text:** `What is the largest county in size in Massachusetts ?`
- **Intended Label:** `location`
- **Generated Text:** `"Which county in Massachusetts is the largest by size?"`
- **Review Provenance:** Round 2 random audit (`docs/reviews/issue11/round2_judgments.json`, order 7).
- **Rubric Judgment:** `label_fit: fits`, `naturalness: natural`, `style_fit: natural`, `clear_error: false`, `ambiguous: false`. Rephrases concisely while cleanly retaining the geographic place answer type.

##### 3. Problematic Example (Named Entity Corruption in ADR 0013 v2)
- **Item ID:** `trec.train.04163.seed13.aug4`
- **Artifact:** `data/synthetic/trec/auggpt_seed13.jsonl` (ADR 0013 v2 promoted; SHA-256 `cf526bb9…`)
- **Generation Fingerprint:** `c4d5308aa37a6ca8c8a820fc14e042376101bb2cddf352f25c5b933c2e3a4f56`
- **Source ID:** `trec.train.04163` (from `trec/lowres_seed13.jsonl`)
- **Source Text:** `What is the history of Hot Wheels cars , and how are they produced ?`
- **Intended Label:** `description`
- **Generated Text:** `"Explain the historical background and production techniques of Hot and Wheels cars?"`
- **Review Provenance:** Round 2 random audit (`docs/reviews/issue11/round2_judgments.json`, order 4).
- **Rubric Judgment:** `label_fit: fits`, `naturalness: unnatural`, `clear_error: true`, `ambiguous: false`, `source_meaning: changed`. The named entity "Hot Wheels" is corrupted to "Hot and Wheels". Classified as a clear error in Round 2 lead review; retained under Option A.

##### 4. Problematic Example (Expected Answer-Type Drift)
- **Item ID:** `trec.train.00845.seed13.aug3`
- **Artifact:** `data/synthetic/trec/auggpt_seed13.jsonl` (ADR 0013 v2 promoted; SHA-256 `cf526bb9…`)
- **Generation Fingerprint:** `c4d5308aa37a6ca8c8a820fc14e042376101bb2cddf352f25c5b933c2e3a4f56`
- **Source ID:** `trec.train.00845` (from `trec/lowres_seed13.jsonl`)
- **Source Text:** `What is D.B. Cooper known for ?`
- **Intended Label:** `description`
- **Generated Text:** `"What specific incident does D.B. Cooper's identity refer to?"`
- **Review Provenance:** Round 2 targeted re-check (`docs/reviews/issue11/round2_recheck_judgments.json`, order 2).
- **Rubric Judgment:** `label_fit: wrong`, `clear_error: true`, `ambiguous: false`, `source_meaning: changed`. The question changes the expected answer from an explanation/description of what Cooper is known for to an `entity` answer type (a specific incident/event). Confirmed as an unresolved answer-type error outside the random denominator; retained under Option A.

---

#### Subjective Task: Emotion Classification

##### 1. Good Example (Natural Workplace Frustration)
- **Item ID:** `emotion.few_shot.seed13.anger.0016`
- **Artifact:** `data/synthetic/emotion/few_shot_seed13.scope30-v1.selected.jsonl` (SHA-256 `78948f67bef46675ec01ea68ec02c52527875814b7aca3b94dc381b7cf3e08ad`)
- **Generation Fingerprint:** `cec310b818092931f94e1fad8c19745e4ced1d59e21392cc6abc6a73901a73d4`
- **Demonstration IDs:** `['emotion.train.03691', 'emotion.train.08204', 'emotion.train.10029']`
- **Intended Label:** `anger`
- **Generated Text:** `"i feel angry when my boss assigns tasks i can't finish in time"`
- **Review Provenance:** Round 1 random review (`docs/reviews/issue11/round1_judgments.json`, order 50).
- **Rubric Judgment:** `label_fit: fits`, `naturalness: natural`, `style_fit: matches`, `copied_demonstration: false`, `clear_error: false`, `ambiguous: false`. Expresses direct situational frustration; no demonstration copying.

##### 2. Good Example (Expressive Accomplishment)
- **Item ID:** `emotion.zero_shot.seed13.joy.0004`
- **Artifact:** `data/synthetic/emotion/zero_shot_seed13.scope30-v1.selected.jsonl` (SHA-256 `a7458c204cc4317c0ea0d5b630174a008819107afa807546763871c8ae783285`)
- **Generation Fingerprint:** `4d2ab0f440733aa9d98b1107a776f528534d68318770d4c13652758ec17388f7`
- **Demonstration IDs:** None (zero-shot)
- **Intended Label:** `joy`
- **Generated Text:** `"i feel glad for the small achievements today that made me smile and feel accomplished"`
- **Review Provenance:** Round 1 random review (`docs/reviews/issue11/round1_judgments.json`, order 57).
- **Rubric Judgment:** `label_fit: fits`, `naturalness: natural`, `style_fit: matches`, `clear_error: false`, `ambiguous: false`. Natural emotional expression; fits canonical joy definition.

##### 3. Problematic Example (AugGPT Agent Reversal)
- **Item ID:** `emotion.train.07790.seed13.aug1`
- **Artifact:** `data/synthetic/emotion/auggpt_seed13.jsonl` (SHA-256 `59b2a6fb414c51c43ffb922b85735a81d45eafd513e95feec962eb5a8f46f67d`)
- **Generation Fingerprint:** `629fb0fd6933303afa986594b883d5c39aa358d02d505461b58a871bd46f5982` (length guard)
- **Source ID:** `emotion.train.07790` (from `emotion/lowres_seed13.jsonl`, SHA-256 `7bec4826…`)
- **Source Text:** `i usually doubt my self at this point as i feel i should be that amazing housewife who motors all day and has a list of things they can tell theyre husband they did all day while they were at work and i was at home`
- **Intended Label:** `surprise`
- **Generated Text:** `"I doubt myself now, feeling I should be the amazing housewife who works all day and has a list of what my husband did while I was home."`
- **Review Provenance:** Supplementary targeted check re-adjudication (`docs/reviews/issue11/targeted_long_sources_adjudication_2026-11-01.json`, item 1).
- **Rubric Judgment:** `clear_error: true`, `ambiguous: false`, `source_meaning: changed`. Inverts the agent: the source describes a wife making a list of what *she* did to tell her husband, whereas the paraphrase states a list of what *the husband* did while she was home. Retained under Option A.

##### 4. Problematic Example (AugGPT Subject Shift)
- **Item ID:** `emotion.train.15223.seed13.aug1`
- **Artifact:** `data/synthetic/emotion/auggpt_seed13.jsonl` (SHA-256 `59b2a6fb…`)
- **Generation Fingerprint:** `629fb0fd6933303afa986594b883d5c39aa358d02d505461b58a871bd46f5982` (length guard)
- **Source ID:** `emotion.train.15223` (from `emotion/lowres_seed13.jsonl`)
- **Source Text:** `i don t believe in my weakness he is strong i don t believe i am more than a conqueror and i feel like i m a real fake and it s not fine`
- **Intended Label:** `sadness`
- **Generated Text:** `"I don't believe I'm weak—I'm strong, but I don't think I'm more than a conqueror. I feel like a real fake, and this isn't fine."`
- **Review Provenance:** Supplementary targeted check re-adjudication (`docs/reviews/issue11/targeted_long_sources_adjudication_2026-11-01.json`, item 7).
- **Rubric Judgment:** `clear_error: true`, `ambiguous: false`, `source_meaning: changed`. The source contrasts the author's weakness against a third party's strength (*"he is strong"*), whereas the paraphrase transforms this into a self-affirmation (*"I'm strong"*), reversing propositional meaning. Retained under Option A.

---

### Part 3: Historical Pilot Examples (ADR 0011 / PILOT_REPORT.md)

*Context note:* These eight examples were selected during the preliminary `qwen3:4b` fallback pilot review conducted by Kausthuba Belagod on 31 October 2026 ([ADR 0011](../adr/0011-accept-qwen-pilot-and-uniform-100-target.md); [`docs/PILOT_REPORT.md`](../PILOT_REPORT.md)). They represent historical screening evidence that supported adopting `qwen3:4b` over the failed Llama model, and are distinct from the final frozen `scope30-v1` artifacts above.

| Task | Setting | Row ID | Intended Label | Human Judgment | Generated Text | Historical Inspection Note |
|---|---|---|---|---|---|---|
| **trec** | `zero_shot` | `trec.zero_shot.seed13.description.0005` | `description` | **Good** | `What causes the tides?` | Clear natural description request; fits open search style. |
| **trec** | `few_shot` | `trec.few_shot.seed13.entity.0003` | `entity` | **Good** | `What is the most expensive car ever sold?` | Well-formed entity question; non-human object answer. |
| **trec** | `zero_shot` | `trec.zero_shot.seed13.entity.0005` | `entity` | **Bad** | `What is the next event?` | Underspecified premise; missing required temporal or topical context. |
| **trec** | `few_shot` | `trec.few_shot.seed13.location.0007` | `location` | **Bad** | `What river runs through London?` | Generic trivia question duplicating real-sample question frames. |
| **emotion** | `zero_shot` | `emotion.zero_shot.seed13.sadness.0002` | `sadness` | **Good** | `i feel hurt because my friend broke our promise last week` | Expresses concrete personal sadness and hurt. |
| **emotion** | `few_shot` | `emotion.few_shot.seed13.love.0009` | `love` | **Good** | `i feel my heart swell with love for the way you listen to me.` | Expresses direct interpersonal affection and tenderness. |
| **emotion** | `zero_shot` | `emotion.zero_shot.seed13.joy.0002` | `joy` | **Bad** | `i feel excited about the summer as a vacation` | Unnatural phrasing; awkward syntactic construction (*"as a vacation"*). |
| **emotion** | `few_shot` | `emotion.few_shot.seed13.fear.0005` | `fear` | **Bad** | `i feel cold sweat imagining the worst test result` | Awkward tense construction (*"feel cold sweat imagining"*). |
