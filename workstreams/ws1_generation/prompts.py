"""Prompt templates (bump `prompt_version` in configs/generation.yaml when you edit them).

Zero-shot and few-shot follow Li et al. (2023): a context prompt that sets the
scene, then a generation prompt naming the label, its meaning, the style and a
word limit, plus a rotating "be diverse" nudge. Few-shot adds k real examples
of the label and tells the model not to copy them (Brown et al. 2020 for the
in-context idea). AugGPT follows Dai et al. (2023): rephrase one real sentence
into several paraphrases.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Ollama structured-output schema: {"examples": ["...", ...]}
EXAMPLES_SCHEMA = {
    "type": "object",
    "properties": {"examples": {"type": "array", "items": {"type": "string"}}},
    "required": ["examples"],
}

DIVERSITY_NUDGES = [
    "Pick specific, uncommon subjects rather than the most obvious ones.",
    "Mix very short and longer examples.",
    "Use a different sentence structure in every example.",
    "Vary the vocabulary; avoid repeating the same opening words.",
    "Imagine a different writer for each example.",
    "Cover everyday situations as well as unusual ones.",
]


def _system(task_cfg: dict[str, Any]) -> str:
    return (
        f"You are helping to build {task_cfg['task_context'].strip()}. "
        "You write realistic, varied examples that read like real data, never like a template. "
        'Reply only with JSON of the form {"examples": ["...", "..."]}.'
    )


def _rules(task_cfg: dict[str, Any], call_index: int) -> str:
    nudge = DIVERSITY_NUDGES[call_index % len(DIVERSITY_NUDGES)]
    return (
        f"- at most {task_cfg['max_words']} words each\n"
        "- every example must be different from the others\n"
        "- no numbering, no labels, no explanations\n"
        f"- {nudge}"
    )


def zero_shot(task_cfg: dict[str, Any], label: str, n: int, call_index: int) -> list[dict[str, str]]:
    desc = task_cfg["labels"][label]
    user = (
        f"Write {n} examples of {task_cfg['style']}.\n"
        f'Every example must clearly belong to the label "{label}", which means: {desc}.\n'
        f"Rules:\n{_rules(task_cfg, call_index)}\n"
        f'Return JSON: {{"examples": [{n} strings]}}'
    )
    return [{"role": "system", "content": _system(task_cfg)}, {"role": "user", "content": user}]


def few_shot(
    task_cfg: dict[str, Any], label: str, n: int, examples: list[str], call_index: int
) -> list[dict[str, str]]:
    desc = task_cfg["labels"][label]
    shown = "\n".join(f"{i + 1}. {json.dumps(t, ensure_ascii=False)}" for i, t in enumerate(examples))
    user = (
        f'Here are {len(examples)} real examples with the label "{label}" ({desc}):\n'
        f"{shown}\n\n"
        f"Write {n} NEW examples of {task_cfg['style']} with the same label.\n"
        "The examples above only show the style and the label: do not copy them, "
        "do not paraphrase them, and do not reuse their subjects.\n"
        f"Rules:\n{_rules(task_cfg, call_index)}\n"
        f'Return JSON: {{"examples": [{n} strings]}}'
    )
    return [{"role": "system", "content": _system(task_cfg)}, {"role": "user", "content": user}]


def auggpt(text: str, k: int) -> list[dict[str, str]]:
    system = "You are a helpful assistant that rephrases text and makes sentences smooth."
    user = (
        f"Please rephrase the following sentence {k} times, giving {k} different rephrasings "
        f"that keep its meaning:\n{json.dumps(text, ensure_ascii=False)}\n"
        f'Return JSON: {{"examples": [{k} strings]}}'
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def _clean(s: str) -> str:
    s = s.strip().strip('"').strip("“”").strip()
    return " ".join(s.split())


def parse_examples(raw: str) -> tuple[list[str], str]:
    """Return (examples, how) where how is 'json', 'json-embedded', 'lines' or 'failed'."""
    text = _THINK.sub("", raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    def from_obj(obj: Any) -> list[str] | None:
        if isinstance(obj, dict):
            for key in ("examples", "paraphrases", "rephrasings", "sentences"):
                if isinstance(obj.get(key), list):
                    obj = obj[key]
                    break
        if isinstance(obj, list):
            items = [_clean(x) for x in obj if isinstance(x, str)]
            return [x for x in items if x]
        return None

    try:
        got = from_obj(json.loads(text))
        if got:
            return got, "json"
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match:
        try:
            got = from_obj(json.loads(match.group(1)))
            if got:
                return got, "json-embedded"
        except json.JSONDecodeError:
            pass
    lines = [_clean(_BULLET.sub("", ln)) for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln.split()) >= 2 and not ln.endswith(":")]
    if len(lines) >= 2:
        return lines, "lines"
    return [], "failed"
