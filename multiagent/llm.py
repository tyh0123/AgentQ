"""
Pluggable judgment backend.

If Claude is reachable (the `anthropic` SDK is installed AND a credential is
present) the critics delegate their *judgment* to Claude; otherwise they fall
back to the same physics rules, so the demo runs fully offline. Either way the
concrete *fix* is computed deterministically by the critic — the LLM decides
"is this ok and why", the code decides "by how much".

This mirrors AgentQ's two-layer thesis: deterministic computation vs.
LLM-mediated judgment.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import config

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "ok":        {"type": "boolean"},
        "issues":    {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["ok", "issues", "rationale"],
    "additionalProperties": False,
}


def available() -> bool:
    """True only if we can actually reach Claude right now."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except Exception:
        return False
    return True


def judge(system: str, context: str) -> Dict[str, object]:
    """Ask Claude for a structured verdict. Only called when available()."""
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=2000,
        thinking={"type": "adaptive"},
        output_config={"effort": config.LLM_EFFORT,
                       "format": {"type": "json_schema", "schema": _VERDICT_SCHEMA}},
        system=system,
        messages=[{"role": "user", "content": context}],
    )
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)


def backend_name() -> str:
    return f"Claude ({config.LLM_MODEL})" if available() else "deterministic rules (offline)"
