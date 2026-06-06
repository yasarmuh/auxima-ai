"""Tripwire: the intake path must stay self-hosted-only until it redacts prompts.

Unlike the assist path (`AssistService._invoke`), the intake/quote pipeline sends
the **raw outbound prompt** to its LLM caller and only redacts the *response*
fields afterward (`intake/service.py` builds the prompt from raw `lead_text`;
`redact_json` is applied to the validated output, not the input). That is safe
**today** solely because `build_intake_service` wires an Ollama (self-hosted)
caller, so the full prompt never leaves the Kingdom.

The day someone wires a cloud / fallback caller into intake (mirroring the assist
chain), the raw lead/SoV/quote text — names, addresses, free-text narrative —
would egress to cloud with NO pre-egress redaction and no `local_only` pin. These
tests fail at that moment, forcing the change author to move redaction to the
LLM-caller boundary (or add the assist-style gate) BEFORE any cloud egress.

See `Docs/Planning/ai-llm-compliance-audit.md` §4 R8 (intake redaction parity).
"""
from __future__ import annotations

from auxima_ai.bootstrap import build_intake_service
from auxima_ai.config import Settings
from auxima_ai.intake.ollama import OllamaLLMCaller


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_intake_wires_self_hosted_only_caller():
    """If this fails because a cloud caller was wired, STOP: intake does not
    redact the outbound prompt. Move redaction to the caller boundary first."""
    svc = build_intake_service(_settings())
    assert isinstance(svc.llm, OllamaLLMCaller), (
        "intake LLM caller is no longer self-hosted-only; intake sends the raw "
        "prompt un-redacted (see ai-llm-compliance-audit.md §4 R8) — add "
        "pre-egress redaction at the LLM-caller boundary before wiring cloud."
    )
