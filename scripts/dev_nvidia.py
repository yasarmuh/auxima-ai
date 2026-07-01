"""Dev harness: run an assist draft against a NVIDIA build.nvidia.com model (SYNTHETIC data only).

⚠️  DEV / PROTOTYPING ONLY. This sends the prompt to NVIDIA's hosted API (US cloud). Never pass real
customer data — synthetic only (ADR-OD-NVIDIA-dev-llm). Requires the dev opt-in env:

    export AUXIMA_DEV_LLM_ENABLED=1
    export NVIDIA_API_KEY=nvapi-...
    export NVIDIA_DEV_MODEL=meta/llama-3.1-8b-instruct   # optional; this is the default

    ./.venv/Scripts/python.exe scripts/dev_nvidia.py --message "Is my motor policy still active?"
    ./.venv/Scripts/python.exe scripts/dev_nvidia.py --kind email --purpose "follow up on renewal"

It builds a legacy-mode AssistService wired to the NVIDIA dev caller and prints the typed outcome, so
you can eyeball how a frontier model drafts vs. the local Ollama default before committing to a prompt.
"""
from __future__ import annotations

import argparse
import os
import sys

from auxima_ai.assist.nvidia_dev import NvidiaDevLLMCaller
from auxima_ai.assist.schema import DraftEmailRequest, DraftReplyRequest
from auxima_ai.assist.service import AssistService

DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"


def main(argv: list[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description="Draft against a NVIDIA model (dev/synthetic only).")
	parser.add_argument("--kind", choices=["reply", "email"], default="reply")
	parser.add_argument("--message", default="Hi, is my motor policy still active?",
	                    help="[reply] the synthetic inbound customer message")
	parser.add_argument("--purpose", default="follow up on the client's renewal",
	                    help="[email] what the email should achieve")
	parser.add_argument("--language", choices=["en", "ar"], default="en")
	parser.add_argument("--model", default=os.environ.get("NVIDIA_DEV_MODEL", DEFAULT_MODEL))
	args = parser.parse_args(argv)

	# NvidiaDevLLMCaller enforces AUXIMA_DEV_LLM_ENABLED=1 + a key; let its error speak plainly.
	try:
		caller = NvidiaDevLLMCaller()
	except Exception as e:  # noqa: BLE001 - surface the guard message to the dev, don't trace-dump
		print(f"cannot start NVIDIA dev caller: {e}", file=sys.stderr)
		return 2

	svc = AssistService(llm=caller)
	print(f"# model={args.model}  kind={args.kind}  (SYNTHETIC data only)\n")
	if args.kind == "reply":
		out = svc.draft_reply(DraftReplyRequest(
			tenant_id="dev", inbound_message=args.message, language=args.language, model_id=args.model,
		))
	else:
		out = svc.draft_email(DraftEmailRequest(
			tenant_id="dev", purpose=args.purpose, language=args.language, model_id=args.model,
		))
	print(out)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
