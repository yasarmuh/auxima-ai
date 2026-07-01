"""Run the sidecar with its assist service pointed at a NVIDIA model (DEV / SYNTHETIC ONLY).

⚠️  This makes the LIVE sidecar draft via NVIDIA's hosted API (US cloud). Use ONLY against synthetic
conversations in a dev bench — never a site holding real customer data (ADR-OD-NVIDIA-dev-llm). It does
NOT touch bootstrap.py (the prod Ollama-first chain), so the wiring guard stays intact; it overrides the
running app's assist singleton at startup instead, an explicit dev act gated on AUXIMA_DEV_LLM_ENABLED=1.

    source .env.nvidia                         # AUXIMA_DEV_LLM_ENABLED=1, NVIDIA_API_KEY, NVIDIA_DEV_MODEL,
                                               # AUXIMA_SIDECAR_SHARED_SECRET (== site_config auxima_sidecar_secret)
    ./.venv/Scripts/python.exe scripts/dev_serve_nvidia.py

Then the Frappe "Draft with AI" button (or the whitelisted draft_reply_for_conversation) returns a real
NVIDIA-drafted reply for a synthetic conversation.
"""
from __future__ import annotations

import os

import uvicorn

from auxima_ai.assist.nvidia_dev import NvidiaDevLLMCaller
from auxima_ai.assist.router import get_assist_service
from auxima_ai.assist.service import AssistService
from auxima_ai.intake.llm import LLMResponse
from auxima_ai.main import app

DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"


class _ForceModelCaller:
	"""Wrap the NVIDIA dev caller so every call uses NVIDIA_DEV_MODEL, ignoring the model_id the
	service passes (the Frappe bridge sends no model_id, so it would otherwise default to an
	`ollama/...` alias NVIDIA doesn't recognise)."""

	def __init__(self, model: str) -> None:
		self._inner = NvidiaDevLLMCaller()
		self._model = model

	def call(self, *, model_id: str, prompt: str) -> LLMResponse:
		return self._inner.call(model_id=self._model, prompt=prompt)


def main() -> None:
	model = os.environ.get("NVIDIA_DEV_MODEL", DEFAULT_MODEL)
	# Constructs NvidiaDevLLMCaller -> raises unless AUXIMA_DEV_LLM_ENABLED=1 + key present.
	nvidia_svc = AssistService(llm=_ForceModelCaller(model))
	# Override at the FastAPI dependency layer, NOT via set_assist_service: the app's startup
	# lifespan runs bootstrap_app() which itself calls set_assist_service(...) and would clobber a
	# pre-set singleton. dependency_overrides intercepts get_assist_service before it is consulted,
	# so it wins regardless of startup order.
	app.dependency_overrides[get_assist_service] = lambda: nvidia_svc
	print("=" * 78)
	print(f"  DEV SIDECAR — assist routed to NVIDIA model: {model}")
	print("  ⚠️  SYNTHETIC DATA ONLY. This is NOT the production Ollama-first chain.")
	print("=" * 78)
	uvicorn.run(app, host="0.0.0.0", port=8088)


if __name__ == "__main__":
	main()
