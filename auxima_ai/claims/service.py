# Copyright (c) 2026, Auxilium Tech and contributors
"""ClaimsCrewService — the P3-01 claims state machine, orchestrated as a LangGraph graph.

Graph (CLAUDE.md §2: LangGraph for stateful flows):

    validate_fnol ──rejected──▶ END
          │ok
          ▼
       triage          (LLM, ADVISORY, local_only=True ALWAYS — narrative may carry health
          │             data; degrade-safe: deterministic heuristic when the LLM chain is
          ▼             down or returns an invalid shape)
    reserve_suggest    (pure Decimal engine — reproducible financial control, no LLM)
          │
          ▼
      route_line       (loss_type → P3-04 sub-crew label)
          │
          ▼
   subcrew_actions     (P3-04: the routed line's sub-crew — deterministic advisory next-actions,
          │             no LLM: loss-adjuster dispatch / RI early-warning / subrogation / legal
          ▼             referral / specialist survey / connector-pending notes)
         END

The LLM step is invoked through an injected callable with the SAME signature as
``AssistService._invoke`` — in production bootstrap passes the bound method, so the tenant
tier gate, rate limit, cost ceiling, redaction and AI-Run-Log emission are all reused, not
re-implemented. Advisory-only: this service never writes a Frappe record (REST isolation,
Rule 2; the broker accepts in the Desk and the auxima Claim controller does the bookkeeping).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, StateGraph

from auxima_ai.claims.prompts import build_triage_prompt, validate_triage_response
from auxima_ai.claims.reserve import suggest_reserve
from auxima_ai.claims.schema import (
	ClaimsProcessOutcome,
	FNOLRequest,
	ReserveSuggestionOut,
	SubCrewRecommendation,
	TriageAssessment,
)
from auxima_ai.claims.subcrew import run_subcrew

logger = logging.getLogger(__name__)

DEFAULT_TRIAGE_MODEL = "qwen2.5:32b"  # Tier A general (CLAUDE.md §2); bootstrap may override

# Heuristic severity thresholds — degrade-path COMMERCIAL defaults, not regulator facts.
_HEURISTIC_HIGH = Decimal("500000")
_HEURISTIC_MEDIUM = Decimal("50000")

# loss_type → P3-04 sub-crew. marine/engineering ride the specialty desk until P3-04 splits them.
_SUBCREW_BY_LINE = {
	"motor": "motor",
	"property": "property",
	"medical": "medical",
	"liability": "liability",
	"marine": "specialty",
	"engineering": "specialty",
	"other": "general",
}


class _Invoke(Protocol):
	def __call__(
		self, *, tenant_id: str, model_id: str, prompt: str, local_only: bool = False, **kw: Any
	) -> Any: ...


class ClaimsState(TypedDict, total=False):
	"""The LangGraph state threaded through the nodes (plain dict — no checkpointing in v1)."""

	request: FNOLRequest
	audit_trail: list[str]
	rejected_reason: str
	triage: TriageAssessment
	reserve: ReserveSuggestionOut
	subcrew: str
	subcrew_actions: list[SubCrewRecommendation]
	degraded: bool


@dataclass
class ClaimsCrewService:
	invoke: _Invoke
	model_id: str = DEFAULT_TRIAGE_MODEL
	_graph: Any = field(init=False, default=None)

	def __post_init__(self) -> None:
		self._graph = self._build_graph()

	# --- nodes ----------------------------------------------------------------------------

	def _validate_fnol(self, state: ClaimsState) -> ClaimsState:
		req = state["request"]
		state["audit_trail"].append("validate_fnol")
		if not req.description.strip():
			state["rejected_reason"] = "FNOL narrative is empty"
		elif req.reported_date < req.incident_date:
			state["rejected_reason"] = "reported_date precedes the incident date"
		return state

	def _triage(self, state: ClaimsState) -> ClaimsState:
		req = state["request"]
		state["audit_trail"].append("triage")
		prompt = build_triage_prompt(
			loss_type=req.loss_type, incident_date=req.incident_date,
			reported_date=req.reported_date, description=req.description,
			estimated_amount=req.estimated_amount, currency=req.currency,
		)
		try:
			# local_only ALWAYS: the narrative is unstructured free text that can carry health
			# data (medical claims are special-category) — the regex redactor has no NER, so
			# this prompt never egresses to a cloud step regardless of tenant tier/approval.
			response = self.invoke(
				tenant_id=req.tenant_id, model_id=self.model_id, prompt=prompt, local_only=True,
			)
			state["triage"] = validate_triage_response(response.payload)
		except Exception as e:  # noqa: BLE001 — ANY triage failure degrades, never crashes FNOL
			logger.warning("claims triage degraded to heuristic for %s: %s", req.claim_ref, e)
			state["triage"] = self._heuristic_triage(req)
			state["degraded"] = True
		return state

	def _heuristic_triage(self, req: FNOLRequest) -> TriageAssessment:
		"""Deterministic fallback when the LLM chain is unavailable — amount-banded severity."""
		amount = Decimal(req.estimated_amount or "0")
		severity = "high" if amount >= _HEURISTIC_HIGH else (
			"medium" if amount >= _HEURISTIC_MEDIUM else "low"
		)
		complexity = "complex" if req.loss_type in ("liability", "engineering") else "standard"
		return TriageAssessment(
			severity=severity, complexity=complexity, fraud_indicators=[], source="heuristic",
		)

	def _reserve_suggest(self, state: ClaimsState) -> ClaimsState:
		req = state["request"]
		state["audit_trail"].append("reserve_suggest")
		suggestion = suggest_reserve(req.estimated_amount, req.loss_type)
		state["reserve"] = ReserveSuggestionOut(
			suggested_reserve=str(suggestion.suggested_reserve), basis=suggestion.basis,
		)
		return state

	def _route_line(self, state: ClaimsState) -> ClaimsState:
		state["audit_trail"].append("route_line")
		state["subcrew"] = _SUBCREW_BY_LINE[state["request"].loss_type]
		return state

	def _subcrew_actions(self, state: ClaimsState) -> ClaimsState:
		"""Run the routed line's sub-crew (P3-04) — deterministic advisory next-actions, no LLM."""
		state["audit_trail"].append("subcrew_actions")
		state["subcrew_actions"] = run_subcrew(
			state["subcrew"], state["request"], state["triage"], state.get("reserve"),
		)
		return state

	# --- graph ----------------------------------------------------------------------------

	def _build_graph(self) -> Any:
		g = StateGraph(ClaimsState)
		g.add_node("validate_fnol", self._validate_fnol)
		g.add_node("triage", self._triage)
		g.add_node("reserve_suggest", self._reserve_suggest)
		g.add_node("route_line", self._route_line)
		g.add_node("subcrew_actions", self._subcrew_actions)
		g.set_entry_point("validate_fnol")
		g.add_conditional_edges(
			"validate_fnol",
			lambda s: "rejected" if s.get("rejected_reason") else "ok",
			{"rejected": END, "ok": "triage"},
		)
		g.add_edge("triage", "reserve_suggest")
		g.add_edge("reserve_suggest", "route_line")
		g.add_edge("route_line", "subcrew_actions")
		g.add_edge("subcrew_actions", END)
		return g.compile()

	# --- public ----------------------------------------------------------------------------

	def process(self, request: FNOLRequest) -> ClaimsProcessOutcome:
		state: ClaimsState = {"request": request, "audit_trail": [], "degraded": False}
		final: ClaimsState = self._graph.invoke(state)
		if final.get("rejected_reason"):
			return ClaimsProcessOutcome(
				status="rejected", claim_ref=request.claim_ref,
				audit_trail=final["audit_trail"], reason=final["rejected_reason"],
			)
		return ClaimsProcessOutcome(
			status="ok", claim_ref=request.claim_ref,
			audit_trail=final["audit_trail"],
			triage=final.get("triage"),
			reserve=final.get("reserve"),
			subcrew=final.get("subcrew"),
			subcrew_actions=final.get("subcrew_actions", []),
			health_data=request.loss_type == "medical",
			degraded=bool(final.get("degraded")),
		)
