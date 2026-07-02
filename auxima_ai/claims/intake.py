# Copyright (c) 2026, Auxilium Tech and contributors
"""FNOLIntakeService — the CHECKPOINTED multi-turn FNOL intake graph (P3-01c).

A claimant reports a loss over several messages (portal, WhatsApp bridge, phone
transcriber). Each turn runs the graph below against a per-(tenant, session) LangGraph
checkpoint thread, so state accumulates between REST calls:

    ingest ──▶ extract ──▶ assess ──▶ END
    (turn count,  (caller fields >     (missing? → bilingual next question;
     pin the       deterministic >      complete? → assemble the FNOL and hand it
     reported      LLM local_only,      to ClaimsCrewService.process — exactly once)
     date)         degrade-safe)

Field precedence is disclosable and fail-safe: structured caller ``fields`` win, the
deterministic extractor (precision) runs next, the LLM extract step (recall) only fills
what is STILL missing — pinned ``local_only=True`` unconditionally because the narrative
is free text that can carry health data (mirrors the triage step). Any LLM failure
degrades the turn, never crashes it.

The thread key embeds the tenant, so tenant B can never resume tenant A's session.
Checkpoints live in the injected checkpointer (default: in-process ``MemorySaver`` behind
an LRU cap — a restart drops open sessions, which fail LOUDLY as a fresh session, never
silently mix state; swap in a durable checkpointer via the constructor when one is
approved as a dependency).
"""
from __future__ import annotations

import logging
import operator
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from auxima_ai.claims.intake_extract import (
	extract_amount,
	extract_incident_date,
	extract_loss_type,
	is_skip_amount,
)
from auxima_ai.claims.intake_schema import FNOLTurnOutcome, FNOLTurnRequest
from auxima_ai.claims.prompts import build_fnol_extract_prompt, validate_extract_payload
from auxima_ai.claims.schema import ClaimsProcessOutcome, FNOLRequest

logger = logging.getLogger(__name__)

DEFAULT_EXTRACT_MODEL = "llama3.1:8b"  # Tier C parse (CLAUDE.md §2); bootstrap may override
DEFAULT_MAX_TURNS = 12
DEFAULT_MAX_SESSIONS = 5000

_FIELD_ORDER = ("loss_type", "incident_date", "estimated_amount")

INTAKE_QUESTIONS: dict[str, dict[str, str]] = {
	"loss_type": {
		"en": "What type of loss is this? (motor / property / medical / liability / "
		"marine / engineering / other)",
		"ar": "ما نوع الخسارة؟ (مركبات / ممتلكات / طبي / مسؤولية / بحري / هندسي / أخرى)",
	},
	"incident_date": {
		"en": "When did the incident happen? (for example: 2026-06-28)",
		"ar": "متى وقع الحادث؟ (مثال: 2026-06-28)",
	},
	"estimated_amount": {
		"en": "What is the estimated loss amount in SAR? Reply 'skip' if unknown.",
		"ar": "ما قيمة الخسارة التقديرية بالريال السعودي؟ أرسل \"غير معروف\" إذا لم تكن متأكداً.",
	},
}


class IntakeSessionExhausted(Exception):
	"""The session hit its turn cap — file via the Desk instead (router maps to 409)."""


class IntakeState(TypedDict, total=False):
	"""The checkpointed thread state. ``messages`` accumulates via the add reducer; the
	per-turn inputs (turn_*) and per-turn outputs (warnings/degraded) are overwritten each
	invoke; collected fields and ``outcome`` are set-once scalars."""

	messages: Annotated[list[str], operator.add]
	turns: int
	tenant_id: str
	session_id: str
	channel: str
	reported_date: str
	turn_today: str
	turn_fields: dict[str, str] | None
	pending: str | None
	loss_type: str | None
	incident_date: str | None
	estimated_amount: str | None
	amount_skipped: bool
	warnings: list[str]
	degraded: bool
	outcome: ClaimsProcessOutcome | None


class _ClaimsCrew:
	"""Duck-type of ClaimsCrewService — the one method the intake hands off to."""

	def process(self, request: FNOLRequest) -> ClaimsProcessOutcome: ...


@dataclass
class FNOLIntakeService:
	claims: _ClaimsCrew
	invoke: Any = None  # AssistService._invoke-shaped callable; None → deterministic only
	model_id: str = DEFAULT_EXTRACT_MODEL
	max_turns: int = DEFAULT_MAX_TURNS
	max_sessions: int = DEFAULT_MAX_SESSIONS
	checkpointer: Any = None
	_graph: Any = field(init=False, default=None)
	_sessions: OrderedDict = field(init=False, default_factory=OrderedDict)

	def __post_init__(self) -> None:
		if self.checkpointer is None:
			self.checkpointer = MemorySaver()
		self._graph = self._build_graph()

	# --- nodes ----------------------------------------------------------------------------

	def _ingest(self, state: IntakeState) -> IntakeState:
		update: IntakeState = {"turns": state.get("turns", 0) + 1}
		if not state.get("reported_date"):
			update["reported_date"] = state["turn_today"]  # first contact = reported date
		return update

	def _extract(self, state: IntakeState) -> IntakeState:
		message = state["messages"][-1]
		today = date.fromisoformat(state["turn_today"])
		update: IntakeState = {"warnings": [], "degraded": False}
		warnings: list[str] = []

		def have(f: str) -> bool:
			return bool(state.get(f) or update.get(f))

		# 1) structured caller fields win — but a future incident date is user error:
		#    warn and keep asking rather than let process() reject the whole FNOL later.
		for f, value in (state.get("turn_fields") or {}).items():
			if value is None or have(f):
				continue
			if f == "incident_date":
				# the HTTP boundary already validated ISO shape, but internal callers
				# (a future channel bridge) may not go through ProvidedFields — a bad
				# date must warn-and-re-ask, never 500 the turn
				try:
					if date.fromisoformat(value) > today:
						warnings.append(f"incident_date {value} is in the future — ignored")
						continue
				except ValueError:
					warnings.append(f"incident_date {value!r} is not an ISO date — ignored")
					continue
			update[f] = value

		# 2) deterministic extractor (precision; bare numbers only when we asked for one)
		if not have("loss_type") and (lt := extract_loss_type(message)):
			update["loss_type"] = lt
		if not have("incident_date") and (
			d := extract_incident_date(message, today=today)
		):
			update["incident_date"] = d
		if not have("estimated_amount") and not state.get("amount_skipped"):
			asked = state.get("pending") == "estimated_amount"
			if asked and is_skip_amount(message):
				update["amount_skipped"] = True
			elif amount := extract_amount(message, assume_amount=asked):
				update["estimated_amount"] = amount

		# 3) LLM extract (recall) — ONLY for what is still missing, local_only ALWAYS.
		#    Money is NEVER an LLM field: a live 0.5B model garbled "20 thousand riyals"
		#    into 200000 — magnitude-wrong money that passes shape validation. The
		#    deterministic extractor owns amounts; the reporter is re-asked otherwise.
		llm_fields = [
			f for f in _FIELD_ORDER if f != "estimated_amount" and not have(f)
		]
		if llm_fields and self.invoke is not None:
			try:
				response = self.invoke(
					tenant_id=state["tenant_id"], model_id=self.model_id,
					prompt=build_fnol_extract_prompt(
						message=message, missing=llm_fields, today=state["turn_today"],
					),
					local_only=True,
				)
				for f, value in validate_extract_payload(response.payload, today=today).items():
					if f in llm_fields:
						update[f] = value
			except Exception as e:  # noqa: BLE001 — ANY extract failure degrades, never crashes
				logger.warning(
					"fnol intake extract degraded for %s/%s: %s",
					state["tenant_id"], state["session_id"], e,
				)
				update["degraded"] = True

		update["warnings"] = warnings
		return update

	def _assess(self, state: IntakeState) -> IntakeState:
		missing = self._missing(state)
		if missing:
			return {"pending": missing[0]}
		request = FNOLRequest(
			tenant_id=state["tenant_id"],
			claim_ref=f"FNOL-{state['session_id']}",
			loss_type=state["loss_type"],
			incident_date=state["incident_date"],
			reported_date=state["reported_date"],
			description="\n".join(state["messages"]),
			estimated_amount=state.get("estimated_amount") or "0",
		)
		return {"pending": None, "outcome": self.claims.process(request)}

	# --- graph ----------------------------------------------------------------------------

	def _build_graph(self) -> Any:
		g = StateGraph(IntakeState)
		g.add_node("ingest", self._ingest)
		g.add_node("extract", self._extract)
		g.add_node("assess", self._assess)
		g.set_entry_point("ingest")
		g.add_edge("ingest", "extract")
		g.add_edge("extract", "assess")
		g.add_edge("assess", END)
		return g.compile(checkpointer=self.checkpointer)

	# --- helpers ---------------------------------------------------------------------------

	@staticmethod
	def _missing(state: IntakeState) -> list[str]:
		missing = [f for f in ("loss_type", "incident_date") if not state.get(f)]
		if not state.get("estimated_amount") and not state.get("amount_skipped"):
			missing.append("estimated_amount")
		return missing

	def _touch(self, thread_key: str) -> None:
		"""LRU-cap the in-memory sessions — evicted threads are DELETED from the
		checkpointer (bounded memory beats immortal sessions; an evicted reporter simply
		starts over)."""
		self._sessions[thread_key] = True
		self._sessions.move_to_end(thread_key)
		while len(self._sessions) > self.max_sessions:
			evicted, _ = self._sessions.popitem(last=False)
			try:
				self.checkpointer.delete_thread(evicted)
				logger.info("fnol intake session evicted (LRU): %s", evicted)
			except Exception:  # noqa: BLE001 — an eviction failure must never 500 the
				# INNOCENT user whose turn triggered it; the orphaned thread is logged
				# loudly for the operator (matters once a durable checkpointer is swapped in)
				logger.exception("fnol intake eviction FAILED — thread orphaned: %s", evicted)

	def _outcome(self, session_id: str, state: IntakeState) -> FNOLTurnOutcome:
		missing = self._missing(state)
		outcome = state.get("outcome")
		return FNOLTurnOutcome(
			session_id=session_id,
			status="processed" if outcome is not None else "collecting",
			missing=missing,
			next_question=(
				{"field": missing[0], **INTAKE_QUESTIONS[missing[0]]} if missing else None
			),
			collected={f: state.get(f) for f in _FIELD_ORDER},
			warnings=state.get("warnings", []),
			turns=state.get("turns", 0),
			degraded=bool(state.get("degraded")),
			outcome=outcome,
		)

	# --- public ----------------------------------------------------------------------------

	def turn(self, req: FNOLTurnRequest) -> FNOLTurnOutcome:
		thread_key = f"{req.tenant_id}::{req.session_id}"
		config = {"configurable": {"thread_id": thread_key}}
		self._touch(thread_key)
		snapshot: IntakeState = self._graph.get_state(config).values
		if snapshot.get("outcome") is not None:
			# completed sessions repeat their verdict — a retried/late turn must never
			# run the crew twice or restart the session (idempotency doctrine)
			return self._outcome(req.session_id, snapshot)
		if snapshot.get("turns", 0) >= self.max_turns:
			raise IntakeSessionExhausted(
				f"intake session exceeded {self.max_turns} turns — file the claim via the Desk"
			)
		try:
			final: IntakeState = self._graph.invoke(
				{
					"messages": [req.message],
					"turn_today": req.today or date.today().isoformat(),
					"turn_fields": req.fields.model_dump() if req.fields else None,
					"tenant_id": req.tenant_id,
					"session_id": req.session_id,
					"channel": req.channel,
				},
				config,
			)
		except Exception:
			# fail-loud paths (e.g. claims.process in _assess) surface as 500s by design —
			# but LangGraph wraps them, so pin the thread key here or on-call debugs blind
			logger.exception("fnol intake turn failed for thread %s", thread_key)
			raise
		return self._outcome(req.session_id, final)
