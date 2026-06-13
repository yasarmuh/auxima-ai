# Copyright (c) 2026, Auxilium Tech and contributors
"""Line-specific claims sub-crews (P3-04) — deterministic advisory next-actions, no LLM.

``ClaimsCrewService._route_line`` resolves an FNOL's ``loss_type`` to a sub-crew label; this module
is what that label dispatches to. Each sub-crew turns the FNOL + triage + reserve into a list of
**advisory next-actions** (loss-adjuster dispatch, reinsurance early-warning, subrogation review,
legal-panel referral, specialist survey, connector-pending notes). The Desk maps each ``action`` code
to a localized label and the broker acts on it — the crew never dispatches anything itself
(advisory-AI-not-into-immutable-artefacts; CLAUDE.md §4).

The logic is **pure and deterministic** — rules over the triage severity and the estimated amount, no
LLM call, so it is fully reproducible and carries no egress concern. The amount thresholds are
**commercial parameters**, not regulator facts (mirrors ``service._HEURISTIC_*``): a high-value loss
*may* reach a treaty attachment, so we raise an RI early-warning for the broker to check against the
real ``Reinsurance Treaty`` in the Frappe app — the sidecar has no treaty data and does not pretend to.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Callable

from auxima_ai.claims.schema import FNOLRequest, SubCrewRecommendation, TriageAssessment

# Commercial advisory thresholds (SAR) — defaults, not regulator facts; bootstrap/tenant may override.
DEFAULT_ADJUSTER_THRESHOLD = Decimal("100000")  # above this, recommend an independent loss adjuster
DEFAULT_RI_EARLY_WARNING = Decimal("1000000")  # above this, the loss may reach a treaty attachment


def _amount(req: FNOLRequest) -> Decimal:
	return Decimal(req.estimated_amount or "0")


def _adjuster_action(req: FNOLRequest, triage: TriageAssessment, adjuster_threshold: Decimal):
	"""Recommend an independent loss adjuster on a high-severity or high-value loss."""
	if triage.severity == "high" or _amount(req) >= adjuster_threshold:
		return SubCrewRecommendation(
			action="dispatch_loss_adjuster",
			priority="high",
			rationale="High-severity or high-value loss — independent quantum assessment recommended.",
		)
	return None


def _ri_early_warning(req: FNOLRequest, ri_threshold: Decimal):
	"""Flag a loss large enough that it may reach a reinsurance treaty attachment."""
	if _amount(req) >= ri_threshold:
		return SubCrewRecommendation(
			action="reinsurance_early_warning",
			priority="high",
			rationale="Estimated loss may reach a treaty attachment — verify reinsurance recovery in the Claim.",
		)
	return None


def _property_subcrew(req, triage, reserve, *, adjuster_threshold, ri_threshold):
	out = [
		SubCrewRecommendation(
			action="review_salvage_subrogation",
			priority="medium",
			rationale="Property losses commonly carry salvage and third-party subrogation recovery.",
		)
	]
	for rec in (_adjuster_action(req, triage, adjuster_threshold), _ri_early_warning(req, ri_threshold)):
		if rec:
			out.append(rec)
	return out


def _liability_subcrew(req, triage, reserve, **_kw):
	out = []
	if triage.complexity == "complex" or triage.severity == "high":
		out.append(
			SubCrewRecommendation(
				action="refer_legal_panel",
				priority="high",
				rationale="Complex or high-severity liability claim — refer to the legal panel for assessment.",
			)
		)
	out.append(
		SubCrewRecommendation(
			action="reserve_defence_costs",
			priority="medium",
			rationale="Liability claims should carry a reserve for legal defence costs.",
		)
	)
	return out


def _specialty_subcrew(req, triage, reserve, *, adjuster_threshold, ri_threshold):
	out = [
		SubCrewRecommendation(
			action="dispatch_specialist_surveyor",
			priority="high",
			rationale="Marine/engineering losses need a specialist surveyor, not a general adjuster.",
		)
	]
	rec = _ri_early_warning(req, ri_threshold)
	if rec:
		out.append(rec)
	return out


def _motor_subcrew(req, triage, reserve, **_kw):
	out = [
		SubCrewRecommendation(
			action="najm_report_pending",
			priority="medium",
			rationale="Reconcile against the Najm accident report when available (P3-02 connector).",
		)
	]
	if triage.severity == "low":
		out.append(
			SubCrewRecommendation(
				action="fast_track_settlement",
				priority="low",
				rationale="Low-severity motor claim — eligible for fast-track settlement.",
			)
		)
	return out


def _medical_subcrew(req, triage, reserve, **_kw):
	return [
		SubCrewRecommendation(
			action="nphies_adjudication_pending",
			priority="high",
			rationale="Health claim — NPHIES eligibility/adjudication pending; special-category data, in-Kingdom only.",
		)
	]


def _general_subcrew(req, triage, reserve, **_kw):
	return [
		SubCrewRecommendation(
			action="manual_triage",
			priority="medium",
			rationale="No specialised sub-crew for this line — route to manual claims handling.",
		)
	]


# Sub-crew handlers keyed by the label `service._SUBCREW_BY_LINE` resolves.
_SUBCREWS: dict[str, Callable] = {
	"property": _property_subcrew,
	"liability": _liability_subcrew,
	"specialty": _specialty_subcrew,
	"motor": _motor_subcrew,
	"medical": _medical_subcrew,
	"general": _general_subcrew,
}


def run_subcrew(
	label: str,
	request: FNOLRequest,
	triage: TriageAssessment,
	reserve,
	adjuster_threshold: Decimal = DEFAULT_ADJUSTER_THRESHOLD,
	ri_threshold: Decimal = DEFAULT_RI_EARLY_WARNING,
) -> list[SubCrewRecommendation]:
	"""Dispatch to the line's sub-crew and return its advisory next-actions.

	Fail-closed: an unknown label is a routing/registry mismatch (a programming error), so we raise
	rather than silently return no actions — a dropped sub-crew route must not look like "nothing to do".
	"""
	handler = _SUBCREWS.get(label)
	if handler is None:
		raise ValueError(f"no sub-crew registered for route label {label!r}")
	return handler(
		request, triage, reserve, adjuster_threshold=adjuster_threshold, ri_threshold=ri_threshold
	)
