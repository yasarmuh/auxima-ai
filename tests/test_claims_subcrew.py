# Copyright (c) 2026, Auxilium Tech and contributors
"""Tests for the P3-04 line-specific claims sub-crews — deterministic advisory actions, no LLM.

Load-bearing facts: each line yields its own action set; high-severity/high-value triggers the loss
adjuster; a large loss raises an RI early-warning; an unknown route label fails closed (a routing
mismatch must not silently look like "no actions"); the graph runs the sub-crew node after route_line.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from auxima_ai.claims.schema import FNOLRequest, SubCrewRecommendation, TriageAssessment
from auxima_ai.claims.service import ClaimsCrewService
from auxima_ai.claims.subcrew import run_subcrew
from auxima_ai.intake.llm import LLMResponse


def _fnol(**overrides) -> FNOLRequest:
	body = dict(
		tenant_id="t-sc", claim_ref="CLM-SC-1", loss_type="property",
		incident_date="2026-06-01", reported_date="2026-06-02",
		description="loss narrative", estimated_amount="50000",
	)
	body.update(overrides)
	return FNOLRequest(**body)


def _triage(severity="low", complexity="standard") -> TriageAssessment:
	return TriageAssessment(severity=severity, complexity=complexity, source="heuristic")


def _codes(recs) -> set[str]:
	return {r.action for r in recs}


class TestPropertySubCrew:
	def test_always_reviews_salvage_subrogation(self):
		recs = run_subcrew("property", _fnol(), _triage(), None)
		assert "review_salvage_subrogation" in _codes(recs)

	def test_high_severity_dispatches_adjuster(self):
		recs = run_subcrew("property", _fnol(estimated_amount="1000"), _triage(severity="high"), None)
		assert "dispatch_loss_adjuster" in _codes(recs)

	def test_high_value_dispatches_adjuster_even_if_low_severity(self):
		recs = run_subcrew("property", _fnol(estimated_amount="150000"), _triage(severity="low"), None)
		assert "dispatch_loss_adjuster" in _codes(recs)

	def test_small_low_severity_no_adjuster(self):
		recs = run_subcrew("property", _fnol(estimated_amount="5000"), _triage(severity="low"), None)
		assert "dispatch_loss_adjuster" not in _codes(recs)

	def test_large_loss_raises_ri_early_warning(self):
		recs = run_subcrew("property", _fnol(estimated_amount="2000000"), _triage(severity="high"), None)
		assert "reinsurance_early_warning" in _codes(recs)

	def test_threshold_override(self):
		recs = run_subcrew(
			"property", _fnol(estimated_amount="60000"), _triage(severity="low"), None,
			adjuster_threshold=Decimal("50000"),
		)
		assert "dispatch_loss_adjuster" in _codes(recs)


class TestLiabilitySubCrew:
	def test_complex_refers_legal_panel(self):
		recs = run_subcrew("liability", _fnol(loss_type="liability"), _triage(complexity="complex"), None)
		assert "refer_legal_panel" in _codes(recs)

	def test_always_reserves_defence_costs(self):
		recs = run_subcrew("liability", _fnol(loss_type="liability"), _triage(), None)
		assert "reserve_defence_costs" in _codes(recs)

	def test_simple_low_no_legal_referral(self):
		recs = run_subcrew("liability", _fnol(loss_type="liability"), _triage(severity="low", complexity="standard"), None)
		assert "refer_legal_panel" not in _codes(recs)


class TestSpecialtyMotorMedicalGeneral:
	def test_specialty_dispatches_surveyor(self):
		recs = run_subcrew("specialty", _fnol(loss_type="marine"), _triage(), None)
		assert "dispatch_specialist_surveyor" in _codes(recs)

	def test_motor_najm_pending(self):
		recs = run_subcrew("motor", _fnol(loss_type="motor"), _triage(), None)
		assert "najm_report_pending" in _codes(recs)

	def test_motor_low_severity_fast_track(self):
		recs = run_subcrew("motor", _fnol(loss_type="motor"), _triage(severity="low"), None)
		assert "fast_track_settlement" in _codes(recs)

	def test_medical_nphies_pending(self):
		recs = run_subcrew("medical", _fnol(loss_type="medical"), _triage(), None)
		assert "nphies_adjudication_pending" in _codes(recs)

	def test_general_manual_triage(self):
		recs = run_subcrew("general", _fnol(loss_type="other"), _triage(), None)
		assert "manual_triage" in _codes(recs)

	def test_recommendations_are_typed(self):
		recs = run_subcrew("general", _fnol(loss_type="other"), _triage(), None)
		assert all(isinstance(r, SubCrewRecommendation) for r in recs)


class TestFailClosed:
	def test_unknown_label_raises(self):
		with pytest.raises(ValueError):
			run_subcrew("no_such_crew", _fnol(), _triage(), None)


class TestGraphIntegration:
	"""The sub-crew node runs as part of the LangGraph state machine, after route_line."""

	@staticmethod
	def _stub_invoke(**kw):
		return LLMResponse(
			payload={"severity": "high", "complexity": "complex", "fraud_indicators": []},
			prompt_tokens=1, completion_tokens=1, latency_ms=1,
		)

	def test_outcome_carries_subcrew_actions(self):
		svc = ClaimsCrewService(invoke=self._stub_invoke)
		out = svc.process(_fnol(loss_type="property", estimated_amount="2000000"))
		assert out.status == "ok"
		assert out.subcrew == "property"
		codes = {r.action for r in out.subcrew_actions}
		assert "dispatch_loss_adjuster" in codes  # high severity from stub
		assert "reinsurance_early_warning" in codes  # 2M estimate
		assert out.audit_trail == [
			"validate_fnol", "triage", "reserve_suggest", "route_line", "subcrew_actions",
		]

	def test_rejected_fnol_has_no_subcrew_actions(self):
		svc = ClaimsCrewService(invoke=self._stub_invoke)
		out = svc.process(_fnol(reported_date="2026-05-01"))  # before incident -> rejected
		assert out.status == "rejected"
		assert out.subcrew_actions == []
