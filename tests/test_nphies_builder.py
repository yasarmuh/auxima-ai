"""Tests for auxima_ai.nphies.builder (P3-03) — FHIR R4 construction, Decimal money totals,
and fail-closed validation on malformed input."""
from __future__ import annotations

from decimal import Decimal

import pytest

from auxima_ai.nphies.builder import build_claim, build_eligibility_request


def test_eligibility_request_shape():
	d = build_eligibility_request("Patient/123", "Organization/ins", "2026-06-08")
	assert d["resourceType"] == "CoverageEligibilityRequest"
	assert d["status"] == "active"
	assert d["patient"]["reference"] == "Patient/123"
	assert d["insurer"]["reference"] == "Organization/ins"


def test_eligibility_request_invalid_date_fails_closed():
	with pytest.raises(Exception):
		build_eligibility_request("Patient/123", "Organization/ins", "not-a-date")


def test_claim_shape_and_decimal_total():
	d = build_claim(
		"Patient/123", "Organization/prov", "Coverage/cov",
		items=[
			{"service_code": "X1", "unit_price": "100.00", "quantity": 2},  # net 200.00
			{"service_code": "X2", "unit_price": "50.50", "quantity": 1},   # net 50.50
		],
		created="2026-06-08",
	)
	assert d["resourceType"] == "Claim"
	assert len(d["item"]) == 2
	assert Decimal(str(d["total"]["value"])) == Decimal("250.50")
	assert Decimal(str(d["item"][0]["net"]["value"])) == Decimal("200.00")


def test_claim_invalid_status_fails_closed():
	with pytest.raises(Exception):
		build_claim(
			"Patient/123", "Organization/prov", "Coverage/cov",
			items=[{"service_code": "X1", "unit_price": "10", "quantity": 1}],
			created="bad-date",
		)
