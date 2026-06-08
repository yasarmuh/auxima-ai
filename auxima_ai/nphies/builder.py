"""auxima_ai.nphies.builder — FHIR R4 resource construction for KSA NPHIES (P3-03 slice).

Builds the FHIR R4 resources NPHIES exchanges — ``CoverageEligibilityRequest`` (eligibility
check) and ``Claim`` (submission) — via fhir.resources (Pydantic-v2 validation, so malformed
input fails-closed at construction rather than producing a bad payload). Returns validated
dicts ready to be wrapped in an NPHIES message Bundle. Money is Decimal-exact.

SCOPE (honest): base FHIR R4 construction + validation ONLY. DEFERRED (external): NPHIES
PROFILE conformance (KSA extensions/code-systems), the message-Bundle envelope, and the live
NPHIES endpoint exchange (onboarding-gated). SCOPE-CONDITIONAL: whether THIS broker performs
health-claims administration at all (broker-vs-TPA) is an open product decision (plan P3-03) —
this module is the offline-computable capability either way. No frappe/auxima import (sidecar
Rule 2 — process isolation).
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from fhir.resources.claim import Claim
from fhir.resources.coverageeligibilityrequest import CoverageEligibilityRequest

_CURRENCY = "SAR"


def _money(value) -> Decimal:
	return Decimal(str(value or 0))


def _q(amount: Decimal) -> Decimal:
	return amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def build_eligibility_request(
	patient_ref: str, insurer_ref: str, created: str, purpose: str = "validation"
) -> dict:
	"""A FHIR R4 CoverageEligibilityRequest. ``created`` must be a valid FHIR date (else it
	raises at construction — fail-closed). ``*_ref`` are FHIR references (e.g. 'Patient/123')."""
	resource = CoverageEligibilityRequest(
		status="active",
		purpose=[purpose],
		patient={"reference": patient_ref},
		created=created,
		insurer={"reference": insurer_ref},
	)
	return resource.model_dump()


def build_claim(
	patient_ref: str,
	provider_ref: str,
	coverage_ref: str,
	items: list[dict],
	created: str,
	claim_type: str = "institutional",
) -> dict:
	"""A FHIR R4 Claim. ``items`` = list of {service_code, unit_price, quantity}. Each line's net
	and the claim total are Decimal-exact (unit_price x quantity). Fail-closed via FHIR validation."""
	claim_items: list[dict] = []
	total = Decimal(0)
	for index, line in enumerate(items, start=1):
		unit = _money(line["unit_price"])
		qty = _money(line.get("quantity", 1))
		net = _q(unit * qty)
		total += net
		claim_items.append(
			{
				"sequence": index,
				"productOrService": {"coding": [{"code": str(line["service_code"])}]},
				"unitPrice": {"value": unit, "currency": _CURRENCY},
				"quantity": {"value": qty},
				"net": {"value": net, "currency": _CURRENCY},
			}
		)
	total = _q(total)
	resource = Claim(
		status="active",
		type={"coding": [{"code": claim_type}]},
		use="claim",
		patient={"reference": patient_ref},
		created=created,
		provider={"reference": provider_ref},
		priority={"coding": [{"code": "normal"}]},
		insurance=[{"sequence": 1, "focal": True, "coverage": {"reference": coverage_ref}}],
		item=claim_items,
		total={"value": total, "currency": _CURRENCY},
	)
	return resource.model_dump()
