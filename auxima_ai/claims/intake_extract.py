# Copyright (c) 2026, Auxilium Tech and contributors
"""Deterministic FNOL field extractor for the multi-turn intake graph (P3-01c).

Pure and bilingual (EN + AR, incl. Arabic-Indic digits). Every function FAILS CLOSED —
ambiguous loss-type signals, future/invalid dates and unmarked bare numbers all return
``None`` ("ask the reporter"), never a guess. The keyword maps are COMMERCIAL defaults,
not regulator facts; they are deliberately narrow — recall is the LLM extract step's job
(local-only), precision is this module's.
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

# Arabic-Indic (U+0660-0669) and Eastern Arabic-Indic (U+06F0-06F9) → Western digits.
_DIGIT_TRANSLATION = {0x0660 + i: str(i) for i in range(10)} | {
	0x06F0 + i: str(i) for i in range(10)
}

# English keywords match on word boundaries (so "car" never fires inside "cargo");
# Arabic keywords match as substrings (the definite article ال glues onto the word,
# defeating \b). Narrow on purpose — one wrong guess is worse than one extra question.
_EN_KEYWORDS = {
	"motor": ("motor", "car", "vehicle", "truck", "van", "crash", "collision"),
	"property": ("property", "fire", "warehouse", "flood", "burglary"),
	"medical": ("medical", "hospital", "injury", "injured"),
	"liability": ("liability", "liable", "third party"),
	"marine": ("marine", "cargo", "vessel", "shipment", "at sea"),
	"engineering": ("engineering", "machinery", "construction"),
	"other": ("other",),
}
_AR_KEYWORDS = {
	"motor": ("سيارة", "مركبات", "مركبة", "حادث مروري"),
	"property": ("حريق", "ممتلكات", "مستودع", "سطو"),
	"medical": ("مستشفى", "طبي", "إصابة"),
	"liability": ("مسؤولية",),
	"marine": ("بحري", "شحنة", "سفينة"),
	"engineering": ("هندسي", "إنشاءات", "معدات"),
	"other": ("أخرى",),
}

_ISO_DATE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DAY_FIRST = re.compile(r"\b(\d{1,2})[/](\d{1,2})[/](\d{4})\b")  # KSA convention dd/mm/yyyy
_RELATIVE_DAYS = {"yesterday": 1, "أمس": 1, "امس": 1, "today": 0, "اليوم": 0}

_NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_CURRENCY = r"(?:SAR|ر\.س|ريال|riyals?)"
# Word magnitudes ("20 thousand riyals" / "٢٠ ألف ريال") are deterministic territory —
# a live 0.5B model garbled "20 thousand" into 200000; money never comes from the LLM.
_MAGNITUDE_WORDS = {
	"thousand": Decimal("1000"), "ألف": Decimal("1000"), "الف": Decimal("1000"),
	"million": Decimal("1000000"), "مليون": Decimal("1000000"),
}
_MAGNITUDE = r"(?:thousand|million|ألف|الف|مليون)"
_MARKED_AMOUNT = re.compile(
	rf"(?:{_CURRENCY}\s*({_NUMBER})(?:\s+({_MAGNITUDE}))?)"
	rf"|(?:({_NUMBER})(?:\s+({_MAGNITUDE}))?\s*{_CURRENCY})",
	re.IGNORECASE,
)
_SUFFIXED_AMOUNT = re.compile(rf"\b({_NUMBER})\s*([km])\b", re.IGNORECASE)
_WORD_MAGNITUDE_AMOUNT = re.compile(rf"\b({_NUMBER})\s+({_MAGNITUDE})\b", re.IGNORECASE)
_BARE_AMOUNT = re.compile(rf"\b({_NUMBER})\b")

_SKIP_TOKENS = frozenset({"skip", "unknown", "n/a", "لا أعرف", "لا اعرف", "غير معروف"})


def _normalize(text: str) -> str:
	return text.translate(_DIGIT_TRANSLATION)


def extract_loss_type(text: str) -> str | None:
	"""Loss type from free text; exactly one type must match, else None (ask, don't guess)."""
	lowered = _normalize(text).lower()
	matched = {
		loss_type
		for loss_type, words in _EN_KEYWORDS.items()
		if any(re.search(rf"\b{re.escape(w)}\b", lowered) for w in words)
	} | {
		loss_type
		for loss_type, words in _AR_KEYWORDS.items()
		if any(w in lowered for w in words)
	}
	return matched.pop() if len(matched) == 1 else None


def _valid_past_date(year: int, month: int, day: int, today: date) -> str | None:
	try:
		candidate = date(year, month, day)
	except ValueError:
		return None
	return candidate.isoformat() if candidate <= today else None


def extract_incident_date(text: str, *, today: date) -> str | None:
	"""Incident date from free text. An EXPLICIT date that is invalid or in the future fails
	closed to None (no fall-through to relative words — the reporter typed something specific
	and wrong; re-ask rather than reinterpret)."""
	normalized = _normalize(text)
	if m := _ISO_DATE.search(normalized):
		return _valid_past_date(int(m[1]), int(m[2]), int(m[3]), today)
	if m := _DAY_FIRST.search(normalized):
		return _valid_past_date(int(m[3]), int(m[2]), int(m[1]), today)
	lowered = normalized.lower()
	for word, days_ago in _RELATIVE_DAYS.items():
		if word in lowered:
			return (today - timedelta(days=days_ago)).isoformat()
	return None


def _format_amount(value: Decimal) -> str:
	if value == value.to_integral_value():
		return str(int(value))
	return str(value)


def _parse_number(raw: str) -> Decimal | None:
	try:
		return Decimal(raw.replace(",", ""))
	except InvalidOperation:
		return None


def extract_amount(text: str, *, assume_amount: bool = False) -> str | None:
	"""Loss amount from free text. A number counts only when currency-marked (SAR/ريال/ر.س)
	or magnitude-marked (50k/1.2m/"20 thousand"/"٢٠ ألف") — a bare number could be a plate
	or CR number. Pass ``assume_amount=True`` only when the intake JUST asked for the amount."""
	normalized = _normalize(text)
	if m := _MARKED_AMOUNT.search(normalized):
		value = _parse_number(m[1] or m[3])
		if value is not None:
			word = (m[2] or m[4] or "").lower()
			return _format_amount(value * _MAGNITUDE_WORDS.get(word, Decimal("1")))
		return None
	if m := _SUFFIXED_AMOUNT.search(normalized):
		value = _parse_number(m[1])
		if value is not None:
			multiplier = Decimal("1000") if m[2].lower() == "k" else Decimal("1000000")
			return _format_amount(value * multiplier)
	if assume_amount:
		if m := _WORD_MAGNITUDE_AMOUNT.search(normalized):
			value = _parse_number(m[1])
			if value is not None:
				return _format_amount(value * _MAGNITUDE_WORDS[m[2].lower()])
		if m := _BARE_AMOUNT.search(normalized):
			value = _parse_number(m[1])
			return _format_amount(value) if value is not None else None
	return None


def is_skip_amount(text: str) -> bool:
	"""True when the reporter explicitly declines the amount question ("skip"/"غير معروف")."""
	return text.strip().lower() in _SKIP_TOKENS
