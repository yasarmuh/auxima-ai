# Copyright (c) 2026, Auxilium Tech and contributors
"""P3-01c deterministic FNOL-intake field extractor — bilingual, fail-closed (returns None
rather than guessing: ambiguous loss types, future dates and unmarked bare numbers are all
"ask the reporter", never a silent wrong value)."""
from __future__ import annotations

from datetime import date

from auxima_ai.claims.intake_extract import (
	extract_amount,
	extract_incident_date,
	extract_loss_type,
	is_skip_amount,
)

TODAY = date(2026, 7, 2)


class TestLossType:
	def test_english_keywords(self):
		assert extract_loss_type("Our delivery van was in a car crash") == "motor"
		assert extract_loss_type("A fire broke out at the site") == "property"
		assert extract_loss_type("Employee was admitted to hospital after the fall") == "medical"
		assert extract_loss_type("A third party is holding us liable for damages") == "liability"
		assert extract_loss_type("The cargo was damaged at sea") == "marine"

	def test_arabic_keywords(self):
		assert extract_loss_type("وقع حادث سيارة أمام المكتب") == "motor"
		assert extract_loss_type("نشب حريق في المستودع") == "property"
		assert extract_loss_type("تلفت الشحنة أثناء النقل البحري") == "marine"

	def test_literal_type_names_parse(self):
		# the intake question offers the type names as options — replies must parse
		assert extract_loss_type("motor") == "motor"
		assert extract_loss_type("Property") == "property"
		assert extract_loss_type("مركبات") == "motor"
		assert extract_loss_type("ممتلكات") == "property"

	def test_ambiguous_is_none(self):
		# motor + property signals in one message → ask, never guess
		assert extract_loss_type("the truck crashed into the warehouse and started a fire") is None

	def test_no_signal_is_none(self):
		assert extract_loss_type("something bad happened yesterday") is None
		assert extract_loss_type("") is None


class TestIncidentDate:
	def test_iso_date(self):
		assert extract_incident_date("It happened on 2026-06-28 at noon", today=TODAY) == "2026-06-28"

	def test_day_first_slash_date(self):
		# KSA convention: dd/mm/yyyy
		assert extract_incident_date("on 28/06/2026 around 3pm", today=TODAY) == "2026-06-28"

	def test_relative_words(self):
		assert extract_incident_date("it happened yesterday evening", today=TODAY) == "2026-07-01"
		assert extract_incident_date("حصل الحادث أمس", today=TODAY) == "2026-07-01"
		assert extract_incident_date("the pipe burst today", today=TODAY) == "2026-07-02"
		assert extract_incident_date("وقع الحادث اليوم", today=TODAY) == "2026-07-02"

	def test_arabic_indic_digits(self):
		assert extract_incident_date("بتاريخ ٢٠٢٦-٠٦-٣٠", today=TODAY) == "2026-06-30"

	def test_future_date_fails_closed(self):
		assert extract_incident_date("on 2026-08-01", today=TODAY) is None

	def test_invalid_and_absent(self):
		assert extract_incident_date("on 2026-13-45 supposedly", today=TODAY) is None
		assert extract_incident_date("no date here", today=TODAY) is None


class TestAmount:
	def test_currency_marked_amounts(self):
		assert extract_amount("estimated at SAR 50,000") == "50000"
		assert extract_amount("الخسارة حوالي 5000 ريال") == "5000"
		assert extract_amount("about 75000 ر.س") == "75000"

	def test_magnitude_suffixes(self):
		assert extract_amount("roughly 50k in damages") == "50000"
		assert extract_amount("could reach 1.2m") == "1200000"

	def test_word_magnitudes(self):
		# live 2026-07-02: a 0.5B model turned "20 thousand riyals" into 200000 — prose
		# magnitudes must be DETERMINISTIC territory, the LLM never supplies money
		assert extract_amount("damage maybe 20 thousand riyals") == "20000"
		assert extract_amount("could be 1.5 million SAR") == "1500000"
		assert extract_amount("الخسارة حوالي 20 ألف ريال") == "20000"
		assert extract_amount("قد تصل إلى 2 مليون ريال") == "2000000"
		assert extract_amount("20 thousand") is None, "word magnitude still needs a currency marker"
		assert extract_amount("20 thousand", assume_amount=True) == "20000"

	def test_arabic_indic_digits(self):
		assert extract_amount("٥٠٠٠ ريال تقريبا") == "5000"

	def test_bare_number_needs_context(self):
		# an unmarked number could be a plate/CR/phone — only accept when we asked for the amount
		assert extract_amount("50000") is None
		assert extract_amount("50000", assume_amount=True) == "50000"
		assert extract_amount("50,000", assume_amount=True) == "50000"

	def test_no_amount(self):
		assert extract_amount("no figures yet", assume_amount=True) is None


class TestSkipAmount:
	def test_skip_tokens(self):
		assert is_skip_amount("skip")
		assert is_skip_amount("Unknown")
		assert is_skip_amount("لا أعرف")
		assert is_skip_amount("غير معروف")

	def test_ordinary_text_is_not_skip(self):
		assert not is_skip_amount("maybe 5000")
		assert not is_skip_amount("")
