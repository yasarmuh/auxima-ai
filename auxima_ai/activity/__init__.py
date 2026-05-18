"""Auxima Activity rows — the CRM spine'\''s append-only event log.

CLAUDE §4 invariant: every state change visible to a broker handler
emits exactly one ``Auxima Activity`` row. This package builds those
rows (the Frappe-side ``Auxima Activity`` doctype reads them).

:mod:`.row` provides:
  - :class:`ActivityRow` — frozen dataclass with the canonical fields
    and per-field validation.
  - :class:`RetentionClass` — three-bucket classification per S-25
    (WORM-audit / operational / ephemeral).
  - :func:`build_activity_row` — constructor that fills the ULID +
    UTC timestamp + redacted payload defaults.
"""
