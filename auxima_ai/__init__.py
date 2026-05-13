"""auxima-ai — the FastAPI sidecar for Auxima Insure.

Architectural invariants (enforced by tests/test_isolation.py + CI):
  - This package NEVER imports `frappe`, `auxima`, or `erpnext`. Communication with the
    `auxima` Frappe app is REST-only.
  - The sidecar runs as a separate process in a separate directory tree.
  - The sidecar is licensed independently (MIT) and may be packaged separately.

See CLAUDE.md §3 Rule 2 + Docs/Planning/Development-Plan-v1.md §4.
"""
__version__ = "0.1.0"
