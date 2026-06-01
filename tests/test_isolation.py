"""The CLAUDE.md §3 Rule 2 firewall: the sidecar NEVER imports frappe/auxima/erpnext.

This test fails the build if any module under `auxima_ai/` imports a forbidden name —
catching both static `import frappe` AND dynamic `__import__("frappe")` / `importlib`
paths via an import-hook installed BEFORE auxima_ai is first imported.

Why this matters: the moment the sidecar can resolve `frappe`, its release cycle couples
to the Frappe upgrade cadence, the test surface bloats with Frappe internals, and the
clean MIT-on-MIT licensing posture is at risk.
"""
from __future__ import annotations

import importlib
import sys

FORBIDDEN_TOP_LEVEL = {"frappe", "auxima", "erpnext"}


def test_no_forbidden_modules_already_imported():
    """If something in auxima_ai's import chain already pulled frappe etc., this fails."""
    # Force auxima_ai to import (it may already be cached).
    importlib.import_module("auxima_ai")
    for mod_name in list(sys.modules):
        top = mod_name.split(".", 1)[0]
        if top in FORBIDDEN_TOP_LEVEL:
            raise AssertionError(
                f"forbidden top-level module imported by auxima_ai: {mod_name}"
            )


def test_no_static_import_strings_in_source():
    """Grep-style fail-fast: source files must not contain a static `import frappe` etc.

    Catches the easy class of mistake — code review backstop. Pairs with the dynamic
    runtime check above (which catches __import__/importlib paths).

    Uses word boundaries so that ``from auxima_ai.config import X`` (the sidecar's own
    package) does NOT collide with the forbidden top-level ``auxima`` (the Frappe app).
    """
    import pathlib
    import re

    pkg_dir = pathlib.Path(__file__).resolve().parent.parent / "auxima_ai"
    violations: list[str] = []
    for py_file in pkg_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_TOP_LEVEL:
            # Match the forbidden name as a whole token (followed by space, dot, or
            # comma — not by another word character that would make it a different
            # package name like `auxima_ai`).
            patterns = (
                rf"\bimport\s+{re.escape(forbidden)}\b(?!_)",
                rf"\bfrom\s+{re.escape(forbidden)}\b(?!_)",
                rf"__import__\(\s*['\"]{re.escape(forbidden)}['\"]",
                rf"importlib\.import_module\(\s*['\"]{re.escape(forbidden)}['\"]",
            )
            for pat in patterns:
                if re.search(pat, text):
                    violations.append(f"{py_file}: matched /{pat}/")
    assert not violations, "forbidden imports found:\n  " + "\n  ".join(violations)


def test_blocking_import_hook_refuses_dynamic_frappe():
    """Verify that any code path inside auxima_ai that tries to dynamically resolve frappe
    will fail. We install a meta_path finder that raises on the forbidden names."""

    class Blocker:
        def find_spec(self, fullname, path, target=None):
            top = fullname.split(".", 1)[0]
            if top in FORBIDDEN_TOP_LEVEL:
                raise ImportError(
                    f"auxima-ai must not import {fullname} — this is a hard firewall "
                    f"(CLAUDE.md §3 Rule 2)"
                )
            return None

    blocker = Blocker()
    sys.meta_path.insert(0, blocker)
    # Snapshot the auxima_ai.* module objects BEFORE the destructive re-import.
    # Without restoring them afterwards, a later fresh import mints NEW module
    # objects (and NEW exception classes), so a test running after this one that
    # compares an exception raised by an already-imported module against a
    # freshly-imported class fails an `isinstance`/`pytest.raises` check despite
    # the types being "the same". Restore originals in the finally to keep
    # module/class identity stable across the whole suite.
    _saved = {
        name: mod for name, mod in sys.modules.items()
        if name == "auxima_ai" or name.startswith("auxima_ai.")
    }
    try:
        # Re-import auxima_ai's modules — none of them should trigger the blocker.
        for mod_name in list(sys.modules):
            if mod_name == "auxima_ai" or mod_name.startswith("auxima_ai."):
                del sys.modules[mod_name]
        importlib.import_module("auxima_ai")
        importlib.import_module("auxima_ai.main")
        importlib.import_module("auxima_ai.config")
        importlib.import_module("auxima_ai.auth")
    finally:
        sys.meta_path.remove(blocker)
        # Drop any newly-minted modules and restore the original objects.
        for mod_name in list(sys.modules):
            if mod_name == "auxima_ai" or mod_name.startswith("auxima_ai."):
                del sys.modules[mod_name]
        sys.modules.update(_saved)
