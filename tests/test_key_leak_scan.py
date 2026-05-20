"""Tests for the AC-10 sidecar-key leak scanner (S-54 R11 / AC-10).

AC-10: a pre-commit hook greps for an actual key VALUE assigned to
``SIDECAR_PRIMARY_KEY_B64`` / ``SIDECAR_SECONDARY_KEY_B64`` and rejects the
commit — without false-positiving on the variable NAME (docs, .env.example,
``=$REF``, empty assignment). A real key is ``base64(32 bytes)`` = 44 chars.

The scanner MUST NOT itself echo the matched secret (that would just move the
leak from git to the terminal/CI log) — findings carry the var name +
line number only.
"""
from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

from auxima_ai.key_leak_scan import scan_paths, scan_text

# A syntactically real key value: base64 of 32 bytes -> exactly 44 chars.
_REAL_KEY = base64.b64encode(bytes(range(32))).decode("ascii")
_REAL_KEY_2 = base64.b64encode(bytes(range(31, -1, -1))).decode("ascii")


def test_real_key_value_is_detected() -> None:
    findings = scan_text(f"SIDECAR_PRIMARY_KEY_B64={_REAL_KEY}\n")
    assert len(findings) == 1
    assert findings[0].var == "SIDECAR_PRIMARY_KEY_B64"
    assert findings[0].line == 1


def test_secondary_key_value_is_detected() -> None:
    findings = scan_text(f'export SIDECAR_SECONDARY_KEY_B64="{_REAL_KEY_2}"')
    assert len(findings) == 1
    assert findings[0].var == "SIDECAR_SECONDARY_KEY_B64"


def test_yaml_colon_assignment_is_detected() -> None:
    findings = scan_text(f"SIDECAR_PRIMARY_KEY_B64: {_REAL_KEY}")
    assert len(findings) == 1


def test_finding_never_contains_the_secret_value() -> None:
    [finding] = scan_text(f"SIDECAR_PRIMARY_KEY_B64={_REAL_KEY}")
    rendered = repr(finding) + finding.message()
    assert _REAL_KEY not in rendered


# ---------------------------------------------------------------------------
# Must NOT false-positive
# ---------------------------------------------------------------------------


def test_bare_variable_name_in_prose_is_not_flagged() -> None:
    assert scan_text("Set SIDECAR_PRIMARY_KEY_B64 in the Site Config.") == []


def test_empty_assignment_is_not_flagged() -> None:
    assert scan_text("SIDECAR_PRIMARY_KEY_B64=") == []


def test_shell_reference_is_not_flagged() -> None:
    assert scan_text("SIDECAR_PRIMARY_KEY_B64=${OTHER_VAR}") == []
    assert scan_text("SIDECAR_PRIMARY_KEY_B64=$OTHER_VAR") == []


def test_placeholder_is_not_flagged() -> None:
    assert scan_text("SIDECAR_PRIMARY_KEY_B64=<base64-32-bytes>") == []
    assert scan_text("SIDECAR_PRIMARY_KEY_B64=changeme") == []


def test_unrelated_base64_blob_is_not_flagged() -> None:
    # A 44-char base64 NOT assigned to a sidecar key var is out of scope.
    assert scan_text(f"some_other_field = {_REAL_KEY}") == []


# ---------------------------------------------------------------------------
# scan_paths + CLI exit code
# ---------------------------------------------------------------------------


def test_scan_paths_reports_file_and_line(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text(f"X=1\nSIDECAR_PRIMARY_KEY_B64={_REAL_KEY}\n", encoding="utf-8")
    findings = scan_paths([f])
    assert len(findings) == 1
    assert findings[0].line == 2
    assert findings[0].path == f


def test_cli_exits_nonzero_on_leak_and_zero_when_clean(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty.env"
    dirty.write_text(f"SIDECAR_PRIMARY_KEY_B64={_REAL_KEY}\n", encoding="utf-8")
    clean = tmp_path / "clean.env"
    clean.write_text("SIDECAR_PRIMARY_KEY_B64=${REF}\n", encoding="utf-8")

    leak = subprocess.run(
        [sys.executable, "-m", "auxima_ai.key_leak_scan", "--paths", str(dirty)],
        capture_output=True, text=True,
    )
    assert leak.returncode == 1
    assert "SIDECAR_PRIMARY_KEY_B64" in leak.stdout + leak.stderr
    assert _REAL_KEY not in leak.stdout + leak.stderr  # never echo the secret

    ok = subprocess.run(
        [sys.executable, "-m", "auxima_ai.key_leak_scan", "--paths", str(clean)],
        capture_output=True, text=True,
    )
    assert ok.returncode == 0
