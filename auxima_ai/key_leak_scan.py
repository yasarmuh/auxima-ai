"""Sidecar-key leak scanner — the AC-10 pre-commit guard (S-54 R11).

S-54 R11: the sidecar HMAC keys (``SIDECAR_PRIMARY_KEY_B64`` /
``SIDECAR_SECONDARY_KEY_B64``) are ``base64(32 random bytes)`` and MUST NEVER
be checked into git. AC-10 requires a pre-commit hook that rejects a commit
carrying an actual key VALUE — while NOT false-positiving on the variable
NAME (which legitimately appears in docs, ``.env.example``, and config code
as ``=$REF`` / empty / placeholder).

Detection rule: the variable name, an ``=`` or ``:`` assignment, then a token
that is *shaped exactly* like a 32-byte base64 secret (43 base64 chars + one
``=`` pad = 44 chars). Shell refs (``$VAR``, ``${VAR}``), empty assignments,
and angle/word placeholders do not match that shape, so they pass.

Run as a hook:  ``python -m auxima_ai.key_leak_scan``  (scans ``git diff
--cached`` staged files; exits 1 on any finding). Or ``--paths f1 f2`` to
scan explicit files (used by tests + ad-hoc audits).

Self-protection: a finding reports the variable name + file:line ONLY. It
never echoes the matched secret — that would relocate the leak from git into
the terminal / CI log.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

#: ``base64(32 bytes)`` is 43 significant chars + exactly one ``=`` pad.
#: The trailing negative lookahead stops a match inside a longer base64 blob.
_KEY_VALUE_RE = re.compile(
    r"(SIDECAR_(?:PRIMARY|SECONDARY)_KEY_B64)\s*[=:]\s*['\"]?"
    r"([A-Za-z0-9+/]{43}=)(?![A-Za-z0-9+/=])"
)


@dataclass(frozen=True)
class Finding:
    """A detected key-value leak. Carries NO secret material."""

    var: str
    line: int
    path: Path | None = None

    def message(self) -> str:
        where = f"{self.path}:" if self.path is not None else "line "
        return (
            f"{where}{self.line}: {self.var} appears to be assigned a literal "
            f"32-byte base64 key value — never commit a real key (S-54 R11 / AC-10)."
        )


def scan_text(text: str, *, path: Path | None = None) -> list[Finding]:
    """Return one :class:`Finding` per line assigning a real key value."""
    findings: list[Finding] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        match = _KEY_VALUE_RE.search(line)
        if match:
            findings.append(Finding(var=match.group(1), line=lineno, path=path))
    return findings


def scan_paths(paths: list[Path]) -> list[Finding]:
    """Scan each path; unreadable/binary files are skipped (not a leak signal)."""
    findings: list[Finding] = []
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        findings.extend(scan_text(text, path=Path(path)))
    return findings


def _staged_files() -> list[Path]:
    """Files staged for commit (the pre-commit hook's scan set)."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True, text=True, check=True,
    )
    return [Path(p) for p in out.stdout.splitlines() if p.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject committed sidecar key values (AC-10).")
    parser.add_argument(
        "--paths", nargs="*", type=Path,
        help="Explicit files to scan. Default: git-staged files.",
    )
    args = parser.parse_args(argv)

    paths = args.paths if args.paths is not None else _staged_files()
    findings = scan_paths(paths)
    if findings:
        print("BLOCKED: a sidecar key value would be committed:", file=sys.stderr)
        for f in findings:
            print(f"  {f.message()}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
