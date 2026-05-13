# auxima-ai

The Auxima Insure AI sidecar — FastAPI service for the multi-agent CRM.

This repo is **deliberately separate** from the Frappe `auxima` app (see CLAUDE.md §3 Rule 2 and Docs/Planning/Development-Plan-v1.md §4). The sidecar:

- Communicates with the Frappe app **only over REST**.
- **NEVER imports `frappe`, `auxima`, or `erpnext`** — enforced by `tests/test_isolation.py`.
- Runs as a separate process (port 8088 default).
- Is licensed independently (MIT).
- Iterates on its own release cadence — independent of Frappe upgrades.

## What's in this P0-C scaffold

The minimum to prove the sidecar pattern works end-to-end:

- `auxima_ai/main.py` — FastAPI app with `/healthz` (unauthenticated) and `/v1/whoami` (auth-required).
- `auxima_ai/auth.py` — shared-secret middleware (`X-Auxima-Sidecar-Token`) with constant-time compare; fails closed if the secret env var is empty.
- `auxima_ai/config.py` — env-driven Pydantic Settings (12-factor).
- `tests/test_healthz.py` — covers the auth happy path, missing token (401), wrong token (401), unconfigured secret (503 — fail-closed).
- `tests/test_isolation.py` — three-layer firewall: (a) checks no forbidden module is already in `sys.modules`, (b) greps source for `import frappe` / `import auxima` / `import erpnext` / dynamic equivalents, (c) installs a blocking meta_path finder and re-imports — any sidecar code that tries to dynamically resolve `frappe` fails the test.
- `.github/workflows/ci.yml` — runs `ruff check` + `pytest` on every push and PR.

## NOT in this scaffold

The full AI surface lands per `Docs/Planning/Vibe-Coding-Plan-v1.md` packs:

- LiteLLM router integration (P0-D / P1-10).
- CrewAI + LangGraph integration (P1-10).
- The 10 AI endpoints (intake-extract, recommend, KYC-screen, etc.).
- Ollama client wiring.
- The HMAC-over-body-+-timestamp-+-nonce auth upgrade (GAP-16).
- Outbound calls to Frappe to write Auxima Activity rows.
- Prometheus metrics, OpenTelemetry tracing.

## Run it

```bash
# Set the shared secret (must match the Frappe-side setting).
export AUXIMA_SIDECAR_SHARED_SECRET="$(openssl rand -hex 32)"

# Install in dev mode.
python -m pip install -e .[test]

# Start the server.
uvicorn auxima_ai.main:app --host 0.0.0.0 --port 8088 --reload

# Sanity:
curl http://localhost:8088/healthz
curl -H "X-Auxima-Sidecar-Token: $AUXIMA_SIDECAR_SHARED_SECRET" http://localhost:8088/v1/whoami
```

## Run tests

```bash
pytest -v
```

All three isolation tests MUST pass. They are the firewall.

## License

MIT — see [LICENSE](LICENSE).

## Trademark

"Auxima" and "Auxima Insure" are trademarks of Auxilium Tech. This repository is permissively licensed (MIT) for the code only; trademark rights are reserved separately. See CLAUDE.md §3 Rule 3.
