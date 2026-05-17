"""Outbound + inbound webhook utilities.

Per `BMS/Docs/Planning/slices/S-34-outbound-webhooks.md`:
  - signer  — HMAC-SHA256 over timestamped body (Stripe-style v1 scheme).
  - delivery / retry policy — future modules.

The signer is a pure-stdlib utility; no FastAPI / Frappe deps.
"""
