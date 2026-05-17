"""Pre-call token estimation for cost / quota gating.

Real tokenisation requires the model's tokeniser (tiktoken for OpenAI,
sentencepiece for Llama, etc) which we deliberately do NOT pull in —
each provider's tokeniser is a 1MB+ dependency and most calls only
need an approximate count to gate the per-tenant ceiling and rate
limit. The estimator in :mod:`.estimator` returns a conservative
upper-bound estimate (errs toward over-counting so ceilings hold)
based on Unicode-script-aware character heuristics:

  - English / Latin script: ~4 characters per token (GPT family).
  - Arabic / CJK / RTL scripts: ~2 characters per token.
  - Mixed scripts: blends the two proportionally.

The output is an ``int`` ceiling — never a fractional or negative
count — so it composes cleanly with the
:func:`auxima_ai.cost.pricing.cost_for` and the
:class:`auxima_ai.policy.enforcer.PolicyEnforcer` admit path.
"""
