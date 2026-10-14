---
name: radtrax-review
description: Review a change or pull request in radtrax against its own rules — CODE_REVIEW.md, the contracts in AGENTS.md, reuse of what the stack already provides, and numerical correctness (pytrees, gradients, precision, batching, units, package boundaries). Use when asked to review a diff, branch or PR in this repo.
---

# Review a change

1. **Get the diff** as `CODE_REVIEW.md` describes, or from the PR. Note
   which modules, tests and docs it touches.
2. **Reuse** — run the `reuse-reviewer` subagent on the diff: a re-written
   solver, covariance, matched filter or SRF integration is the main way the
   stack drifts from itself.
3. **Numerics** — run the `numerics-reviewer` subagent in parallel with
   step 2: static arrays, NaN gradients, precision, batching, units,
   tolerances, package boundaries.
4. **Contracts** — for each module touched, the rules under "The contracts"
   in `AGENTS.md`: Equinox modules with arrays as leaves, protocols as
   abstract modules, derivatives from autodiff with a gradient test,
   `filter_vmap` outside `jacrev`, float64 for spectroscopy, no `plumax`
   import, geo code only in `radtrax.geo`, exceptions deriving from
   `RadtraxError` and their stdlib base.
5. **Checklist** — the rest of `CODE_REVIEW.md` (public API, API page and
   `docs/api/index.md` row, capability index, MyST prose, tests and tiers),
   skipping what ruff, ty or the tests already enforce.
6. **Verify claims** — for anything you flag as a bug, run it
   (`uv run python -c "..."`) and report what it printed.

Report in the format `CODE_REVIEW.md` gives (overview, suggestions with
priority, file, lines and a concrete change, summary).
