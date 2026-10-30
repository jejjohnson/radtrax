# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Reuse first.** Search `docs/api/capabilities.md` (every public name in
  the package) before writing a helper; the "Reuse before you write" table in
  `AGENTS.md` maps the usual hand-rolled code to what already exists.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - building: `add-guide-page`, `add-notebook`;
  - extension points, one per protocol: `add-opacity-provider`,
    `add-solver`, `add-surface`, `add-instrument`, `add-noise-model`;
  - shipping: `pre-pr-check`, `radtrax-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `radtrax-review`; run them on any diff that adds code, before
  committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/api/capabilities.md` (gaussx, lineax and optimistix included) or
    elsewhere in the stack?
  - `numerics-reviewer`: static arrays in Equinox modules, NaN gradients,
    float32 in spectroscopy, Jacobians of batches, hand derivatives without
    an autodiff check, units, tolerances, package boundaries.
- **Partial test runs** need `--no-cov` (`addopts` turns coverage on and its
  90 % gate fails any subset).
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
