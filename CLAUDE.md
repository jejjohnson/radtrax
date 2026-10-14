# CLAUDE.md

The rules for every agent live in `AGENTS.md`; this file adds only what is
specific to Claude Code.

@AGENTS.md

## Claude Code specifics

- **Template markers.** A `<!-- template-example: … -->` (or
  `# template-example: …`) comment marks a passage about the bundled example
  package; after adoption, rewrite it from the new code (see the top of
  `AGENTS.md`).
- **Reuse first.** Search `docs/api/capabilities.md` (every public name in
  the package) before writing a helper; the "Reuse before you write" table in
  `AGENTS.md` maps the usual hand-rolled code to what already exists.
- **Skills** in `.claude/skills/` load on their own when a task matches
  their description (or run them as `/<name>`):
  - adopting the template: `adopt-template`;
  - building: `add-function`, `add-transform`, `add-cli-command`,
    `add-guide-page`, `add-notebook` (the first three are the example
    package's extension points: rewrite them on adoption);
  - shipping: `pre-pr-check`, `mypackage-review`, `squash-commit`;
  - GitHub housekeeping: `create-gh-issue`, `link-gh-issues`.
- **Subagents** (`.claude/agents/`), both read-only, both used by
  `mypackage-review`; run them on any diff that adds code, before
  committing:
  - `reuse-reviewer`: does the diff re-implement something in
    `docs/api/capabilities.md` or a shared private helper?
  - `correctness-reviewer`: validation at the boundary, mutation of inputs,
    the fit / apply lifecycle, numeric edge cases, error types and
    messages, doctest accuracy, test tolerances (a general reviewer tuned to
    the example: replace it with one for your domain).
- **Partial test runs** need `--no-cov` (`addopts` turns coverage on and its
  90 % gate fails any subset).
- **GitHub.** When the `gh` CLI is unavailable, use the GitHub MCP tools for
  the same operations (PRs, issues, review threads, check runs).
