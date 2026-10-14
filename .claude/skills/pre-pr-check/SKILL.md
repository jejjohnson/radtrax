---
name: pre-pr-check
description: Run the repository's full pre-PR verification — ruff lint and format on the whole repo, ty on the package and scripts, the whole test suite with doctests and the coverage gate, the capability index, the lockfile and the assembled docs build. Use before committing, pushing or opening a pull request, and after any change to public API, dependencies, docstrings or docs.
---

# Pre-PR check

Run from the repo root and fix what fails before committing. Report results
honestly: what ran, what passed, what was skipped and why.

## Always

```bash
uv run --group lint ruff check .          # entire repo: tests/, scripts/, notebook cells
uv run --group lint ruff format --check .
make typecheck                            # ty on src/mypackage and scripts (what CI runs)
make test-cov                             # every test + doctests, with the 90 % coverage gate (what CI runs)
```

`make test` is the same suite without coverage, for quick iterations. For a
part of the suite, pass `--no-cov` (a subset never reaches the gate):
`uv run pytest --no-cov tests/<file>.py`.

## When the public API changed

```bash
make capabilities
uv run pytest --no-cov tests/test_capabilities.py tests/test_public_api.py
```

…plus the API page (`docs/api/<module>.md`, a `nav` entry in `mkdocs.yml`
for a new module) and the row in `docs/api/index.md`. `__all__` stays
sorted; a new module joins `SUBMODULES` in `tests/test_public_api.py`.

## When dependencies changed

`uv lock` and commit `uv.lock`. A runtime dependency needs discussion
(`AGENTS.md`, "Boundaries"); an optional one sits behind an extra and an
in-function import that names it.

## When docs or docstrings changed

- Every `Examples:` block runs under the test suite (`--doctest-modules`);
  check the expected output is what the code prints.
- `make docs-api` (MkDocs, strict) for docstrings and `docs/api/`;
  `make docs-check` (strict, both halves) and `make docs` (the assembled
  site with link checking, what `docs.yml` runs) for prose and notebooks —
  both need the mystmd CLI (`npm install -g mystmd`); if the theme download
  is blocked, see `docs/README.md`.

## Before pushing

- `git status` shows no stray files (`.plans/`, notebook `.py` drafts,
  `coverage.xml`, `site/`, `public/` are ignored; check nothing else is).
- `make precommit` runs the git hooks (whitespace, end-of-file, YAML, large
  files, ruff); fix what it reports in the files you touched.
- Conventional Commits title with a lowercase subject; `!` and a
  `BREAKING CHANGE:` footer for a breaking change.
- Push only to your feature branch, only when asked.
