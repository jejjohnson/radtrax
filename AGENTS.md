# AGENTS.md

Standing instructions for **every** coding agent working in this repository
(Claude Code, Copilot, Codex, Gemini, …). This is the single source of truth:
`CLAUDE.md` and `.github/copilot-instructions.md` point here.

## What this repo is

`radtrax` (formerly rtmx) is the GeoStack's differentiable radiative transfer
package for methane and multi-gas remote sensing: a forward model
$y = F(x)$ from an atmospheric state to what a sensor measures, written in
JAX and Equinox so that every Jacobian a retrieval needs,
$K = \partial F / \partial x$, comes from automatic differentiation. The
design is *radtrax · 3* in the GeoStack docs
(`docs/geomodels/radtrax/03_design.md` in jejjohnson/geostack); the moves
into it are *package boundaries* (`docs/geomodels/package_boundaries.md`).

Other packages build on it (plumax first), so **the package is a library of
primitives**: new code composes what is here, and anything new lands where
the next person will find and reuse it.

It is one package, `src/radtrax/`. The public API is `radtrax.__all__`,
re-exported from `src/radtrax/__init__.py`. It holds v0, the layered forward
model, in the layout below (issues #1–#4 are the wave epics of v0, #100 is
v1's), and `tests/test_public_api.py` fails if the layout drifts:

| Module | Owns | Lands in |
|---|---|---|
| `exceptions.py` | `RadtraxError`, the base of every error, and one subclass per kind of failure (`ShapeError`, `GridError`, `ParameterError`, …) | now |
| `forward_model.py` | `ForwardModel`: state → opacity → solver → instrument, per pixel (#37); its matched-filter target, `unit_absorption_target`, lives in `retrieval/matched_filter/target.py` | now |
| `atmosphere/` | `state`: `AtmosphericState`; `hydrostatics`: `layers_from_levels`; `afgl`: the AFGL US Standard profile (#32); `gas`: ideal-gas constants, `number_density`; `geometry`: `air_mass_factor` (the one home of each, #27) and `ViewingGeometry`, the JAX geometry surfaces and solvers take (#35), with its spherical-shell `layer_air_mass` (#104) | now |
| `spectroscopy/` | `opacity`: the `OpacityProvider` protocol and `TableOpacity`, per-layer τ (L, G) bilinear in (ln p, T) (#33); `tables`: single-slab cross-section interpolation, Beer–Lambert transmittance, the plume ratio spectrum; `table_cache`: v0's tables in the user cache, `load_opacity`; `synthetic`: the synthetic methane spectra (radtrax · 3's Voigt table, radtrax · 1/2's Sentinel-2 spectrum) for docs, notebooks and tests (#115); `hitran/`: offline table building through HAPI (`[hitran]`, `python -m radtrax.spectroscopy.hitran GAS ...`) | ported (#20); `opacity`, `table_cache` (#33, #34) |
| `optics/` | `properties`: `OpticalProperties`, τ (L, G) on its grid between opacity and solver (#35); Rayleigh, aerosols, delta-M from v3 | now |
| `solvers/` | `beer_lambert`: the `RTSolver` protocol and `BeerLambert`, radiance (G,) from optics, surface and geometry (#35); scattering solvers from v3 | now |
| `surface/` | `lambertian`: the `Surface` protocol and `Lambertian` (#35); `polynomial`: `PolynomialAlbedo`, a Legendre-series albedo in ν (#106); `ross_li`: `RossLi` and its kernels `ross_thick`, `li_sparse_reciprocal` (#105) | now |
| `instrument/` | `spectral`: the `Instrument` protocol and `BandInstrument`, JAX band integration with a spectral shift (#36), and `GaussianInstrument`, Gaussian bands shifted analytically in cm⁻¹ or nm (#96); `srf`: the one SRF band integration (exact Gaussian FWHM, #28), whose matrix `BandInstrument.from_srf` takes; `psf`: JAX point-spread and ground-sampling operators; `noise`: the `NoiseModel` protocol, `DiagonalNoise` and `SNRNoise` as lineax diagonal operators (#108) | ported (#21); `spectral` (#36); `noise` (#108) |
| `slab/` | plumax's single-slab Beer–Lambert models (exact, Maclaurin, Taylor), the nB table (relative to a background column, #36), observation config; v0's regression oracle | ported (#21) |
| `retrieval/` | `var3d/`: 3D-Var (observation operator, B, control transforms, three solvers, diagnostics); `matched_filter/`: the one core (backgrounds, clusters, streaming, targets from the observation operator and from `ForwardModel`, xarray I/O); `mbmp`: the Sentinel-2 MBMP signal (#30) | ported (#22, #23) |
| `geo/` | `sentinel2_srf`: Sentinel-2 SRFs from georeader's bundled table (#28), and `sentinel2_band_responses`, its raw B11 and B12 responses on a grid (#115); `matched_filter`: pipekit operators on GeoTensors around the one core (#29); `mbmp`: the MBMP operator, geotoolz's former SBMP (#30) (`[geo]`) | ported (#28–#30) |

### Boundaries

These are the package-boundaries rules that apply here; break one and the
GeoStack package graph gets a cycle.

- **radtrax never imports plumax**, not even lazily. Radiative transfer does
  not know about plumes; plumax's plume-to-radiance coupling imports radtrax.
  An import-linter contract (`[tool.importlinter]` in `pyproject.toml`)
  checks it.
- **Layers point down: GeoModels → GeoML → GeoNumerics.** radtrax is a
  GeoModels package. It may depend on GeoNumerics (gaussx, lineax,
  optimistix) and GeoStack (geotoolz, georeader, through extras), and on
  GeoML: vardax holds the variational machinery radtrax customises (#109,
  through `radtrax[var]`), and geonnax its Legendre basis (#106). Nothing in GeoNumerics or GeoML may import
  radtrax. No other GeoModels package sits below it either:
  finitevolx, spectraldiffx and somax are not dependencies, since a layer
  grid is three arrays and a band integration is a matrix product.
- **Geo carriers arrive through extras.** `radtrax[geo]` adds georeader
  (with the `openpyxl`, `tqdm` and `typing-extensions` it needs but does not
  declare), geotoolz and pipekit; the base install stays JAX-only. The
  dev group installs `radtrax[geo]`, so CI runs the geo tests. Code in `geo/` imports only
  public geotoolz names, never `geotoolz._src`.
- **Optional dependencies are lazy.** Anything behind an extra (`hitran`,
  `geo`, `bayes`) is imported inside the function that needs it,
  and a missing extra raises an `ImportError` that names the extra.

Whatever the package, dependencies point one way. When other projects build
on it, list them here and keep their imports working: a rename or removal of
a public name goes through a deprecation (the old name keeps working with a
`DeprecationWarning` that names the replacement for at least one release).

## What it is built on

Each foundation brings a rule. Breaking one runs fine on one pixel and fails
on a swath, under `jit` or in CI.

| Foundation | The package uses it for | The rule it brings |
|---|---|---|
| **JAX** | every forward model and its derivatives | Pure functions of arrays; no Python side effects or data-dependent Python control flow inside traced code. Float64 for spectroscopy (`jax_enable_x64`; `tests/conftest.py` sets it). |
| **Equinox** | modules for state (tables, SRFs, profiles, noise models) | Arrays are leaves, never `eqx.field(static=True)` (Equinox warns and hashing breaks under `filter_jit`); choose what is not differentiated with `eqx.partition` or `jax.lax.stop_gradient`. |
| **jaxtyping** | shape annotations on every public array argument | Name dimensions once and reuse them (the vocabulary is under "Shapes" below); the test suite checks every shape at run time. |
| **lineax**, **optimistix** | linear solves, fixed points, Gauss–Newton and Levenberg–Marquardt with `ImplicitAdjoint` | Use them rather than hand-written iterations; JAXopt is unmaintained. |
| **gaussx** | structured covariances (`LowRankUpdate`, Woodbury), block-tridiagonal solves | A structured operator, never a dense inverse; `gx.solve` dispatches on structure. |
| **NumPy**, **xarray** | offline table building and I/O | Off the gradient path only; nothing differentiable is NumPy. |

## Reuse before you write

Before writing a helper, a solver or a covariance, find out whether it
exists:

1. **Search the capability index.**
   [`docs/api/capabilities.md`](docs/api/capabilities.md) lists every
   public name once, at its public home, with a one-line summary. It is
   generated (`make capabilities`) and checked by
   `tests/test_capabilities.py`, so it is current.
2. **Search the stack.** Much of radtrax is ported, not written: plumax's
   `radtran`, `hapi_lut`, `assimilation` and `matched_filter`, and geotoolz's
   `matched_filter` and `plume.SBMP`. Port with tests, do not rewrite.
3. **If it is missing, add it to the module that owns the concept**, at the
   lowest layer every caller can reach, not inline in the one caller you
   have today.
4. **One object, one name.** A name means one thing across the package and
   its submodules (`tests/test_capabilities.py`); a deliberate exception is
   listed with its reason in `ALLOWED_SHARED_NAMES` in
   `scripts/capabilities.py`.

| You are about to write… | Use instead |
|---|---|
| `jnp.linalg.inv(S) @ v`, a Woodbury identity by hand | `gaussx` operators and `gx.solve` |
| A Gauss–Newton or Levenberg–Marquardt loop | `optimistix.GaussNewton` / `LevenbergMarquardt` (`ImplicitAdjoint`) |
| A fixed-point iteration | `optimistix.FixedPointIteration` |
| L-BFGS for a MAP retrieval | `optax.lbfgs` (`[bayes]`) |
| A tridiagonal or block-tridiagonal solve | `lineax` (scalar bands) or `gaussx.BlockTriDiag` (blocks) |
| A Jacobian written by hand | `jax.jacrev` / `jax.jacfwd`; keep a hand Jacobian only as a test oracle |
| A loop over pixels | `eqx.filter_vmap` outside `jax.jacrev` |

## The contracts

Everything composes because it keeps these contracts.

### 1. Forward-model components

- **Modules carry state; functions carry transforms.** A table, a line
  list, an SRF or a noise covariance is an `eqx.Module`; a Voigt profile, a
  Chapman factor or a delta-M scaling is a function.
- **Protocols at the seams.** `OpacityProvider`, `RTSolver`, `Surface`,
  `Instrument` and `NoiseModel` are abstract Equinox modules
  (`eqx.AbstractVar` fields and `abc.abstractmethod` methods on the base,
  concrete fields on the subclass), so `filter_jit` compiles each
  combination.
- **`vmap` outside `jacrev`.** A swath is a `filter_vmap` over pixels of a
  per-pixel Jacobian, never a Jacobian of a batch.
- **Gradients are tested.** Every public differentiable function has a test
  comparing `jax.jacrev` against central finite differences:
  `from _gradcheck import assert_gradients_match` in a test module
  (`tests/_gradcheck.py`; rtol 1e-6 on a float64 reduced state by default).

### 2. Numerics

- **Float64** for line-by-line and table opacity; float32 moves line
  centres by a visible fraction of a Doppler width.
- **No NaN gradients from safe values.** A branch that is 0/0 at a point
  (e.g. $\mu = \mu_0$) uses the double-where pattern: a safe denominator in
  the branch not taken, or `jnp.where` still propagates the NaN.
- **Units in names or docstrings**: wavenumbers in cm⁻¹, columns in
  molecules/cm² or kg/m², pressures in Pa unless the name says otherwise.

### 3. Exceptions

- Every error derives from `RadtraxError` **and** the closest
  standard-library exception, so `except ValueError` keeps working. A new
  error goes into `exceptions.py`, is exported, and is tested in
  `tests/test_exceptions.py`.
- Raise an existing class from `exceptions.py` rather than a built-in;
  `tests/test_exceptions.py` scans `src/radtrax` and fails on any
  `raise ValueError` (or other built-in) outside its allow-list. Checks
  under `jit` (`eqx.error_if`) raise a `RuntimeError` at run time
  (`equinox.EquinoxRuntimeError` under `eqx.filter_jit`,
  `jax.errors.JaxRuntimeError` under `jax.jit`); say so in the docstring.

### 4. Shapes

- Every public function and method annotates its array arguments and
  returns with a jaxtyping shape (`Float[Array, " B"]`), callables
  included (`Callable[[Float[Array, " N"]], Float[Array, " B"]]`).
  `tests/test_annotations.py` checks it, and jaxtyping's pytest hook
  (`--jaxtyping-packages=radtrax,beartype.beartype` in `addopts`) checks
  the shapes themselves on every test run.
- Quote every dimension string, with a leading space when it would
  otherwise parse as Python (`" B"`, `" B *shape"`): ruff's UP037 unquotes
  such a string into an expression (`B * shape`). Don't write `...` or `"..."`; use a named variadic such as
  `"*shape"` (element-wise functions take and return
  `Float[ArrayLike, "*shape"]`) or `"*batch"`.
- A name with a leading `_` (`"*_batch B"`) is not checked across
  arguments; use it for the items of an iterable whose shapes differ.
- When a shape recurs in a module, give it an alias
  (`Spectrum = Float[np.ndarray, " B"]`).
- The names, reused everywhere:

  | Name | Axis |
  |---|---|
  | `L` | atmospheric layers (`L1` = `L + 1` levels) |
  | `G` | wavenumbers of the model grid |
  | `Ghr` | the high-resolution grid an SRF integrates over |
  | `B` | instrument bands |
  | `P`, `T` | pressures and temperatures of a cross-section table |
  | `H`, `W` | image rows and columns (`h`, `w` after ground sampling) |
  | `N` | pixels, flattened |
  | `M` | observations or residuals, flattened |
  | `S` | samples (draws, ensemble members) |
  | `time` | time steps or scenes |
  | `K` | clusters |
  | `r` | rank of a low-rank factor |
  | `k` | side of a convolution kernel |

### The public API

- Export a public name from its module's `__all__` and from
  `src/radtrax/__init__.py` (import and `__all__`); both stay sorted (ruff's
  `RUF022` and `test_all_is_sorted_and_unique` agree on the order).
- A new top-level module declares a sorted `__all__`, joins `SUBMODULES` in
  `tests/test_public_api.py`, and gets its own page in `docs/api/` (a
  `::: radtrax.<module>` block), a `nav` entry in `mkdocs.yml` and a row in
  `docs/api/index.md`.
- Run `make capabilities` after any public API change.
- Link prose to the API by the **top-level** name, see "Documentation".

## What enforces them

Most rules here are tests, so CI tells you when one breaks. Read a test's
docstring before changing what it checks.

| Test | Enforces |
|---|---|
| `tests/test_public_api.py` | `__all__` sorted, unique and importable; `SUBMODULES` is exactly the set of modules and `SUBPACKAGES` the set of subpackages; each declares a docstring and a sorted `__all__`; the `py.typed` marker ships; `__version__` is dotted |
| `lint-imports` (`make lint`, `lint.yml`) | radtrax never imports plumax |
| `scripts/check_base_install.py` (`base-install.yml`) | Gate P5: a fresh `uv pip install .` holds no georeader, geotoolz or plumax, and every radtrax module imports without the extras |
| `tests/test_capabilities.py` | `docs/api/capabilities.md` is current; no name is bound to two different objects |
| `tests/conftest.py` | Float64 is on for the whole suite |
| `--doctest-modules` on `src/radtrax` (`addopts`) | Every docstring `Examples:` block runs and prints what it says |
| `tests/test_exceptions.py` | The hierarchy: every error is a `RadtraxError` |
| `tests/test_gradcheck.py` | The gradient-check harness rejects a 1 % wrong derivative, including on an entry 1e-4 of the Jacobian's scale, and accepts correct ones |
| `tests/test_build_docs.py` | The two-tool docs pipeline: inventory parsing, anchor repair, link checking, the inventory port and the absolute nav URL in `docs/myst.yml`, the files copied to the site root (`ROOT_FILES`) |
| `[tool.coverage.report] fail_under = 90` | Coverage of `src/radtrax` in every coverage-enabled run (CI, `make test-cov`) |
| `strict_markers = true` | Only the registered tier markers (`slow`, `integration`) exist |
| ruff (`select = E, F, I, UP, B, SIM, RUF`), ty | Lint (the whole repo), format, types on `src/radtrax` and `scripts` |

## Recipes

Step-by-step recipes for the common jobs live as plain Markdown in
`.claude/skills/<name>/SKILL.md` (Claude Code loads them automatically; any
agent can read and follow them), and two read-only review checklists in
`.claude/agents/`:

| Job | Recipe |
|---|---|
| Add or update a guide page (MyST prose) | `add-guide-page` |
| Add or update an example notebook | `add-notebook` |
| Verify before a PR | `pre-pr-check` |
| Review a change | `radtrax-review` (+ the read-only `.claude/agents/reuse-reviewer.md` and `numerics-reviewer.md`) |
| Write a squash commit message | `squash-commit` |
| Open or link GitHub issues | `create-gh-issue`, `link-gh-issues` (templates in `.github/ISSUE_TEMPLATE/`; `make gh-labels`, `gh-sub`, `gh-block`, `gh-show`) |

Recipes for radtrax's extension points (a new opacity provider, solver,
surface, instrument or noise model) are tracked in #118; until then, follow
the existing implementation behind each protocol.

Downstream users get the package's guidance through `docs/llms.txt`
(copied to the site root by `ROOT_FILES` in `scripts/build_docs.py`); update
it when the public API or the headline usage changes. A Claude Code plugin
for downstream packages follows once the public API exists.

## Working in the repo

Always run Python tools through `uv run` (never the system Python); `git`,
`ls` and other non-Python commands need no `uv run`.

```bash
make install              # uv sync --all-groups + pre-commit hooks
make test                 # every test + doctests, no coverage (fast)
make test-cov             # the same with coverage and the 90 % gate (what CI runs)
make lint                 # ruff check .   (entire repo)
make format               # ruff format . && ruff check --fix .
make typecheck            # ty check src/radtrax scripts (what CI runs)
make capabilities         # regenerate docs/api/capabilities.md
make docs                 # build both docs halves into public/ and check every link
make docs-check           # strict validation of both halves, no themed render
make docs-api             # MkDocs API reference only (mkdocs build --strict)
make docs-serve           # build, then serve public/ at :8000
make precommit            # every pre-commit hook on every file
make build                # wheel + sdist into dist/
```

`addopts` turns coverage on, and a partial run never reaches the 90 % gate,
so pass `--no-cov` when running part of the suite:

```bash
uv run pytest --no-cov tests/test_exceptions.py -v
```

### Test tiers

The markers are registered in `pyproject.toml` (`strict_markers = true`):

- **Unmarked (fast):** unit tests and doctests, each well under a second.
- **`@pytest.mark.slow`:** a test over ~1 s (a large sweep, a long fit). For
  a parametrised test, mark only the expensive cases
  (`pytest.param(..., marks=pytest.mark.slow)`).
- **`@pytest.mark.integration`:** end-to-end workflows, optional
  dependencies, external tools.

CI ("Tests", `ci.yml`) runs every tier with coverage on Python 3.12 and 3.13
(`uv run pytest -v`). Select the fast tier with
`uv run pytest -m "not slow and not integration"`; when the slow tiers grow,
give them a CI job of their own rather than dropping them.

### Tests that assert on random draws

radtrax draws no random numbers yet. When it does, randomness comes from a
`jax.random` key the caller passes; nothing seeds a global state.

- **Incidental randomness** (any draw would do): seed it
  (`jax.random.key(0)`, `numpy.random.default_rng(0)`) so a
  failure reproduces.
- **Sampling behaviour under test** (a Monte Carlo estimate, a statistical
  property): bound the statistic by its own sampling distribution (a number
  of standard errors, a test's p-value floor), not by a tolerance tuned on
  one draw, and say in a comment where the bound comes from.

### Before every commit

All of these must pass, from the repo root:

1. `make test`: zero failures (doctests included).
2. `uv run --group lint ruff check .`: the **entire** repo, which includes
   `tests/`, `scripts/` and the code cells of `docs/notebooks/*.ipynb`.
   Never lint a subdirectory: `ruff check src/radtrax/` misses errors CI
   catches.
3. `uv run --group lint ruff format --check .`
4. `make typecheck` (`ty check src/radtrax scripts`, as CI runs it).
5. After changing a public API: `make capabilities`, and the `docs/api`
   page.
6. After changing docs, docstrings, `mkdocs.yml` or `docs/myst.yml`:
   `make docs` (or at least `make docs-api` when only docstrings changed).
7. After changing a dependency: `uv lock`, and commit `uv.lock`.

## Coding principles

1. **Think before coding.** State assumptions; if a request has several
   readings, name them instead of picking one silently; if something is
   unclear, stop and ask; push back when a simpler approach exists.
2. **Simplicity first.** The minimum code that solves the problem: no
   speculative features, no single-use abstractions, no configurability
   nobody asked for, no error handling for impossible cases.
3. **Surgical changes.** Touch only what the task needs; match the existing
   style; don't refactor or add docstrings to code you didn't change;
   remove only what your change made unused, and mention (don't delete)
   unrelated dead code.
4. **Goal-driven.** Turn the task into a check (a failing test, a
   reproduced bug, a closed form to match) and loop until it passes; for
   multi-step work, state the plan with a check per step.

Also: Python 3.12+, `from __future__ import annotations` in every module,
type hints on every public function, `X | None` over `Optional`, specific
exceptions with `raise … from …`, Google-style docstrings, pure functions
with side effects isolated and explicit. When you suggest a change, propose
the code; when you fix a bug or add a feature, add the test that checks it.

## Git, commits and pull requests

- **Never** push to or merge into `main` unless explicitly told to ("push
  to main", "merge to main"). Work on a feature branch and commit locally;
  never run `git push` unless asked. "Merge the branch" means push the
  feature branch, not merge into `main`. Confirm before any action that
  affects a shared branch.
- Commit messages and PR titles follow
  [Conventional Commits](https://www.conventionalcommits.org/) with a
  lowercase subject (`feat(spectroscopy): add …`); CI validates PR titles
  (`conventional-commits.yml`). Types: `feat`, `fix`, `docs`, `style`,
  `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`. Breaking
  changes use `!` and a `BREAKING CHANGE:` footer.
- Releases are cut by release-please (`release-please-config.json`): it
  bumps the version and writes `CHANGELOG.md` from the `feat`, `fix`,
  `perf` and `revert` commits; don't bump versions by hand.
- **Never replace or remove an existing PR title or description.** An agent
  called for a small follow-up tends to write a fresh description scoped to
  its own work and silently discard the rest: read the existing description
  first and only append checklist items or update their status. Change the
  title only to fix a Conventional Commits violation.
- Code review follows [`CODE_REVIEW.md`](CODE_REVIEW.md); issues follow the
  label taxonomy and epic model in [`docs/contributing.md`](docs/contributing.md).

### Pull Request Review Comments

After fixing a review comment, resolve its thread. Don't resolve threads you
didn't address.

```bash
# 1. List the review threads and their IDs
gh api graphql -f query='
  query($owner: String!, $repo: String!, $pr: Int!) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        reviewThreads(first: 100) {
          nodes { id isResolved comments(first: 1) { nodes { body path line } } }
        }
      }
    }
  }' -f owner=OWNER -f repo=REPO -F pr=PR_NUMBER

# 2. Resolve an addressed thread
gh api graphql -f query='mutation($threadId: ID!) {
  resolveReviewThread(input: {threadId: $threadId}) { thread { isResolved } } }' \
  -f threadId=THREAD_ID
```

When the `gh` CLI is unavailable, use the GitHub MCP tools for the same
operations.

## Documentation

The docs are built by **two tools** and deployed as one site
(`docs/README.md` has the rationale); `docs.yml` builds them on every PR and
push to `main`, and `pages.yml` deploys `public/` to GitHub Pages from
`main`.

| Half | Tool | Source | Deployed at |
|---|---|---|---|
| Prose — home, guides, example notebooks | mystmd | `docs/*.md`, `docs/guide/`, `docs/notebooks/` (toc in `docs/myst.yml`) | `/` |
| API reference | MkDocs + mkdocstrings | `docs/api/` (`docs_dir` in `mkdocs.yml`) | `/reference/` |

`make docs` runs `scripts/build_docs.py`: it builds the API with
`mkdocs build --strict`, serves `site/` on port 8910 so mystmd can read its
`objects.inv` (mystmd loads inventories only over http), runs
`myst build --html --strict`, rewrites the localhost API URLs to
`/reference/`, repairs anchors lowercased by mystmd's `$`-expansion bug, and
fails if any internal link in the assembled `public/` site does not resolve.
mystmd is a Node CLI (`npm install -g mystmd`), not a uv dependency; if its
theme download is blocked, `docs/README.md` says how to build against a local
clone of the theme.

- **Prose is MyST Markdown**, not MkDocs-Material syntax: `:::{note}`,
  `:::{tab-set}`, `:::{dropdown}` directives, not `!!!` / `===` / `???`.
- **Link to the API** with the `xref:` protocol and the **top-level**
  exported name: ``[`RadtraxError`](xref:api#radtrax.RadtraxError)``, not
  `radtrax.exceptions.RadtraxError` (mystmd lowercases `$`-abbreviated anchors;
  see `docs/README.md`). A missing target fails `myst build --strict`.
- **URLs are flat**: a page's URL is its basename, with `_` turned into `-`
  (`guide/quickstart.md` → `/quickstart/`,
  `notebooks/smoothing_deep_dive.ipynb` → `/smoothing-deep-dive/`), so keep
  basenames unique across `docs/`, `docs/guide/` and `docs/notebooks/`;
  frontmatter `slug:` is ignored.
- **Site-nav URLs stay absolute** (`site.nav` in `docs/myst.yml`):
  the theme prefixes root-relative ones with `BASE_URL` a second time
  (`tests/test_build_docs.py` checks it).
- **API pages** (`docs/api/*.md`) are MkDocs pages: prose plus `:::` blocks,
  `!!!` admonitions, listed in the `nav` of `mkdocs.yml`; docstrings are
  Google style (`docstring_style: google`), maths through MathJax
  (`pymdownx.arithmatex`).
- **Notebooks** in `docs/notebooks/` are committed as executed `.ipynb`
  files, authored as jupytext percent `.py` and deleted after conversion;
  mystmd renders the stored outputs without re-executing, so each must be
  listed in the `docs/myst.yml` toc. Figures render inline with
  `plt.show()`: no `savefig`, no committed PNGs. `notebooks/` at the root
  holds scratch notebooks, outside the docs. Full standards:
  `.github/instructions/docs-examples.instructions.md`.

## Plans

Plans and scratch design notes go in `.plans/` (gitignored, never
committed); track work in GitHub issues. Draft a whole wave of issues from
`docs/templates/wave-backlog.md` (see `docs/contributing.md`).
