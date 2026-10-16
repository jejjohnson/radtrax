# Code Review Agent Instructions

Standing instructions for **all** agents performing code reviews on this
repository. Read "Reuse before you write", "The contracts" and "What enforces
them" in [`AGENTS.md`](AGENTS.md) first; this file is the checklist and the
report format. Most defects worth finding are a contract broken in a way the
tests cannot see (a JAX array in a static field, a NaN gradient at a valid
input, a Jacobian of a batch, a hand-written derivative with no autodiff
check, an import that points up the package graph) or a re-implementation
of something the stack already has, not style.

---

## How to Obtain the Diff

Use the following command to get the diff for review:

```bash
BASE_BRANCH="$(git rev-parse --verify main >/dev/null 2>&1 && echo main || echo master)"
git --no-pager diff --no-prefix --unified=100000 --minimal $(git merge-base --fork-point "$BASE_BRANCH")...HEAD
```

If that fails (e.g. detached HEAD, shallow clone), fall back to:

```bash
git --no-pager diff --no-prefix --unified=100000 --minimal "$BASE_BRANCH"...HEAD
```

### Reading the diff

| Prefix | Meaning |
|--------|---------|
| `+` | Added line |
| `-` | Removed line |
| ` ` (space) | Unchanged context |
| `@@` | Hunk header |

---

## Review Checklist

Skip anything ruff, ty or the tests already enforce (formatting, import
order, `__all__` ordering, doctest output, the coverage gate); review what
they cannot see.

### 1. Reuse and boundaries

- Every function, class or module the diff **adds** has been checked against
  [`docs/api/capabilities.md`](docs/api/capabilities.md) and the private
  helpers in "Reuse before you write" in `AGENTS.md`. A re-implemented
  validator, statistic, filter, padding rule, buffer or transform is a
  **High** finding, with the existing name to use.
- No new runtime dependency without discussion; an optional one sits behind
  an extra and an in-function import that names the extra. `uv.lock` is
  updated with any dependency change.
- Imports keep pointing one way (the module table in `AGENTS.md`); the
  library never imports the CLI.
- A rename or removal of a public name keeps the old name working with a
  `DeprecationWarning` for at least one release.

### 2. The contracts

- **Boundaries**: no import of `plumax`, lazily or not; nothing from
  geoops or georeader outside `radtrax.geo` and the `[geo]` extra; no
  `geoops._src` import anywhere; an optional dependency is imported inside
  the function and a missing extra names itself in the `ImportError`.
- **Forward-model components** are `eqx.Module`s; arrays are leaves, never
  `eqx.field(static=True)`; abstract bases use `eqx.AbstractVar` and
  `abc.abstractmethod`, concrete classes plain fields.
- **Derivatives** come from `jax.jacrev` / `jax.jacfwd`; a hand Jacobian is
  only a test oracle; every public differentiable function has a
  finite-difference gradient test.
- **Batches**: `eqx.filter_vmap` outside `jax.jacrev`, never a Jacobian of a
  batch; no Python loop over pixels.
- **Numerics**: float64 for spectroscopy; a branch that is 0/0 somewhere
  uses the double-where pattern; units stated (cm⁻¹, molecules/cm², Pa).
- **Exceptions** derive from `RadtraxError` and the closest stdlib
  exception; errors are chained (`raise … from err`).

### 3. Public API and documentation

- New names: in the module's `__all__` and the top-level `__all__`; a new
  module in `SUBMODULES` (`tests/test_public_api.py`), with its
  `docs/api/<module>.md` page, a `nav` entry in `mkdocs.yml` and a row in
  `docs/api/index.md`; `docs/api/capabilities.md` regenerated
  (`make capabilities`).
- Docstrings: Google style; the formula in `$…$`; `Args:`, `Returns:`,
  `Raises:` matching the code; an executable `Examples:` block whose output
  is what the code prints (print rounded values, not 17-digit floats).
- Prose pages: MyST directives, not MkDocs-Material syntax; API links by the
  top-level name (`xref:api#radtrax.<Name>`); a new page or notebook in
  the `docs/myst.yml` toc with a basename unique across `docs/`.
- Notebooks committed executed, with no error outputs, and every number
  quoted in the prose matching the printed output.

### 4. Tests

- Each behaviour is checked against a closed form, a hand computation, a
  reference code (HAPI, the ported slab) or a published value, with the
  tolerance and its reason in the test; edge cases (zero column, μ = μ₀,
  grid edges of a table, invalid arguments) have their own tests.
- New code is covered (the 90 % gate is global, so a well-covered package
  can hide an untested module: look at the diff, not the total).
- Randomness, if any, is seeded; a statistical assertion is bounded by the
  statistic's own sampling distribution, with the bound's source in a
  comment.
- Tier markers right: unmarked under ~1 s, `slow` above, `integration` for
  end-to-end workflows and optional dependencies.

### 5. Modern Python and dependencies

- `from __future__ import annotations`; type hints on every public
  function; `X | None`, built-in generics, PEP 695 `type` aliases.
- `dataclasses` with `slots=True` for data containers (`frozen=True` when
  immutable); `Enum` / `StrEnum` for fixed sets of options; `pathlib`;
  f-strings; context managers for resources.
- Specific exceptions, chained with `raise … from …`; guard clauses over
  deep nesting.
- A dependency, when one is justified, follows the preferences in
  `.github/instructions/python-coding.instructions.md`.

### 6. Security

- No secrets or credentials in code, tests, notebooks or docs.
- `subprocess` without `shell=True`; file paths taken from users are not
  joined into places they could escape.

---

## Package-Specific Checks

Every example here runs (float64 on).

### Arrays are leaves, not static fields

```python
import equinox as eqx
import jax
import jax.numpy as jnp


# ❌ Equinox warns "A JAX array is being set as static!", and the module
#    stops hashing under filter_jit
class BadSRF(eqx.Module):
    weights: jax.Array = eqx.field(static=True)


# ✅ A leaf; choose what is not differentiated with eqx.partition
class SRF(eqx.Module):
    weights: jax.Array  # (B, G)

    def __call__(self, radiance):  # (G,) → (B,)
        return self.weights @ radiance
```

### No NaN gradient at a valid input

```python
# M(μ) = (e^{−τ/μ₀} − e^{−τ/μ}) / (1/μ − 1/μ₀), with limit τ e^{−τ/μ₀} at μ = μ₀


# ❌ jnp.where picks the right value, but the branch not taken is still
#    differentiated: jax.grad(multiplier_naive)(0.5, 0.5) is nan
def multiplier_naive(mu, mu0, tau=1.0):
    general = (jnp.exp(-tau / mu0) - jnp.exp(-tau / mu)) / (1 / mu - 1 / mu0)
    return jnp.where(jnp.isclose(mu, mu0), tau * jnp.exp(-tau / mu0), general)


# ✅ Double where: a safe input in the branch not taken
def multiplier_safe(mu, mu0, tau=1.0):
    close = jnp.isclose(mu, mu0)
    mu_s = jnp.where(close, mu0 + 1.0, mu)
    general = (jnp.exp(-tau / mu0) - jnp.exp(-tau / mu_s)) / (1 / mu_s - 1 / mu0)
    return jnp.where(close, tau * jnp.exp(-tau / mu0), general)
```

### `vmap` outside `jacrev`

```python
def y_of(theta):  # (3,) → (7,)
    return jnp.exp(-theta[0] * jnp.linspace(0, 1, 7)) * theta[1] + theta[2]


thetas = jnp.ones((256, 3))

# ❌ The Jacobian of a batch is (256, 7, 256, 3), almost all zeros
# K = jax.jacrev(jax.vmap(y_of))(thetas)

# ✅ A per-pixel Jacobian, mapped over pixels: (256, 3) → (256, 7, 3)
K = eqx.filter_vmap(jax.jacrev(y_of))(thetas)
```

### A hand Jacobian is an oracle, not the implementation

Ported code (plumax's slab forward model) returns derivatives written by
hand. Keep them, but test them against `jax.jacrev`, and write new code so
that autodiff is the only derivative.

---

## Output Format

Format each review using this structure:

````
# Code Review for ${feature_description}

Overview of the changes, including the purpose, context, and files involved.

## Suggestions

### ${emoji} ${Summary of suggestion with necessary context}

* **Priority**: ${priority_emoji} ${priority_label}
* **File**: `${relative/path/to/file.py}`
* **Line(s)**: ${line_numbers}
* **Details**: Explanation of the issue and why it matters
* **Current Code**:
  ```python
  # problematic code
  ```
* **Suggested Change**:
  ```python
  # improved code with explanation
  ```

### (additional suggestions…)

## Summary

Brief summary of overall code quality and key action items.
````

---

## Priority Levels

| Emoji | Level | Use when |
|-------|-------|----------|
| 🔥 | **Critical** | Bugs, security issues, or code that will fail |
| ⚠️ | **High** | Significant issues affecting maintainability or correctness |
| 🟡 | **Medium** | Improvements for code quality or consistency |
| 🟢 | **Low** | Minor polish or optional enhancements |

## Suggestion Type Emojis

Prefix each suggestion title with a type indicator:

| Emoji | Type |
|-------|------|
| 🐛 | Bug or potential bug |
| 🔒 | Security concern |
| 🔧 | Change request (must fix) |
| ♻️ | Refactor suggestion |
| 📝 | Documentation improvement |
| 🎨 | Style / formatting issue |
| ⚡ | Performance consideration |
| 🧪 | Testing suggestion |
| ❓ | Question or clarification needed |
| ⛏️ | Nitpick (very minor) |
| 💭 | Design consideration |
| 👍 | Positive feedback (highlight good patterns) |
| 🌱 | Future consideration (not blocking) |

---

## Review Tone

- Be **constructive** and **specific**
- **Acknowledge** good patterns and decisions (use 👍 liberally)
- Explain the *why* behind every suggestion
- Offer **concrete alternatives**, not just criticism
- Recognize that context matters — ask clarifying questions when needed
- Keep feedback **actionable**: every suggestion should have a clear next step
