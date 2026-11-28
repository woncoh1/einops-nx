# Correctness verification

## Verified result

**692 tests and 4 doctests passed**, with no failures and warnings treated as
errors. Formatting and a forced library compilation with warnings treated as
errors also passed.

The verification used Linux x86_64, Elixir/Mix 1.20.5 compiled for OTP 29,
Erlang/OTP 29.1.3, Nx 1.0.0, complex 1.0.0, and telemetry 1.4.3. The toolchain
matches `mise.toml`, and all dependency versions match `mix.lock`.
The suite explicitly selects `Nx.BinaryBackend` and `Nx.Defn.Evaluator`.
EXLA, Torchx, and other native compiler/backend combinations were not tested.

Python einops 0.8.2 and NumPy 2.3.5 generated the independent oracles. All four
fixture files reproduced byte-for-byte after formatting. The original 67-case
fixture file is unchanged.

In this verification environment, the Hex registry connection failed. The
three locked dependency archives were downloaded from Hex's CDN, their inner
and outer checksums were checked against `mix.lock`, and Mix reported each
dependency as `ok`. No dependency source or lockfile changes were needed.

## Run locally

From the repository root, fetch dependencies before formatting or testing:

```sh
mise exec -- mix deps.get
mise exec -- mix format --check-formatted
mise exec -- mix compile --force --warnings-as-errors
mise exec -- mix test --warnings-as-errors --seed 0
```

Python is not needed to run the Elixir suite: expected outputs and gradients
are committed as Elixir literals. Add `--trace` to the test command for
individual test names.

## Python value and shape coverage

| Operation | Python cases | Coverage |
|---|---:|---|
| `rearrange` | 34 original | Permutations, ordered merges, inferred splits, patchification, singletons, scalars, and zero/multiple-dimension ellipses |
| `repeat` | 8 original | Element versus sequence repetition, multiple nontrailing axes, splitting before repetition, scalar inputs, and ellipses |
| `reduce` with `sum`, `mean`, `min`, `max`, `prod` | 25 original | Five patterns per reducer: axis removal, permutation/merge, scalar outputs, ellipsis reduction, and local pooling |
| `reduce` with `any`, `all` | 10 supplemental | The same five patterns with all-zero, all-nonzero, and mixed signed inputs; both true and false outputs |
| `parse_shape` | 24 supplemental | Exact string-keyed maps, scalars, ignored axes, singleton sizes, and leading/middle/trailing ellipses |

The 77 tensor-output cases each run in six configurations: eager, JIT, and
ahead-of-time compilation from templates, each with ordinary tensors and with
two vectorized axes. This gives **462 forward reference comparisons**.
Assertions check ordinary shape, Nx dtype, cleared ordinary names, retained
vectorized axes, and output values. Float comparisons use `atol = rtol = 1e-5`.
Python boolean results are compared as unsigned `0/1` Nx tensors.

Vectorized checks use distinct positive scalings of each Python input across
six lanes. Their expectations follow the scaling of the reference results;
products use the scale raised to the number of reduced factors. Boolean checks
mix unmodified lanes and all-zero lanes. This tests that operations do not mix
vectorized lanes with ordinary axes.

The 24 `parse_shape` cases check Python's exact maps on eager tensors, templates,
and tensors with two vectorized axes. They also consume the metadata in JIT and
template-compiled tensor expressions, exercise vectorized JIT, and differentiate
expressions whose output shape comes from the parsed dimensions. Reading a
template demonstrates that tensor contents are unnecessary.

## Gradient oracle

Every original transformation case has a separate gradient reference with the
same operation, shape, pattern, reducer, and supplied lengths: **67 cases and
1,234 scalar partial derivatives**. A test checks the correspondence between
the forward and gradient corpora.

The Python generator evaluates `sum(einops_output * weights)` in NumPy float64
and computes central differences with step `1e-5`. Nonuniform output weights
make element order and repeated-axis accumulation observable. Inputs are
distinct, positive, bounded values, giving unique min/max extrema and stable
products. Each derivative must agree when the step is halved before the fixture
is emitted; the stored values are rounded to eight decimal places.

The Elixir suite compares `Nx.Defn.grad` results, both directly and through
explicit Evaluator JIT, with these independent references in float32 using
`atol = rtol = 1e-4`. `Nx.Defn.grad` itself traces a numerical computation;
these are two invocation paths through Nx autodiff.

Four additional analytic regressions cover:

- Products with negative factors, one zero, and multiple zeros.
- Equal sharing of minimum gradients at two-way and three-way ties.
- Equal sharing of maximum gradients at two-way and three-way ties.
- Weighted repetition gradients accumulated independently in vectorized lanes.

Min/max ties do not have a unique mathematical derivative. Their tests pin Nx's
subgradient convention rather than claim Python finite-difference equivalence.
Boolean reductions are validated for values, without a smooth-gradient claim.
`parse_shape` returns static metadata; its gradient checks concern the tensor
expressions that use that metadata.

## Error diagnostics and the ellipsis correction

The additional compatibility suite checks **27 invalid-call scenarios** in
eager execution, JIT tracing, and compilation from templates. Each scenario
requires the same complete `Einops.Error` exception, including `operation`,
`pattern`, `shape`, `axes`, and `reason`. Covered cases include unsupported
Python syntax, malformed patterns, invalid lengths, rank errors, unsupported
reducers/callbacks, and operation-specific constraints. Existing tests also
cover ambiguous splits, runtime tensor lengths, and other argument validation.

This verification found and fixed one semantic mismatch: an input ellipsis
could previously disappear from `rearrange` or `repeat` when it expanded to zero
dimensions. For example, Python rejects `"b ... -> b"` even for shape `{2}`.
Validation now checks ellipsis retention before expansion, so the Nx operation
raises a contextual `Einops.Error` too. Retained/grouped empty ellipses and
removing an ellipsis through `reduce` remain valid and have regression tests.

The project promises contextual errors for its documented unsupported syntax;
it does not duplicate Python's exact exception wording.

## Compatibility boundaries

- The reference comparisons cover the documented shared syntax. This is not
  a claim of complete Python einops API or dtype equivalence.
- Dtype promotion and numerical behavior follow Nx. In particular, Python's
  boolean dtype maps to Nx's unsigned logical results, and integer reduction
  policies can differ between the libraries.
- Axis labels use the documented ASCII grammar. Trailing underscores and
  general whitespace are accepted by this implementation; Python's broader
  Unicode label syntax is unsupported.
- `parse_shape` accepts labels, `_`, and ungrouped `...`. Python's additional
  grouping and singleton-literal syntax is deliberately rejected.
- Empty/nested groups, anonymous numeric factors other than `1`, tensor-list
  inputs, and custom reduction callbacks remain unsupported. Lengths must be
  positive integers known while tracing.
- Nx tensor dimensions must be positive. Scalars and an ellipsis representing
  zero ordinary axes are supported; zero-sized tensor dimensions are a
  different case and cannot be constructed as Nx tensors.
- Complex numbers, nonfinite values, every dtype, and optional native
  backends/compilers are outside this Python reference verification.

## Test organization

| File | Checks |
|---|---|
| `einops_test.exs` | All four operations, seven reducers, values/shapes, scalar/singleton behavior, grouping, ellipses, names, vectorization, Nx dtype behavior, and doctests |
| `error_test.exs` | Parser and argument errors, operation constraints, and exception fields |
| `defn_test.exs` | Numerical definitions, JIT/AOT, shape specialization, static options, runtime-length rejection, metadata, vectorization, and a composed gradient |
| `reference_test.exs` | 67 original and 10 logical Python fixtures across six execution configurations, plus corpus guards |
| `parse_shape_reference_test.exs` | 24 Python metadata oracles with eager/template/vectorized and compiled/differentiable use |
| `gradient_reference_test.exs` | 67 Python numerical gradient oracles, a corpus-alignment guard, and four analytic edge cases |
| `compatibility_error_test.exs` | Consistent eager/JIT/AOT errors, empty-ellipsis regressions, and ordinary-axis diagnostics on vectorized tensors |

Data-only fixture files are explicitly excluded from Mix test discovery using
`:test_ignore_filters`; their test modules load them as external resources.

## Regenerate the independent fixtures

The optional Python dependencies are pinned in `tools/reference-requirements.txt`.
From the repository root:

```sh
python -m pip install -r tools/reference-requirements.txt
python tools/generate_reference.py
python tools/generate_gradient_reference.py
mise exec -- mix format "test/fixtures/*.exs"
git diff --exit-code -- test/fixtures
mise exec -- mix test --warnings-as-errors
```

The value/shape generator writes the original 67 cases, the 10 logical cases,
and the 24 shape maps. The gradient generator writes its separate 67-case
oracle. Both accept `--output-dir PATH` to generate elsewhere for comparison.
Each fixture header records the Python package versions and generation command.
