# Verification status

## Completed in the authoring environment

- Created the library source, Mix project, examples, and ExUnit suite.
- Generated 67 reference fixtures with Python einops 0.8.2 and NumPy 2.3.5.
  They use signed, non-monotone data and store both output shapes and values.
- Reviewed the source against the Nx 1.0 documentation and source metadata.

## Not executed

Elixir/Erlang and Mix are not installed in the authoring environment. Dependency
fetching was also unavailable. Therefore, the Elixir code has **not been compiled**,
`mix format` has **not been run**, and the ExUnit suite has **not been run**.
No passing-test count, coverage percentage, or backend compatibility is claimed.
The generated Python fixtures do not establish that the Elixir implementation
passes them.

## Run locally

```sh
mix deps.get
mix format
mix compile --warnings-as-errors
mix test --trace
mix test --cover
```

The suite explicitly selects `Nx.BinaryBackend` and `Nx.Defn.Evaluator`; no native
compiler installation is necessary for these tests. Optional backend/compiler
validation should be done separately before advertising support.

## Test organization

| File | Intended checks |
|---|---|
| `einops_test.exs` | Values and shapes; all four operations; seven reducers; singletons/scalars; grouping; ellipsis; names; vectorization; dtype behavior; doctests |
| `error_test.exs` | Malformed syntax; duplicate labels; ambiguous splits; nondivisible sizes; rank mismatches; unused or duplicate lengths; invalid types; operation-specific constraints; exception fields |
| `defn_test.exs` | Numerical definitions; explicit JIT; compilation from templates; shape specialization; static options; host-side metadata; rejection of runtime lengths; vectorization; gradients |
| `reference_test.exs` | 67 generated comparisons against expected values from Python einops |

Reference fixtures are committed as Elixir literals. Running the Elixir tests
does not require Python. To regenerate them during development:

```sh
python -m pip install numpy einops
python tools/generate_reference.py
mix format test/fixtures/reference_cases.exs
mix test test/reference_test.exs
```

The generator records the Python package versions in the fixture header. The
reference comparison covers the shared semantics, not identical dtype policies
between NumPy and Nx. Nx-specific dtype behavior is checked against native Nx
operations in the separate eager tests.
