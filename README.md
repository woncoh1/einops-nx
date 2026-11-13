# Einops for Nx

Four operations, ordinary Nx tensors, one shared transformation pipeline.
This is an unpublished implementation draft, not an existing Hex release.

**Validation status:** the Elixir source and ExUnit tests have not been compiled
or executed in the authoring environment, which has no Elixir/Erlang runtime.
The 67 committed Python-reference fixtures were generated successfully.
See `TESTING.md` before treating this draft as a release.

## Get started

Requires Elixir 1.17 or later within 1.x, and Nx 1.0.x.
Nx is the only runtime dependency; the tests use the built-in ExUnit framework.

```sh
cd einops
mix deps.get
mix format
mix compile --warnings-as-errors
mix test
mix run examples/quickstart.exs
```

From another Mix project, use a local path dependency:

```elixir
{:einops, path: "../einops"}
```

No dependency lockfile or native compiler/backend is bundled.

## API

```elixir
Einops.rearrange(tensor, pattern, axes \\ [])
Einops.reduce(tensor, pattern, reduction, axes \\ [])
Einops.repeat(tensor, pattern, axes \\ [])
Einops.parse_shape(tensor, pattern)
```

Axis lengths are a keyword list or a string-keyed map. They are positive
integers, not tensors. Patterns address positional, ordinary axes; their
labels are independent of Nx axis names.

```elixir
x = Nx.iota({2, 3, 4}, type: :f32)

Einops.rearrange(x, "b h c -> b c h")
# Shape: {2, 4, 3}

Einops.rearrange(x, "b h (heads d) -> b heads h d", heads: 2)
# Shape: {2, 2, 3, 2}

Einops.reduce(x, "b h c -> b c", :mean)
# Shape: {2, 4}

Einops.repeat(x, "b h c -> b h c copies", copies: 5)
# Shape: {2, 3, 4, 5}

Einops.parse_shape(x, "batch _ channels")
# %{"batch" => 2, "channels" => 4}
```

The order inside a group matters:

```elixir
x = Nx.tensor([1, 2, 3])

Einops.repeat(x, "n -> (n copies)", copies: 2)
# [1, 1, 2, 2, 3, 3]

Einops.repeat(x, "n -> (copies n)", copies: 2)
# [1, 2, 3, 1, 2, 3]
```

The supported reducers are `:sum`, `:mean`, `:min`, `:max`, `:prod`, `:any`,
and `:all`. Numerical behavior and dtype promotion are delegated to Nx.
With no eliminated elementary axes, `reduce` performs only rearrangement;
it does not force a boolean conversion or dtype promotion.

## Pattern rules

| Syntax | Meaning |
|---|---|
| `batch`, `patch_h` | A local axis label: ASCII letter, then letters/digits/underscores |
| `(h w)` | Merge or split axes; rightmost factors vary fastest |
| `1` | An anonymous singleton dimension, including inside a group |
| `...` | Zero or more dimensions; at most once per side |
| `(...)` | Merge ellipsis dimensions; allowed on the output side only |
| `_` | Skip one dimension in `parse_shape` only |
| Empty side | Scalar shape, as in `"b h ->"` |

A named label occurs at most once per side. `rearrange` must retain every
named/ellipsis axis; `repeat` may introduce axes but not remove them; `reduce`
may remove axes but not invent named ones. Anonymous singleton dimensions may
be inserted or removed independently. A zero-length ellipsis contributes no
axes, and an output group containing only that ellipsis has size one.

For each input group, at most one factor may be unknown. Every supplied
length must match the actual shape, every split must divide exactly, and
unused or duplicate length arguments are errors. Even a size-one named axis
must be retained by `rearrange`; use the literal `1` to mark a removable
singleton instead.

`parse_shape` accepts only labels, `_`, and ungrouped `...`. Its result has
string keys and can be passed directly as an axis-length map.

## Diagnostics

```elixir
Einops.rearrange(Nx.iota({2, 12}), "b (h w) -> b h w", h: 5)
```

```text
** (Einops.Error) cannot split input axis 1 of length 12 (h w): 12 is not divisible by 5

Operation: rearrange
Pattern:   "b (h w) -> b h w"
Shape:     {2, 12}
Axes:      [h: 5]
```

The exception also exposes `operation`, `pattern`, `shape`, `axes`, and
`reason` as fields. Axis indexes in diagnostics are zero-based.

## Numerical definitions

The public functions use `Nx.Defn.deftransform`: parsing, validation, and shape
inference happen in ordinary Elixir while tracing. Only Nx tensor operations
are placed in the numerical expression. Patterns, reducer choices, and supplied
axis lengths must be known at that point, not computed from tensor values.

```elixir
defmodule AttentionLayout do
  import Nx.Defn

  @spec split_heads(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn split_heads(x) do
    Einops.rearrange(x, "batch tokens (heads width) -> batch heads tokens width",
      heads: 8
    )
  end
end
```

Pass variable static lengths to an outer `defn` as keyword options, not as
numeric tensor arguments. `test/defn_test.exs` shows that pattern as well as
JIT, ahead-of-time compilation, and gradient tests.

`parse_shape` returns host-side metadata. When using its map inside numerical
code, consume it in a transform, rather than expecting an ordinary metadata
map to be returned by the compiled tensor computation:

```elixir
@spec flatten(Nx.Tensor.t()) :: Nx.Tensor.t()
defn flatten(x), do: flatten_from_shape(x)

@spec flatten_from_shape(Nx.Tensor.t()) :: Nx.Tensor.t()
deftransformp flatten_from_shape(x) do
  %{"batch" => batch} = Einops.parse_shape(x, "batch ...")
  Nx.reshape(x, {batch, :auto})
end
```

Vectorized axes are left to Nx and are not described in the pattern. Ordinary
axis names are cleared consistently, including for identity transformations.
No labels are converted into atoms and no tensor contents are extracted into
Elixir lists by the implementation.

## Implementation

```text
lib/einops.ex          Public transforms and Nx execution
lib/einops/pattern.ex  Small parser and ellipsis expansion
lib/einops/plan.ex     Shape inference and operation validation
lib/einops/error.ex    Contextual exceptions
```

The shared execution sequence is:

```text
reshape -> optional reduction -> transpose -> optional broadcast -> reshape
```

There is no process, cache, custom tensor wrapper, backend selection, or code
evaluation in the implementation. Pattern work happens on every eager call,
and during tracing for compiled calls.

## Deliberate boundaries

This draft is an einops-inspired subset, not full Python einops compatibility.
No `einsum`, packing, Axon adapters, callbacks, nested/empty groups, arithmetic
expressions, tensor-list inputs, or anonymous numeric factors other than `1`.
Use `ph: 2` instead of writing `(h 2)`. There is no custom reduction callback.

EXLA and Torchx are not dependencies and have not been validated here. Backend
support and supported dtypes ultimately depend on the Nx operations used.
This draft is not publication-ready: runtime verification, a license, and Hex
package naming/metadata remain release decisions.

## Acknowledgments

This library is inspired by [einops](https://github.com/arogozhnikov/einops),
created by Alex Rogozhnikov. Its readable tensor-transformation notation and
API are the foundation for this Elixir/Nx implementation.

This is an unofficial implementation and is not affiliated with or endorsed by
Alex Rogozhnikov or the original einops project.

## Primary references

- Nx 1.0 source and version requirements: https://github.com/elixir-nx/nx/blob/v1.0.0/nx/mix.exs
- Nx transforms and compilation: https://hexdocs.pm/nx/1.0.0/Nx.Defn.html
- Nx tensor operations: https://hexdocs.pm/nx/1.0.0/Nx.html
- Python einops semantics: https://einops.rocks/api/rearrange/
- Python reductions: https://einops.rocks/api/reduce/
- Python shape parsing: https://einops.rocks/api/parse_shape/
