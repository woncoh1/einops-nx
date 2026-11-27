defmodule Einops.ParseShapeReferenceTest.Functions do
  import Nx.Defn

  defn broadcast_from_metadata(x, options \\ []) do
    metadata_broadcast(x, options)
  end

  defn metadata_loss(x, options \\ []) do
    x
    |> Nx.multiply(x)
    |> metadata_broadcast(options)
    |> Nx.sum()
  end

  deftransformp metadata_broadcast(x, options) do
    shape =
      x
      |> Einops.parse_shape(Keyword.fetch!(options, :pattern))
      |> Enum.sort()
      |> Enum.map(fn {_label, size} -> size end)
      |> List.to_tuple()

    Nx.broadcast(Nx.sum(x), shape)
  end
end

defmodule Einops.ParseShapeReferenceTest do
  use ExUnit.Case, async: true
  alias Einops.ParseShapeReferenceTest.Functions, as: F

  @fixture_path Path.join(__DIR__, "fixtures/parse_shape_cases.exs")
  @external_resource @fixture_path
  @cases elem(Code.eval_file(@fixture_path), 0)

  for {example, index} <- Enum.with_index(@cases) do
    @example example
    test "Python parse_shape reference #{index}: #{inspect(example.pattern)}" do
      example = @example
      x = Nx.add(Nx.iota(example.shape, type: :f32), 1)
      template = Nx.template(example.shape, :f32)

      assert Einops.parse_shape(x, example.pattern) == example.expected
      assert Einops.parse_shape(template, example.pattern) == example.expected

      vectorized =
        x
        |> Nx.broadcast(List.to_tuple([2, 3 | Tuple.to_list(example.shape)]))
        |> Nx.vectorize([:ensemble, :member])

      assert Einops.parse_shape(vectorized, example.pattern) == example.expected

      output_shape =
        example.expected
        |> Enum.sort()
        |> Enum.map(fn {_label, size} -> size end)
        |> List.to_tuple()

      expected = Nx.broadcast(Nx.sum(x), output_shape)
      fun = fn input -> F.broadcast_from_metadata(input, pattern: example.pattern) end
      jit = Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator)
      compiled = Nx.Defn.compile(fun, [template], compiler: Nx.Defn.Evaluator)

      assert_tensor(jit.(x), expected)
      assert_tensor(compiled.(x), expected)
      assert_tensor(jit.(vectorized), Nx.broadcast(Nx.sum(vectorized), output_shape))

      # parse_shape returns static metadata. Its dimensions may still control
      # the shape of a differentiable tensor expression during tracing.
      gradient = Nx.Defn.grad(fn input -> F.metadata_loss(input, pattern: example.pattern) end)
      repetitions = example.expected |> Map.values() |> Enum.product()
      assert_tensor(gradient.(x), Nx.multiply(x, 2 * repetitions))
    end
  end

  defp assert_tensor(actual, expected) do
    assert Nx.shape(actual) == Nx.shape(expected)
    assert Nx.type(actual) == Nx.type(expected)
    assert actual.vectorized_axes == expected.vectorized_axes
    assert Nx.to_number(Nx.all_close(Nx.devectorize(actual), Nx.devectorize(expected))) == 1
  end
end
