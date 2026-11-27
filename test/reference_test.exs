defmodule Einops.ReferenceTest do
  use ExUnit.Case, async: true

  @fixture_path Path.join(__DIR__, "fixtures/reference_cases.exs")
  @external_resource @fixture_path
  @transform_cases elem(Code.eval_file(@fixture_path), 0)

  @boolean_fixture_path Path.join(__DIR__, "fixtures/boolean_cases.exs")
  @external_resource @boolean_fixture_path
  @boolean_cases elem(Code.eval_file(@boolean_fixture_path), 0)
  @cases @transform_cases ++ @boolean_cases

  test "the original Python corpus contains all 67 transformation cases" do
    assert Enum.frequencies_by(@transform_cases, & &1.operation) == %{
             rearrange: 34,
             repeat: 8,
             reduce: 25
           }
  end

  test "Python boolean fixtures exercise true and false results for both reducers" do
    for reduction <- [:any, :all] do
      values =
        @boolean_cases
        |> Enum.filter(&(&1.reduction == reduction))
        |> Enum.flat_map(& &1.expected)
        |> MapSet.new()

      assert values == MapSet.new([0, 1])
    end
  end

  for {example, index} <- Enum.with_index(@cases),
      execution <- [:eager, :jit, :compiled],
      vectorized? <- [false, true] do
    @example example
    @execution execution
    @vectorized? vectorized?
    test "Python reference #{index}: #{example.operation} #{example.pattern} #{example.reduction}, #{execution}, vectorized=#{vectorized?}" do
      example = @example
      x = fixture_tensor(example.input, example.shape, @vectorized?, :f32, input_scales(example))

      expected =
        fixture_tensor(
          example.expected,
          example.expected_shape,
          @vectorized?,
          output_type(example),
          output_scales(example)
        )

      result = execute(@execution, x, example)

      assert Nx.shape(result) == example.expected_shape
      assert Nx.type(result) == Nx.type(expected)
      assert result.names == List.duplicate(nil, tuple_size(example.expected_shape))
      assert result.vectorized_axes == expected.vectorized_axes

      assert Nx.to_number(
               Nx.all_close(Nx.devectorize(result), Nx.devectorize(expected),
                 atol: 1.0e-5,
                 rtol: 1.0e-5
               )
             ) == 1
    end
  end

  defp fixture_tensor(values, shape, false, type, _scales),
    do: values |> Nx.tensor(type: type) |> Nx.reshape(shape)

  defp fixture_tensor(values, shape, true, type, scales) do
    # Distinct lanes catch accidental mixing of vectorized and ordinary axes.
    Enum.map(scales, fn scale -> Enum.map(values, &(&1 * scale)) end)
    |> Nx.tensor(type: type)
    |> Nx.reshape(List.to_tuple([2, 3 | Tuple.to_list(shape)]))
    |> Nx.vectorize([:batch, :ensemble])
  end

  defp output_type(%{reduction: reduction}) when reduction in [:any, :all], do: :u8
  defp output_type(_example), do: :f32

  defp input_scales(%{reduction: reduction}) when reduction in [:any, :all],
    do: [0, 1, 1, 0, 1, 0]

  defp input_scales(_example), do: [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]

  defp output_scales(%{operation: :reduce, reduction: :prod} = example) do
    # A product of k factors scales by s^k when every input scales by s.
    factors = div(length(example.input), length(example.expected))
    Enum.map(input_scales(example), &:math.pow(&1, factors))
  end

  # Positive scaling commutes with rearrange, repeat, sum, mean, min and max.
  # Logical fixtures alternate unmodified lanes and all-zero (false) lanes.
  defp output_scales(example), do: input_scales(example)

  defp execute(:eager, x, example), do: run_example(x, example)

  defp execute(:jit, x, example) do
    Nx.Defn.jit(fn tensor -> run_example(tensor, example) end, compiler: Nx.Defn.Evaluator).(x)
  end

  defp execute(:compiled, x, example) do
    Nx.Defn.compile(fn tensor -> run_example(tensor, example) end, [Nx.to_template(x)],
      compiler: Nx.Defn.Evaluator
    ).(x)
  end

  defp run_example(x, %{operation: :rearrange} = example),
    do: Einops.rearrange(x, example.pattern, example.axes)

  defp run_example(x, %{operation: :repeat} = example),
    do: Einops.repeat(x, example.pattern, example.axes)

  defp run_example(x, %{operation: :reduce} = example),
    do: Einops.reduce(x, example.pattern, example.reduction, example.axes)
end
