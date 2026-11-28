defmodule Einops.GradientReferenceTest do
  use ExUnit.Case, async: true

  @fixture_path Path.join(__DIR__, "fixtures/gradient_cases.exs")
  @external_resource @fixture_path
  @cases @fixture_path |> Code.eval_file() |> elem(0)

  @transform_fixture_path Path.join(__DIR__, "fixtures/reference_cases.exs")
  @external_resource @transform_fixture_path
  @transform_cases @transform_fixture_path |> Code.eval_file() |> elem(0)

  # NumPy/einops uses float64 central differences; Nx evaluates and
  # differentiates float32 tensors. Account for the two numerical precisions.
  @tolerance 1.0e-4

  test "gradient corpus covers all 67 transformation reference cases" do
    keys = [:operation, :shape, :pattern, :reduction, :axes]
    assert length(@cases) == 67

    assert Enum.map(@cases, &Map.take(&1, keys)) ==
             Enum.map(@transform_cases, &Map.take(&1, keys))
  end

  for {example, index} <- Enum.with_index(@cases, 1) do
    @example example
    test "Python gradient #{index}: #{example.operation} #{example.pattern} #{example.reduction}" do
      example = @example
      x = example.input |> Nx.tensor(type: :f32) |> Nx.reshape(example.shape)
      weights = example.weights |> Nx.tensor(type: :f32) |> Nx.reshape(example.output_shape)
      expected = example.gradient |> Nx.tensor(type: :f32) |> Nx.reshape(example.shape)

      gradient =
        Nx.Defn.grad(fn input ->
          input |> run_example(example) |> Nx.multiply(weights) |> Nx.sum()
        end)

      assert_gradient(gradient.(x), expected)
      assert_gradient(Nx.Defn.jit(gradient, compiler: Nx.Defn.Evaluator).(x), expected)
    end
  end

  test "product gradients handle one zero, multiple zeros, and signed factors" do
    x = Nx.tensor([[0.0, 2.0, -3.0], [0.0, 0.0, 4.0], [2.0, -3.0, 4.0]])
    weights = Nx.tensor([1.0, 2.0, -0.5])
    expected = Nx.tensor([[-6.0, 0.0, 0.0], [0.0, 0.0, 0.0], [6.0, -4.0, 3.0]])

    gradient =
      Nx.Defn.grad(fn input ->
        input
        |> Einops.reduce("batch item -> batch", :prod)
        |> Nx.multiply(weights)
        |> Nx.sum()
      end)

    assert_gradient(gradient.(x), expected)
    assert_gradient(Nx.Defn.jit(gradient, compiler: Nx.Defn.Evaluator).(x), expected)
  end

  test "minimum gradients use Nx's equal split at ties" do
    # There is no unique derivative at a tie. This is Nx's
    # subgradient convention, not a Python finite-difference parity claim.
    x = Nx.tensor([[1.0, 1.0, 1.0, 4.0], [2.0, 3.0, 2.0, 5.0]])
    expected = Nx.tensor([[1 / 3, 1 / 3, 1 / 3, 0.0], [0.5, 0.0, 0.5, 0.0]])
    assert_reduction_gradient(x, :min, expected)
  end

  test "maximum gradients use Nx's equal split at ties" do
    x = Nx.tensor([[4.0, 4.0, 4.0, 1.0], [5.0, 3.0, 5.0, 2.0]])
    expected = Nx.tensor([[1 / 3, 1 / 3, 1 / 3, 0.0], [0.5, 0.0, 0.5, 0.0]])
    assert_reduction_gradient(x, :max, expected)
  end

  test "vectorized repeat gradients accumulate each lane independently" do
    x =
      Nx.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
      |> Nx.vectorize(:ensemble)

    weights = Nx.tensor([1.0, 2.0, 3.0]) |> Nx.reshape({3, 1, 1})

    gradient =
      Nx.Defn.grad(fn input ->
        repeated = Einops.repeat(input, "batch feature -> copies feature batch", copies: 3)
        repeated |> Nx.pow(2) |> Nx.multiply(weights) |> Nx.sum()
      end)

    # Every element appears with weights 1 + 2 + 3, and d(x^2)/dx = 2x.
    expected = Nx.multiply(x, 12)
    assert_gradient(gradient.(x), expected)
    assert_gradient(Nx.Defn.jit(gradient, compiler: Nx.Defn.Evaluator).(x), expected)
  end

  defp assert_reduction_gradient(x, reduction, expected) do
    gradient =
      Nx.Defn.grad(fn input ->
        input |> Einops.reduce("batch item -> batch", reduction) |> Nx.sum()
      end)

    assert_gradient(gradient.(x), expected)
    assert_gradient(Nx.Defn.jit(gradient, compiler: Nx.Defn.Evaluator).(x), expected)
  end

  defp run_example(x, %{operation: :rearrange} = example),
    do: Einops.rearrange(x, example.pattern, example.axes)

  defp run_example(x, %{operation: :repeat} = example),
    do: Einops.repeat(x, example.pattern, example.axes)

  defp run_example(x, %{operation: :reduce} = example),
    do: Einops.reduce(x, example.pattern, example.reduction, example.axes)

  defp assert_gradient(actual, expected) do
    assert Nx.shape(actual) == Nx.shape(expected)
    assert Nx.type(actual) == Nx.type(expected)
    assert actual.vectorized_axes == expected.vectorized_axes

    assert Nx.to_number(
             Nx.all_close(Nx.devectorize(actual), Nx.devectorize(expected),
               atol: @tolerance,
               rtol: @tolerance
             )
           ) == 1
  end
end
