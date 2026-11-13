defmodule Einops.DefnTest.Functions do
  import Nx.Defn

  @spec split_heads(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn split_heads(x) do
    Einops.rearrange(x, "b t (heads d) -> b heads t d", heads: 2)
  end

  @spec pooled(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn pooled(x) do
    Einops.reduce(x, "b (h ph) (w pw) c -> b h w c", :mean, ph: 2, pw: 2)
  end

  @spec repeated(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn repeated(x) do
    Einops.repeat(x, "b c -> c copies b", copies: 3)
  end

  @spec composed(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn composed(x) do
    x
    |> Einops.rearrange("b t (heads d) -> b heads t d", heads: 2)
    |> Einops.reduce("b heads t d -> b heads d", :mean)
    |> Einops.repeat("b heads d -> b heads copies d", copies: 3)
  end

  @spec flatten(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn flatten(x), do: Einops.rearrange(x, "b ... -> b (...)")

  @spec from_metadata(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn from_metadata(x), do: reshape_from_metadata(x)

  @spec reshape_from_metadata(Nx.Tensor.t()) :: Nx.Tensor.t()
  deftransformp reshape_from_metadata(x) do
    %{"batch" => batch} = Einops.parse_shape(x, "batch ...")
    Nx.reshape(x, {batch, :auto})
  end

  @spec split_with_options(Nx.Tensor.t(), keyword(pos_integer())) :: Nx.Tensor.t()
  defn split_with_options(x, axes) do
    Einops.rearrange(x, "b (h w) -> b h w", axes)
  end

  @spec bad_dynamic_length(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  defn bad_dynamic_length(x, length) do
    Einops.rearrange(x, "b (h w) -> b h w", h: length)
  end

  @spec loss(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn loss(x) do
    repeated = Einops.repeat(x, "b f -> b f copies", copies: 3)
    Einops.reduce(repeated * repeated, "b f copies ->", :sum)
  end
end

defmodule Einops.DefnTest do
  use ExUnit.Case, async: true
  alias Einops.DefnTest.Functions, as: F

  test "rearrange inside a numerical definition" do
    x = Nx.iota({2, 3, 4}, type: :f32)
    expected = x |> Nx.reshape({2, 3, 2, 2}) |> Nx.transpose(axes: [0, 2, 1, 3])
    assert_tensor(F.split_heads(x), expected)
  end

  test "reduce inside a numerical definition" do
    x = Nx.iota({2, 6, 8, 3}, type: :f32)
    expected = x |> Nx.reshape({2, 3, 2, 4, 2, 3}) |> Nx.mean(axes: [2, 4])
    assert_tensor(F.pooled(x), expected)
  end

  test "repeat inside a numerical definition" do
    x = Nx.iota({2, 4}, type: :f32)
    expected = x |> Nx.transpose() |> Nx.broadcast({4, 3, 2}, axes: [0, 2])
    assert_tensor(F.repeated(x), expected)
  end

  test "composed transformations under explicit jit" do
    x = Nx.iota({2, 3, 4}, type: :f32)
    expected =
      x
      |> Nx.reshape({2, 3, 2, 2})
      |> Nx.mean(axes: [1])
      |> Nx.broadcast({2, 2, 3, 2}, axes: [0, 1, 3])

    compiled = Nx.Defn.jit(&F.composed/1, compiler: Nx.Defn.Evaluator)
    assert_tensor(compiled.(x), expected)
    assert_tensor(compiled.(Nx.add(x, 1)), Nx.add(expected, 1))
  end

  test "can compile ahead of time from a tensor template" do
    compiled = Nx.Defn.compile(&F.split_heads/1, [Nx.template({2, 3, 4}, :f32)], compiler: Nx.Defn.Evaluator)
    x = Nx.iota({2, 3, 4}, type: :f32)
    assert_tensor(compiled.(x), F.split_heads(x))
  end

  test "shape specialization handles different ellipsis ranks" do
    compiled = Nx.Defn.jit(&F.flatten/1, compiler: Nx.Defn.Evaluator)

    for shape <- [{2}, {2, 3}, {2, 3, 4}] do
      x = Nx.iota(shape, type: :f32)
      assert_tensor(compiled.(x), Nx.reshape(x, {2, :auto}))
    end
  end

  test "parse_shape provides metadata during tracing" do
    x = Nx.iota({2, 3, 4}, type: :f32)
    assert_tensor(F.from_metadata(x), Nx.reshape(x, {2, 12}))
  end

  test "keyword lengths are static defn options" do
    x = Nx.iota({2, 12}, type: :f32)
    assert_tensor(F.split_with_options(x, h: 3), Nx.reshape(x, {2, 3, 4}))
    assert_tensor(F.split_with_options(x, h: 4), Nx.reshape(x, {2, 4, 3}))
  end

  test "runtime tensor values cannot determine axis lengths" do
    assert_raise Einops.Error, ~r/known while tracing/, fn ->
      F.bad_dynamic_length(Nx.iota({2, 12}, type: :f32), Nx.tensor(3))
    end
  end

  test "invalid shapes fail during tracing with Einops diagnostics" do
    assert_raise Einops.Error, ~r/not divisible/, fn ->
      F.split_heads(Nx.iota({2, 3, 5}, type: :f32))
    end
  end

  test "vectorized axes survive a defn call" do
    x = Nx.iota({2, 3, 4}, type: :f32) |> Nx.vectorize(:ensemble)
    expected = x |> Nx.transpose() |> Nx.broadcast({4, 3, 3}, axes: [0, 2])
    assert_tensor(F.repeated(x), expected)
  end

  test "autodiff propagates through repeated and reduced axes" do
    x = Nx.iota({2, 3}, type: :f32)
    gradient = Nx.Defn.grad(&F.loss/1).(x)
    assert_tensor(gradient, Nx.multiply(x, 6))
  end

  @spec assert_tensor(Nx.Tensor.t(), Nx.Tensor.t()) :: :ok
  defp assert_tensor(actual, expected) do
    assert Nx.shape(actual) == Nx.shape(expected)
    assert Nx.type(actual) == Nx.type(expected)
    assert actual.vectorized_axes == expected.vectorized_axes
    assert Nx.to_number(Nx.all_close(Nx.devectorize(actual), Nx.devectorize(expected))) == 1
    :ok
  end
end
