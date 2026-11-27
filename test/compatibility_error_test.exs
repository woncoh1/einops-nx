defmodule Einops.CompatibilityErrorTest do
  use ExUnit.Case, async: true

  # These cases pin both the unsupported Python syntax boundaries and the
  # diagnostics users receive before an Nx expression can execute.
  @cases [
    {"scalar rearrange must retain an empty ellipsis", :rearrange, {}, "... ->", [], nil,
     "cannot remove input '...'"},
    {"rearrange must retain an empty ellipsis", :rearrange, {2}, "b ... -> b", [], nil,
     "cannot remove input '...'"},
    {"rearrange must retain a nonempty ellipsis", :rearrange, {2, 3}, "b ... -> b", [], nil,
     "cannot remove input '...'"},
    {"scalar repeat must retain an empty ellipsis", :repeat, {}, "... -> r", [r: 2], nil,
     "cannot remove input '...'"},
    {"repeat must retain an empty ellipsis", :repeat, {2}, "b ... -> b r", [r: 2], nil,
     "cannot remove input '...'"},
    {"repeat cannot drop a size-one named axis", :repeat, {2, 1}, "b c -> b", [], nil,
     "cannot remove axes"},
    {"a new repeated axis still needs a length", :repeat, {2}, "b -> b r", [], nil,
     "needs an explicit length"},
    {"reduce cannot introduce a named axis", :reduce, {2, 3}, "b c -> b r", [r: 1], :sum,
     "cannot introduce axes"},
    {"unsupported reduction", :reduce, {2, 3}, "b c -> b", [], :median,
     "unsupported reduction :median"},
    {"reducers are atoms rather than Python strings", :reduce, {2, 3}, "b c -> b", [], "sum",
     "unsupported reduction"},
    {"custom reduction callbacks are unsupported", :reduce, {2, 3}, "b c -> b", [], &Nx.sum/2,
     "unsupported reduction"},
    {"anonymous repeat factors need named lengths", :repeat, {2}, "b -> b 2", [], nil,
     "invalid token"},
    {"anonymous reduction factors need named lengths", :reduce, {2, 6}, "b (h 2) -> b h", [],
     :sum, "invalid token"},
    {"empty groups require the singleton literal", :rearrange, {2, 1}, "b () -> b", [], nil,
     "empty groups are not supported"},
    {"nested groups are unsupported", :repeat, {2}, "b -> ((b r))", [r: 2], nil,
     "nested axis groups are not supported"},
    {"parse_shape rejects grouped labels", :parse_shape, {2, 3}, "b (c)", [], nil,
     "parse_shape accepts only"},
    {"parse_shape rejects singleton literals", :parse_shape, {2, 1}, "b 1", [], nil,
     "parse_shape accepts only"},
    {"parse_shape rejects anonymous lengths", :parse_shape, {2, 3}, "2 c", [], nil,
     "invalid token"},
    {"parse_shape rejects grouped ellipses", :parse_shape, {2, 3}, "b (...)", [], nil,
     "parse_shape accepts only"},
    {"parse_shape rejects duplicate named axes", :parse_shape, {2, 2}, "b b", [], nil,
     "duplicate axis"},
    {"parse_shape rejects multiple ellipses", :parse_shape, {2, 3}, "... ...", [], nil,
     "duplicate axis"},
    {"parse_shape checks the minimum rank", :parse_shape, {2}, "b ... c", [], nil,
     "rank mismatch"},
    {"patterns must be valid UTF-8", :rearrange, {2}, <<255>>, [], nil,
     "valid UTF-8 pattern string"},
    {"labels are limited to ASCII", :rearrange, {2}, "β -> β", [], nil, "invalid token"},
    {"static split lengths must divide exactly", :rearrange, {2, 12}, "b (h w) -> b h w", [h: 5],
     nil, "12 is not divisible by 5"},
    {"static length maps require string keys", :repeat, {2}, "b -> b r", %{r: 2}, nil,
     "map keys must be label strings"},
    {"unused static lengths are errors", :reduce, {2, 3}, "b c -> b", [missing: 2], :sum,
     "unused axis length"}
  ]

  for {name, operation, shape, pattern, axes, reduction, reason} <- @cases do
    @example %{
      operation: operation,
      shape: shape,
      pattern: pattern,
      axes: axes,
      reduction: reduction
    }
    @reason reason

    test "#{name}: eager, JIT and template compilation have the same diagnostics" do
      example = @example
      x = Nx.broadcast(Nx.tensor(1.0), example.shape)
      transform = fn tensor -> run_example(tensor, example) end

      eager_error = assert_raise Einops.Error, fn -> transform.(x) end
      assert_context(eager_error, example, @reason)

      jit_error =
        assert_raise Einops.Error, fn ->
          Nx.Defn.jit(transform, compiler: Nx.Defn.Evaluator).(x)
        end

      compiled_error =
        assert_raise Einops.Error, fn ->
          Nx.Defn.compile(transform, [Nx.template(example.shape, :f32)],
            compiler: Nx.Defn.Evaluator
          )
        end

      assert jit_error == eager_error
      assert compiled_error == eager_error
    end
  end

  test "an empty ellipsis can be retained, grouped or reduced" do
    x = Nx.tensor([2.0, -3.0])

    transforms = [
      {fn tensor -> Einops.rearrange(tensor, "b ... -> ... b") end, {2}, [2.0, -3.0]},
      {fn tensor -> Einops.rearrange(tensor, "b ... -> b (...)") end, {2, 1}, [2.0, -3.0]},
      {fn tensor -> Einops.repeat(tensor, "b ... -> r ... b", r: 2) end, {2, 2},
       [2.0, -3.0, 2.0, -3.0]},
      {fn tensor -> Einops.reduce(tensor, "b ... -> b", :sum) end, {2}, [2.0, -3.0]}
    ]

    for {transform, shape, values} <- transforms do
      for result <- [transform.(x), Nx.Defn.jit(transform, compiler: Nx.Defn.Evaluator).(x)] do
        assert Nx.shape(result) == shape
        assert Nx.to_flat_list(result) == values
      end
    end
  end

  test "vectorized diagnostics describe ordinary dimensions" do
    x = Nx.iota({2, 3, 5}) |> Nx.vectorize(:ensemble)
    transform = fn tensor -> Einops.rearrange(tensor, "b (h w) -> b h w", h: 2) end

    eager_error = assert_raise Einops.Error, fn -> transform.(x) end
    jit_error = assert_raise Einops.Error, fn -> Nx.Defn.jit(transform).(x) end

    assert eager_error.shape == {3, 5}
    assert eager_error.reason =~ "input axis 1 of length 5"
    assert eager_error.reason =~ "5 is not divisible by 2"
    assert jit_error == eager_error
  end

  defp assert_context(error, example, reason) do
    assert error.operation == example.operation
    assert error.pattern == example.pattern
    assert error.shape == example.shape
    assert error.axes == example.axes
    assert error.reason =~ reason

    for label <- ["Operation:", "Pattern:", "Shape:", "Axes:"] do
      assert Exception.message(error) =~ label
    end
  end

  defp run_example(tensor, %{operation: :rearrange} = example),
    do: Einops.rearrange(tensor, example.pattern, example.axes)

  defp run_example(tensor, %{operation: :repeat} = example),
    do: Einops.repeat(tensor, example.pattern, example.axes)

  defp run_example(tensor, %{operation: :reduce} = example),
    do: Einops.reduce(tensor, example.pattern, example.reduction, example.axes)

  defp run_example(tensor, %{operation: :parse_shape} = example),
    do: Einops.parse_shape(tensor, example.pattern)
end
