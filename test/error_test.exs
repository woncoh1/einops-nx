defmodule Einops.ErrorTest do
  use ExUnit.Case, async: true

  test "errors retain the complete original call context" do
    x = Nx.iota({2, 12})
    pattern = "b (h w) -> b h w"
    error = assert_raise Einops.Error, fn -> Einops.rearrange(x, pattern, h: 5) end
    assert error.operation == :rearrange
    assert error.pattern == pattern
    assert error.shape == {2, 12}
    assert error.axes == [h: 5]
    assert error.reason =~ "12 is not divisible by 5"
    assert Exception.message(error) =~ "Operation: rearrange"
    assert Exception.message(error) =~ "Shape:     {2, 12}"
  end

  test "does not guess an ambiguous factorization" do
    assert_raise Einops.Error, ~r/supply all but one length/, fn ->
      Einops.rearrange(Nx.iota({2, 12}), "b (h w) -> b h w")
    end
  end

  for {pattern, message} <- [
        {"b c", "exactly one '->'"},
        {"b c -> b c -> b c", "exactly one '->'"},
        {"b c -> b b", "duplicate axis"},
        {"b b -> b", "duplicate axis"},
        {"b c -> b ...", "requires '...'"},
        {"... ... -> ...", "duplicate axis"},
        {"b (...) -> b ...", "inside an input group"},
        {"b (c -> b c", "unclosed '('"},
        {"b c) -> b c", "unexpected ')'"},
        {"b ((c)) -> b c", "nested axis groups"},
        {"b () -> b", "empty groups"},
        {"b c -> b $", "invalid token"},
        {"b c -> b _", "invalid token"},
        {"b c -> b 2", "invalid token"},
        {"b c -> b (c * 2)", "invalid token"},
        {"a b c -> a b c", "rank mismatch"},
        {"a b c ... -> ... a b c", "rank mismatch"},
        {"b c -> b", "cannot remove axes"},
        {"b c -> b c r", "cannot introduce axes"},
        {"b 1 -> b", "shape mismatch"}
      ] do
    @pattern pattern
    @message message
    test "rejects malformed or invalid rearrangement #{inspect(pattern)}" do
      error = assert_raise Einops.Error, fn -> Einops.rearrange(Nx.iota({2, 3}), @pattern) end
      assert error.reason =~ @message
    end
  end

  for {axes, message} <- [
        {[h: 5], "not divisible"},
        {[h: 3, w: 5], "shape mismatch"},
        {[b: 3, h: 3], "shape mismatch"},
        {[h: 0], "positive integer"},
        {[h: -2], "positive integer"},
        {[h: 3.0], "positive integer"},
        {[h: "3"], "positive integer"},
        {[h: 3, h: 3], "more than once"},
        {[height: 3], "unused axis length"},
        {%{h: 3}, "map keys must be label strings"},
        {[{"h", 3}], "keyword list"},
        {nil, "keyword list"}
      ] do
    @axes axes
    @message message
    test "rejects invalid axis lengths #{inspect(axes)}" do
      error = assert_raise Einops.Error, fn -> Einops.rearrange(Nx.iota({2, 12}), "b (h w) -> b h w", @axes) end
      assert error.reason =~ @message
    end
  end

  test "does not allow a named size-one axis to disappear in rearrange" do
    assert_raise Einops.Error, ~r/cannot remove axes/, fn ->
      Einops.rearrange(Nx.iota({2, 1}), "b c -> b")
    end
  end

  test "repeat requires explicit lengths for new axes" do
    assert_raise Einops.Error, ~r/needs an explicit length/, fn ->
      Einops.repeat(Nx.iota({2, 3}), "b c -> b c copies")
    end
  end

  test "repeat cannot remove input axes" do
    assert_raise Einops.Error, ~r/cannot remove axes/, fn ->
      Einops.repeat(Nx.iota({2, 3}), "b c -> b copies", copies: 2)
    end
  end

  test "reduce cannot invent output axes" do
    assert_raise Einops.Error, ~r/cannot introduce axes/, fn ->
      Einops.reduce(Nx.iota({2, 3}), "b c -> b copies", :sum, copies: 2)
    end
  end

  test "unsupported reducers fail before tensor execution" do
    assert_raise Einops.Error, ~r/unsupported reduction :median/, fn ->
      Einops.reduce(Nx.iota({2, 3}), "b c -> b", :median)
    end
  end

  test "requires a tensor and a pattern string" do
    assert_raise Einops.Error, ~r/expected an Nx.Tensor/, fn -> Einops.rearrange([1, 2], "n -> n") end
    assert_raise Einops.Error, ~r/pattern string/, fn -> Einops.rearrange(Nx.tensor([1]), :pattern) end
    assert_raise Einops.Error, ~r/pattern string/, fn -> Einops.rearrange(Nx.tensor([1]), <<255>>) end
  end

  for pattern <- ["b b", "b", "b (c)", "b 1", "b -> c", "... ..."] do
    @pattern pattern
    test "parse_shape rejects #{inspect(pattern)}" do
      assert_raise Einops.Error, fn -> Einops.parse_shape(Nx.iota({2, 3}), @pattern) end
    end
  end
end
