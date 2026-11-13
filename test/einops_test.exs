defmodule EinopsTest do
  use ExUnit.Case, async: true
  doctest Einops

  describe "rearrange" do
    test "transposes values, not just dimensions" do
      result = Einops.rearrange(Nx.iota({2, 3}), "b c -> c b")
      assert Nx.to_list(result) == [[0, 3], [1, 4], [2, 5]]
    end

    test "merges dimensions in the specified order" do
      x = Nx.iota({2, 3, 4})
      expected = x |> Nx.transpose(axes: [1, 0, 2]) |> Nx.reshape({6, 4})
      assert_tensor(Einops.rearrange(x, "b h c -> (h b) c"), expected)
    end

    test "infers exactly one unknown factor" do
      x = Nx.iota({2, 12})
      assert_tensor(Einops.rearrange(x, "b (h w) -> b h w", h: 3), Nx.reshape(x, {2, 3, 4}))
    end

    test "splits and permutes in one operation" do
      x = Nx.iota({2, 3, 4})
      expected = x |> Nx.reshape({2, 3, 2, 2}) |> Nx.transpose(axes: [0, 2, 1, 3])
      assert_tensor(Einops.rearrange(x, "b h (heads d) -> b heads h d", heads: 2), expected)
    end

    test "accepts a string-keyed length map" do
      x = Nx.iota({2, 12})
      assert_tensor(Einops.rearrange(x, "b (h w) -> b h w", %{"h" => 3}), Nx.reshape(x, {2, 3, 4}))
    end

    test "validates all supplied dimensions, even when none need inference" do
      x = Nx.iota({2, 12})
      assert_tensor(Einops.rearrange(x, "b (h w) -> b h w", b: 2, h: 3, w: 4), Nx.reshape(x, {2, 3, 4}))
    end

    test "inserts and removes singleton dimensions" do
      x = Nx.iota({2, 1, 3})
      assert_tensor(Einops.rearrange(x, "b 1 c -> 1 b c 1"), Nx.reshape(x, {1, 2, 3, 1}))
    end

    test "handles singleton factors inside groups" do
      x = Nx.iota({2, 3})
      assert_tensor(Einops.rearrange(x, "b (1 c) -> (b 1) c"), x)
    end

    test "handles scalar input and output" do
      x = Nx.tensor(7)
      y = Einops.rearrange(x, " -> 1 1")
      assert Nx.shape(y) == {1, 1}
      assert_tensor(Einops.rearrange(y, "1 1 -> "), x)
      assert_tensor(Einops.rearrange(x, " -> "), x)
    end

    test "moves a middle ellipsis" do
      x = Nx.iota({2, 3, 4, 5})
      assert_tensor(Einops.rearrange(x, "b ... c -> c ... b"), Nx.transpose(x, axes: [3, 1, 2, 0]))
    end

    test "composes output ellipsis dimensions" do
      x = Nx.iota({2, 3, 4, 5})
      assert_tensor(Einops.rearrange(x, "b ... c -> b (... c)"), Nx.reshape(x, {2, 60}))
    end

    test "ellipsis may expand to zero dimensions" do
      x = Nx.iota({2, 3})
      assert_tensor(Einops.rearrange(x, "b ... c -> c b ..."), Nx.transpose(x))
      assert_tensor(Einops.rearrange(Nx.tensor(9), "... -> (...)"), Nx.tensor([9]))
    end

    test "patchification is invertible" do
      x = Nx.iota({2, 4, 6, 3})
      patches = Einops.rearrange(x, "b (h ph) (w pw) c -> b (h w) (ph pw c)", ph: 2, pw: 3)
      restored = Einops.rearrange(patches, "b (h w) (ph pw c) -> b (h ph) (w pw) c", h: 2, ph: 2, pw: 3)
      assert Nx.shape(patches) == {2, 4, 18}
      assert_tensor(restored, x)
    end

    test "preserves vectorized axes" do
      x = Nx.iota({2, 3, 4}) |> Nx.vectorize(:ensemble)
      assert_tensor(Einops.rearrange(x, "h c -> c h"), Nx.transpose(x))
    end

    test "labels are positional and ordinary axis names are cleared" do
      x = Nx.iota({2, 3}, names: [:height, :width])
      y = Einops.rearrange(x, "width height -> height width")
      assert Nx.shape(y) == {3, 2}
      assert y.names == [nil, nil]
      assert Einops.rearrange(x, "a b -> a b").names == [nil, nil]
    end

    test "preserves integer dtype" do
      x = Nx.tensor([[1, 2], [3, 4]], type: :u8)
      assert Nx.type(Einops.rearrange(x, "a b -> (b a)")) == {:u, 8}
    end
  end

  describe "reduce" do
    for {reduction, function} <- [sum: :sum, mean: :mean, min: :reduce_min, max: :reduce_max, prod: :product, any: :any, all: :all] do
      @reduction reduction
      @function function
      test "#{reduction} agrees with its Nx operation" do
        x = Nx.tensor([[0.0, 2.0, 3.0], [-1.0, 4.0, 5.0]])
        assert_tensor(Einops.reduce(x, "b h -> b", @reduction), apply(Nx, @function, [x, [axes: [1]]]))
      end
    end

    test "reduces, permutes and merges" do
      x = Nx.iota({2, 3, 4, 5}, type: :f32)
      expected = x |> Nx.sum(axes: [1]) |> Nx.transpose(axes: [2, 0, 1]) |> Nx.reshape({10, 4})
      assert_tensor(Einops.reduce(x, "b h w c -> (c b) w", :sum), expected)
    end

    test "performs local mean pooling" do
      x = Nx.iota({2, 6, 8, 3}, type: :f32)
      expected = x |> Nx.reshape({2, 3, 2, 4, 2, 3}) |> Nx.mean(axes: [2, 4])
      assert_tensor(Einops.reduce(x, "b (h ph) (w pw) c -> b h w c", :mean, ph: 2, pw: 2), expected)
    end

    test "can reduce every dimension" do
      x = Nx.iota({2, 3}, type: :f32)
      assert_tensor(Einops.reduce(x, "b h ->", :sum), Nx.sum(x))
    end

    test "can reduce ellipsis dimensions" do
      x = Nx.iota({2, 3, 4}, type: :f32)
      assert_tensor(Einops.reduce(x, "b ... -> b", :mean), Nx.mean(x, axes: [1, 2]))
    end

    test "can insert singleton axes in its output" do
      x = Nx.iota({2, 3, 4}, type: :f32)
      assert_tensor(Einops.reduce(x, "b h c -> b 1 1", :mean), Nx.mean(x, axes: [1, 2], keep_axes: true))
    end

    test "does not apply the reducer when no elementary axes disappear" do
      x = Nx.tensor([-2, 0, 3])
      assert_tensor(Einops.reduce(x, "n -> n", :any), x)
    end

    test "preserves vectorization while reducing ordinary axes" do
      x = Nx.iota({2, 3, 4}, type: :f32) |> Nx.vectorize(:ensemble)
      assert_tensor(Einops.reduce(x, "h c -> c", :mean), Nx.mean(x, axes: [0]))
    end

    test "delegates dtype promotion to Nx" do
      x = Nx.tensor([[1, 2], [3, 4]], type: :u8)
      assert_tensor(Einops.reduce(x, "b h -> b", :sum), Nx.sum(x, axes: [1]))
    end
  end

  describe "repeat" do
    test "distinguishes element repetition from sequence tiling" do
      x = Nx.tensor([1, 2, 3])
      assert Nx.to_flat_list(Einops.repeat(x, "n -> (n copies)", copies: 2)) == [1, 1, 2, 2, 3, 3]
      assert Nx.to_flat_list(Einops.repeat(x, "n -> (copies n)", copies: 2)) == [1, 2, 3, 1, 2, 3]
    end

    test "can introduce multiple nontrailing axes" do
      x = Nx.iota({2, 3})
      expected = x |> Nx.transpose() |> Nx.broadcast({3, 4, 2, 5}, axes: [0, 2])
      assert_tensor(Einops.repeat(x, "b c -> c r b s", r: 4, s: 5), expected)
    end

    test "can split input dimensions before repeating" do
      x = Nx.iota({6})
      expected = x |> Nx.reshape({2, 3}) |> Nx.broadcast({2, 4, 3}, axes: [0, 2]) |> Nx.reshape({8, 3})
      assert_tensor(Einops.repeat(x, "(h w) -> (h copies) w", h: 2, copies: 4), expected)
    end

    test "repeats a scalar" do
      result = Einops.repeat(Nx.tensor(7), " -> rows columns", rows: 2, columns: 3)
      assert_tensor(result, Nx.broadcast(Nx.tensor(7), {2, 3}))
    end

    test "preserves vectorized axes" do
      x = Nx.iota({2, 3}) |> Nx.vectorize(:ensemble)
      assert_tensor(Einops.repeat(x, "n -> n r", r: 2), Nx.broadcast(x, {3, 2}, axes: [0]))
    end

    test "retains ellipsis and supports singleton insertion" do
      x = Nx.iota({2, 3})
      expected = Nx.broadcast(x, {2, 3, 2, 1}, axes: [0, 1])
      assert_tensor(Einops.repeat(x, "... -> ... copies 1", copies: 2), expected)
    end

    test "an introduced axis of length one still needs its length" do
      x = Nx.tensor([1, 2], type: :u8)
      assert_tensor(Einops.repeat(x, "n -> n copies", copies: 1), Nx.reshape(x, {2, 1}))
    end
  end

  describe "parse_shape" do
    test "extracts string-keyed dimension sizes" do
      assert Einops.parse_shape(Nx.iota({2, 3, 4}), "batch height channels") == %{"batch" => 2, "height" => 3, "channels" => 4}
    end

    test "reads a template without inspecting tensor contents" do
      assert Einops.parse_shape(Nx.template({2, 3}, :f32), "b c") == %{"b" => 2, "c" => 3}
    end

    test "permits repeated ignored axes" do
      assert Einops.parse_shape(Nx.iota({2, 3, 4}), "_ height _") == %{"height" => 3}
    end

    test "skips zero or more ellipsis dimensions" do
      assert Einops.parse_shape(Nx.iota({2, 3, 4, 5}), "batch ... channels") == %{"batch" => 2, "channels" => 5}
      assert Einops.parse_shape(Nx.iota({2, 5}), "batch ... channels") == %{"batch" => 2, "channels" => 5}
    end

    test "handles scalars and wholly ignored shapes" do
      assert Einops.parse_shape(Nx.tensor(1), "") == %{}
      assert Einops.parse_shape(Nx.tensor(1), "...") == %{}
      assert Einops.parse_shape(Nx.iota({2, 3}), "...") == %{}
    end

    test "reads only ordinary axes" do
      x = Nx.iota({2, 3, 4}) |> Nx.vectorize(:ensemble)
      assert Einops.parse_shape(x, "h c") == %{"h" => 3, "c" => 4}
    end

    test "its output can supply lengths to a transformation" do
      lengths = Einops.parse_shape(Nx.iota({3, 4}), "h w")
      x = Nx.iota({12})
      assert_tensor(Einops.rearrange(x, "(h w) -> h w", lengths), Nx.reshape(x, {3, 4}))
    end
  end

  @spec assert_tensor(Nx.Tensor.t(), Nx.Tensor.t()) :: :ok
  defp assert_tensor(actual, expected) do
    assert Nx.shape(actual) == Nx.shape(expected)
    assert Nx.type(actual) == Nx.type(expected)
    assert actual.vectorized_axes == expected.vectorized_axes
    assert Nx.to_flat_list(actual) == Nx.to_flat_list(expected)
    :ok
  end
end
