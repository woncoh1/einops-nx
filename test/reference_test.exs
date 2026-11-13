defmodule Einops.ReferenceTest do
  use ExUnit.Case, async: true

  @fixture_path Path.join(__DIR__, "fixtures/reference_cases.exs")
  @external_resource @fixture_path
  @cases elem(Code.eval_file(@fixture_path), 0)

  for {example, index} <- Enum.with_index(@cases) do
    @example example
    test "Python reference #{index}: #{example.operation} #{example.pattern} #{example.reduction}" do
      example = @example
      x = example.input |> Nx.tensor(type: :f32) |> Nx.reshape(example.shape)

      result =
        case example.operation do
          :rearrange -> Einops.rearrange(x, example.pattern, example.axes)
          :repeat -> Einops.repeat(x, example.pattern, example.axes)
          :reduce -> Einops.reduce(x, example.pattern, example.reduction, example.axes)
        end

      expected = example.expected |> Nx.tensor(type: :f32) |> Nx.reshape(example.expected_shape)
      assert Nx.shape(result) == example.expected_shape
      assert Nx.to_number(Nx.all_close(result, expected, atol: 1.0e-5, rtol: 1.0e-5)) == 1
    end
  end
end
