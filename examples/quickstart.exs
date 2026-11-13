x = Nx.iota({2, 3, 4}, type: :f32)

IO.inspect(Einops.rearrange(x, "b h c -> b c h"), label: "Transpose")
IO.inspect(Einops.reduce(x, "b h c -> b c", :mean), label: "Mean over height")
IO.inspect(Einops.repeat(x, "b h c -> b h c copies", copies: 2), label: "Repeat")
IO.inspect(Einops.parse_shape(x, "batch _ channels"), label: "Dimensions")

defmodule EinopsQuickstart do
  import Nx.Defn

  @spec split_heads(Nx.Tensor.t()) :: Nx.Tensor.t()
  defn split_heads(x) do
    Einops.rearrange(x, "b tokens (heads d) -> b heads tokens d", heads: 2)
  end
end

IO.inspect(EinopsQuickstart.split_heads(x), label: "Inside defn")

try do
  Einops.rearrange(x, "b h (heads d) -> b heads h d", heads: 3)
rescue
  error in Einops.Error -> IO.puts(Exception.message(error))
end
