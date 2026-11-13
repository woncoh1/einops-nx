defmodule Einops do
  @moduledoc """
  Pattern-based tensor transformations for Nx.

  Patterns address ordinary axes, not vectorized axes or Nx dimension names.
  Results have unnamed ordinary axes; vectorized axes are preserved.
  Labels are local strings and never converted into atoms.

  The four functions are `Nx.Defn` transforms. Patterns and supplied lengths
  must be known while tracing. Tensor values are never inspected on the host.
  """

  import Nx.Defn, only: [deftransform: 2]
  alias Einops.{Error, Pattern, Plan}

  @type axis_lengths :: keyword(pos_integer()) | %{optional(String.t()) => pos_integer()}
  @type reduction :: :sum | :mean | :min | :max | :prod | :any | :all

  @reductions %{
    sum: :sum,
    mean: :mean,
    min: :reduce_min,
    max: :reduce_max,
    prod: :product,
    any: :any,
    all: :all
  }

  @doc """
  Reorders, splits, merges, inserts or removes singleton dimensions.

      iex> x = Nx.iota({2, 3, 4})
      iex> Nx.shape(Einops.rearrange(x, "b h c -> b (h c)"))
      {2, 12}
      iex> Nx.shape(Einops.rearrange(x, "b h (heads d) -> b heads h d", heads: 2))
      {2, 2, 3, 2}
  """
  @spec rearrange(Nx.Tensor.t(), String.t(), axis_lengths()) :: Nx.Tensor.t()
  deftransform rearrange(tensor, pattern, axes \\ []) do
    run(tensor, context!(tensor, :rearrange, pattern, axes), nil)
  end

  @doc """
  Aggregates input axes absent from the output using an Nx reduction.

  Accepts `:sum`, `:mean`, `:min`, `:max`, `:prod`, `:any`, or `:all`.
  Dtype promotion and numerical behavior follow the corresponding Nx operation.
  With no eliminated elementary axes, only rearrangement is performed.

      iex> x = Nx.iota({2, 3}, type: :f32)
      iex> Einops.reduce(x, "b h -> b", :mean) |> Nx.to_flat_list()
      [1.0, 4.0]
  """
  @spec reduce(Nx.Tensor.t(), String.t(), reduction(), axis_lengths()) :: Nx.Tensor.t()
  deftransform reduce(tensor, pattern, reduction, axes \\ []) do
    context = context!(tensor, :reduce, pattern, axes)

    unless Map.has_key?(@reductions, reduction) do
      Error.fail!(
        context,
        "unsupported reduction #{inspect(reduction)}; expected :sum, :mean, :min, :max, :prod, :any or :all"
      )
    end

    run(tensor, context, reduction)
  end

  @doc """
  Introduces repeated axes. Every new named axis needs an explicit length.

      iex> x = Nx.tensor([1, 2, 3])
      iex> Einops.repeat(x, "n -> (n copies)", copies: 2) |> Nx.to_flat_list()
      [1, 1, 2, 2, 3, 3]
      iex> Einops.repeat(x, "n -> (copies n)", copies: 2) |> Nx.to_flat_list()
      [1, 2, 3, 1, 2, 3]
  """
  @spec repeat(Nx.Tensor.t(), String.t(), axis_lengths()) :: Nx.Tensor.t()
  deftransform repeat(tensor, pattern, axes \\ []) do
    run(tensor, context!(tensor, :repeat, pattern, axes), nil)
  end

  @doc """
  Returns a string-keyed map of ordinary dimension sizes.

  Use `_` to skip one dimension or `...` to skip zero or more dimensions.
  Groups and singleton literals are not accepted. Inside `defn`, consume the
  returned host-side metadata in a transform; it is not a tensor result.

      iex> Einops.parse_shape(Nx.iota({2, 3, 4}), "batch _ channels")
      %{"batch" => 2, "channels" => 4}
  """
  @spec parse_shape(Nx.Tensor.t(), String.t()) :: %{optional(String.t()) => pos_integer()}
  deftransform parse_shape(tensor, pattern) do
    tensor |> context!(:parse_shape, pattern, []) |> Pattern.shape()
  end

  @spec context!(term(), atom(), term(), term()) :: Error.context()
  defp context!(tensor, operation, pattern, axes) do
    shape = if match?(%Nx.Tensor{}, tensor), do: tensor.shape, else: :unknown
    context = %{operation: operation, pattern: pattern, shape: shape, axes: axes}

    if shape == :unknown do
      Error.fail!(context, "expected an Nx.Tensor; use Nx.tensor/1 to convert a number or list")
    end

    context
  end

  @spec run(Nx.Tensor.t(), Error.context(), reduction() | nil) :: Nx.Tensor.t()
  defp run(tensor, context, reduction) do
    plan = Plan.new(context)
    tensor = Nx.reshape(tensor, plan.expanded_shape)

    tensor =
      if plan.reduction_axes == [] do
        tensor
      else
        apply(Nx, Map.fetch!(@reductions, reduction), [tensor, [axes: plan.reduction_axes]])
      end

    tensor = Nx.transpose(tensor, axes: plan.permutation)

    tensor =
      if context.operation == :repeat do
        Nx.broadcast(tensor, plan.broadcast_shape, axes: plan.broadcast_axes)
      else
        tensor
      end

    Nx.reshape(tensor, plan.output_shape)
  end
end
