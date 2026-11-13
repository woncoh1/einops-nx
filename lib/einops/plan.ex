defmodule Einops.Plan do
  @moduledoc false
  alias Einops.{Error, Pattern}

  @type t :: %{
          expanded_shape: tuple(),
          reduction_axes: [non_neg_integer()],
          permutation: [non_neg_integer()],
          broadcast_shape: tuple(),
          broadcast_axes: [non_neg_integer()],
          output_shape: tuple()
        }

  @spec new(Error.context()) :: t()
  def new(context) do
    {left, right} = Pattern.transformation(context)
    input = List.flatten(left)
    output = List.flatten(right)
    sizes = axis_lengths!(context, input ++ output)

    sizes =
      left
      |> Enum.zip(Tuple.to_list(context.shape))
      |> Enum.with_index()
      |> Enum.reduce(sizes, fn {{group, dimension}, index}, sizes ->
        infer!(group, dimension, index, sizes, context)
      end)

    removed = input -- output
    added = output -- input
    validate_operation!(removed, added, sizes, context)
    kept = input -- removed
    reordered = Enum.filter(output, &(&1 in kept))

    %{
      expanded_shape: shape(input, sizes),
      reduction_axes: positions(removed, input),
      permutation: positions(reordered, kept),
      broadcast_shape: shape(output, sizes),
      broadcast_axes: positions(reordered, output),
      output_shape: right |> Enum.map(&product(&1, sizes)) |> List.to_tuple()
    }
  end

  @spec axis_lengths!(Error.context(), [Pattern.axis()]) :: map()
  defp axis_lengths!(context, labels) do
    pairs =
      cond do
        is_list(context.axes) and Keyword.keyword?(context.axes) ->
          Enum.map(context.axes, fn {key, value} -> {Atom.to_string(key), value} end)

        is_map(context.axes) and not is_struct(context.axes) ->
          Enum.sort(context.axes)

        true ->
          Error.fail!(context, "axis lengths must be a keyword list or a string-keyed map")
      end

    Enum.reduce(pairs, %{}, fn {name, size}, sizes ->
      cond do
        not Pattern.label?(name) ->
          Error.fail!(context, "invalid axis-length key #{inspect(name)}; map keys must be label strings")

        Map.has_key?(sizes, name) ->
          Error.fail!(context, "axis length #{inspect(name)} was supplied more than once")

        name not in labels ->
          Error.fail!(
            context,
            "unused axis length #{inspect(name)}; this label does not occur in the pattern"
          )

        not (is_integer(size) and size > 0) ->
          Error.fail!(
            context,
            "axis #{inspect(name)} needs a positive integer length known while tracing; got #{inspect(size)}"
          )

        true ->
          Map.put(sizes, name, size)
      end
    end)
  end

  @spec infer!([Pattern.axis()], pos_integer(), non_neg_integer(), map(), Error.context()) :: map()
  defp infer!(group, dimension, index, sizes, context) do
    unknown = Enum.reject(group, &Map.has_key?(sizes, &1))
    known_product = group |> Enum.reject(&(&1 in unknown)) |> product(sizes)
    group_text = if group == [], do: "1", else: Enum.map_join(group, " ", &display/1)
    description = "input axis #{index} of length #{dimension} (#{group_text})"

    case unknown do
      [] ->
        if known_product != dimension do
          Error.fail!(
            context,
            "shape mismatch for #{description}: the declared lengths multiply to #{known_product}"
          )
        end

        sizes

      [name] ->
        if rem(dimension, known_product) != 0 do
          Error.fail!(
            context,
            "cannot split #{description}: #{dimension} is not divisible by #{known_product}"
          )
        end

        Map.put(sizes, name, div(dimension, known_product))

      names ->
        Error.fail!(
          context,
          "cannot infer #{Enum.map_join(names, ", ", &display/1)} from #{description}; " <>
            "supply all but one length in each input group"
        )
    end
  end

  @spec validate_operation!([Pattern.axis()], [Pattern.axis()], map(), Error.context()) :: :ok
  defp validate_operation!(removed, added, sizes, context) do
    if removed != [] and context.operation != :reduce do
      Error.fail!(
        context,
        "#{context.operation} cannot remove axes #{inspect(removed)}; use reduce to aggregate them"
      )
    end

    if added != [] and context.operation != :repeat do
      Error.fail!(
        context,
        "#{context.operation} cannot introduce axes #{inspect(added)}; use repeat, or '1' for a singleton"
      )
    end

    for name <- added, not Map.has_key?(sizes, name) do
      Error.fail!(
        context,
        "new axis #{inspect(name)} needs an explicit length, for example #{display(name)}: 2"
      )
    end

    :ok
  end

  @spec positions([Pattern.axis()], [Pattern.axis()]) :: [non_neg_integer()]
  defp positions(labels, order) do
    indexes = order |> Enum.with_index() |> Map.new()
    Enum.map(labels, &Map.fetch!(indexes, &1))
  end

  @spec shape([Pattern.axis()], map()) :: tuple()
  defp shape(labels, sizes), do: labels |> Enum.map(&Map.fetch!(sizes, &1)) |> List.to_tuple()

  @spec product([Pattern.axis()], map()) :: pos_integer()
  defp product(labels, sizes), do: Enum.reduce(labels, 1, &(Map.fetch!(sizes, &1) * &2))

  @spec display(Pattern.axis()) :: String.t()
  defp display({:ellipsis, index}), do: "...[#{index}]"
  defp display(label), do: to_string(label)
end
