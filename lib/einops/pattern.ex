defmodule Einops.Pattern do
  @moduledoc false
  alias Einops.Error

  # A group describes one physical dimension; [] is a singleton dimension.
  # Ellipsis IDs are tuples, so they cannot collide with user-supplied labels.
  @type axis :: String.t() | :ellipsis | :ignore | {:ellipsis, non_neg_integer()}
  @type group :: [axis()] | :ellipsis
  @type side :: [group()]
  @label ~r/\A[A-Za-z][A-Za-z0-9_]*\z/

  @spec label?(term()) :: boolean()
  def label?(name), do: is_binary(name) and Regex.match?(@label, name)

  @spec transformation(Error.context()) :: {side(), side()}
  def transformation(context) do
    validate_string!(context)

    case String.split(context.pattern, "->") do
      [left, right] ->
        expand(parse(left, context), parse(right, context), context)

      _ ->
        Error.fail!(context, "expected exactly one '->' separating input and output")
    end
  end

  @spec shape(Error.context()) :: %{optional(String.t()) => pos_integer()}
  def shape(context) do
    validate_string!(context)
    tokens = tokenize(context.pattern)

    if Enum.any?(tokens, &(&1 in ["(", ")", "1"])) do
      Error.fail!(context, "parse_shape accepts only axis labels, '_' and '...'")
    end

    {groups, _} = expand(parse(context.pattern, context), [], context)

    groups
    |> Enum.zip(Tuple.to_list(context.shape))
    |> Enum.reduce(%{}, fn
      {[name], size}, sizes when is_binary(name) -> Map.put(sizes, name, size)
      _, sizes -> sizes
    end)
  end

  @spec validate_string!(Error.context()) :: :ok
  defp validate_string!(context) do
    unless is_binary(context.pattern) and String.valid?(context.pattern) do
      Error.fail!(context, "expected a valid UTF-8 pattern string known while tracing")
    end

    :ok
  end

  @spec tokenize(String.t()) :: [String.t()]
  defp tokenize(text) do
    text |> String.replace(~r/([()])/, " \\1 ") |> String.split()
  end

  @spec parse(String.t(), Error.context()) :: side()
  defp parse(text, context) do
    groups = text |> tokenize() |> groups([], context)

    groups
    |> List.flatten()
    |> Enum.reject(&(&1 == :ignore))
    |> Enum.reduce(MapSet.new(), fn label, seen ->
      if MapSet.member?(seen, label) do
        Error.fail!(context, "duplicate axis #{inspect(label)} on the same side of the pattern")
      end

      MapSet.put(seen, label)
    end)

    groups
  end

  @spec groups([String.t()], side(), Error.context()) :: side()
  defp groups([], acc, _context), do: Enum.reverse(acc)

  defp groups(["(" | rest], acc, context) do
    {inside, tail} = Enum.split_while(rest, &(&1 != ")"))

    cond do
      "(" in inside -> Error.fail!(context, "nested axis groups are not supported")
      tail == [] -> Error.fail!(context, "unclosed '(' in the pattern")
      inside == [] -> Error.fail!(context, "empty groups are not supported; use '1' for a singleton")
      true -> :ok
    end

    [_closing | remaining] = tail
    group = Enum.flat_map(inside, &axis(&1, context))
    groups(remaining, [group | acc], context)
  end

  defp groups([")" | _], _acc, context) do
    Error.fail!(context, "unexpected ')' in the pattern")
  end

  defp groups(["..." | rest], acc, context), do: groups(rest, [:ellipsis | acc], context)

  defp groups([token | rest], acc, context) do
    groups(rest, [axis(token, context) | acc], context)
  end

  @spec axis(String.t(), Error.context()) :: [axis()]
  defp axis("1", _context), do: []
  defp axis("...", _context), do: [:ellipsis]
  defp axis("_", %{operation: :parse_shape}), do: [:ignore]

  defp axis(token, context) do
    if label?(token) do
      [token]
    else
      Error.fail!(
        context,
        "invalid token #{inspect(token)}; use labels such as 'batch' or 'patch_h', " <>
          "'1' for singleton axes, and named lengths instead of numeric factors; " <>
          "'_' is only allowed in parse_shape"
      )
    end
  end

  @spec expand(side(), side(), Error.context()) :: {side(), side()}
  defp expand(left, right, context) do
    if Enum.any?(left, &(is_list(&1) and :ellipsis in &1)) do
      Error.fail!(context, "'...' cannot appear inside an input group")
    end

    input_ellipsis? = :ellipsis in left
    output_ellipsis? = :ellipsis in List.flatten(right)

    if output_ellipsis? and not input_ellipsis? do
      Error.fail!(context, "output '...' requires '...' in the input pattern")
    end

    rank = tuple_size(context.shape)
    explicit = length(left) - if(input_ellipsis?, do: 1, else: 0)

    if (input_ellipsis? and rank < explicit) or (not input_ellipsis? and rank != explicit) do
      expected = if input_ellipsis?, do: "at least #{explicit}", else: "#{explicit}"
      Error.fail!(context, "rank mismatch: input pattern expects #{expected} dimensions, got #{rank}")
    end

    ids =
      List.duplicate(nil, rank - explicit)
      |> Enum.with_index(fn _, index -> {:ellipsis, index} end)

    {expand_side(left, ids), expand_side(right, ids)}
  end

  @spec expand_side(side(), [axis()]) :: side()
  defp expand_side(side, ids) do
    Enum.flat_map(side, fn
      :ellipsis -> Enum.map(ids, &[&1])
      group ->
        expanded = Enum.flat_map(group, fn
          :ellipsis -> ids
          label -> [label]
        end)

        [expanded]
    end)
  end
end
