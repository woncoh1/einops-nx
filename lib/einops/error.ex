defmodule Einops.Error do
  @moduledoc "A pattern or shape error, with the original call's context."

  defexception [:message, :operation, :pattern, :shape, :axes, :reason]

  @type context :: %{
          operation: atom(),
          pattern: term(),
          shape: tuple() | :unknown,
          axes: term()
        }

  @doc false
  @spec fail!(context(), String.t()) :: no_return()
  def fail!(context, reason) do
    message = """
    #{reason}

    Operation: #{context.operation}
    Pattern:   #{inspect(context.pattern)}
    Shape:     #{inspect(context.shape)}
    Axes:      #{inspect(context.axes)}
    """

    raise __MODULE__,
      message: String.trim_trailing(message),
      operation: context.operation,
      pattern: context.pattern,
      shape: context.shape,
      axes: context.axes,
      reason: reason
  end
end
