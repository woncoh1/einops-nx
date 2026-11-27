defmodule Einops.MixProject do
  use Mix.Project

  @spec project() :: keyword()
  def project do
    [
      app: :einops,
      version: "0.1.0-dev",
      elixir: "~> 1.17",
      deps: [{:nx, "~> 1.0.0"}],
      test_ignore_filters: ["test/fixtures/reference_cases.exs"],
      description: "Small, pattern-based tensor transformations for Nx"
    ]
  end

  @spec application() :: keyword()
  def application, do: []
end
