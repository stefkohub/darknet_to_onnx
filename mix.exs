defmodule DarknetToOnnx.MixProject do
  use Mix.Project

  def project do
    [
      app: :darknet_to_onnx,
      version: "0.1.0",
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      build_embedded: Mix.env() == :prod,
      escript: [main_module: DarknetToOnnx.CLI],
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :exla]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # {:dep_from_hexpm, "~> 0.3.0"},
      # {:dep_from_git, git: "https://github.com/elixir-lang/my_dep.git", tag: "0.1.0"}
      # {:configparser_ex, git: "https://github.com/easco/configparser_ex", tag: "master"},
      # {:configparser_ex, git: "https://github.com/stefkohub/configparser_ex", tag: "master"},
      {:configparser_ex, github: "stefkohub/configparser_ex", tag: "master"},
      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon"},
      {:axon_onnx, git: "https://github.com/elixir-nx/axon_onnx", tag: "master"},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
      # {:torchx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "torchx"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true}
    ]
  end
end
