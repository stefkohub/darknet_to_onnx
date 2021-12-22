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
      deps: deps(),
####### From ElixirCI
      build_path: "_build",
      deps_path: "deps",
      lockfile: "mix.lock",
      consolidate_protocols: Mix.env() != :test,
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.html": :test
      ],
      dialyzer: [
        plt_add_apps: [:elixir, :mix],
        flags: [
          :error_handling,
          :race_conditions,
          :underspecs
        ],
        ignore_warnings: ".dialyzer_ignore.exs"
      ]
############
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
      {:credo, "~> 1.6.1", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.1", only: [:dev, :test], runtime: false},
      {:excoveralls, "~> 0.14", only: [:dev, :test], runtime: false},
      {:configparser_ex, git: "https://github.com/stefkohub/configparser_ex", tag: "master"},
      {:axon, "~> 0.1.0-dev", github: "elixir-nx/axon"},
      {:axon_onnx, github: "stefkohub/axon_onnx", tag: "master"},
      {:exla, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "exla"},
      # {:torchx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "torchx"},
      {:nx, "~> 0.1.0-dev", github: "elixir-nx/nx", sparse: "nx", override: true}
    ]
  end
end
