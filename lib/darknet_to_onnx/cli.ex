defmodule DarknetToOnnx.CLI do
  def main(args) do
    args |> parse_args |> do_process
  end

  def parse_args(args) do
    optparseargs = [
      aliases: [m: :model, h: :help, r: :force_raw],
      strict: [model: :string, help: :boolean, force_raw: :boolean]
    ]

    options = [force_raw: false]

    {opts, args, invalid} = OptionParser.parse(args, 
      optparseargs
    )

    if args != [] or invalid != [] do
      raise "Error in command line. Wrong arguments " <> inspect([args, invalid])
      System.halt(1)
    end

    case opts do
      [help: true] -> :help
      opt -> Keyword.merge(options, opt)
    end
    #cond do
    #  Keyword.get(opts, :help) -> :help
    #  Keyword.get(opts, :model) -> Keyword.get(opts, :model)
    #  Keyword.get(opts, :force_raw) -> :force_raw
    #  true -> :help
    #end
  end

  def do_process(:help) do
    IO.puts("""
    optional arguments:
    -h, --help          show this help message and exit
    -r, --force_raw     Force writing raw data instead of converting to actual type. In Elixir it boosts performances.
    -m MODEL, --model MODEL
                        [yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|yolov4-csp|yolov4x-mish]-[{dimension}],
                        where {dimension} could be either a single number (e.g. 288, 416, 608) or 2 numbers, WxH (e.g.
                        416x256)
    """)

    System.halt(0)
  end

  def do_process(args) do
    model = Keyword.get(args, :model)
    force_raw = Keyword.get(args, :force_raw)
    [_mname | rest] = String.split(Path.basename(model), "-", parts: 3)

    if rest == [] or String.match?(Enum.at(rest, -1), ~r/^[[:digit:]]+/u) == False or
         !String.starts_with?(
           Path.basename(model),
           ["yolov3-tiny", "yolov3", "yolov3-spp", "yolov4-tiny", "yolov4", "yolov4-csp", "yolov4x-mish"]
         ) do
      raise "Wrong model. It muse be one of: [yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|yolov4-csp|yolov4x-mish]-{dimension}, where {dimension} could be either a single number (e.g. 288, 416, 608) or 2 numbers, WxH (e.g. 416x256)"
    else
      if File.exists?(model <> ".cfg") and File.exists?(model <> ".weights") do
        if File.exists?(model <> ".onnx") do
          IO.puts("Overwriting previous ONNX model file: " <> model <> ".onnx")
        end

        DarknetToOnnx.darknet_to_onnx(model, model <> ".cfg", model <> ".weights", model <> ".onnx", force_raw)
      else
        raise "Model files doesn't exists: " <> model <> "[.cfg, .weights]"
      end
    end
  end
end
