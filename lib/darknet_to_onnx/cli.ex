defmodule DarknetToOnnx.CLI do
  def main(args) do
    args |> parse_args |> do_process
  end

  def parse_args(args) do
    optparseargs = [
      aliases: [m: :model, h: :help],
      strict: [model: :string, help: :boolean]
    ]

    {opts, args, invalid} = OptionParser.parse(args, optparseargs)

    if args != [] or invalid != [] do
      raise "Error in command line. Wrong arguments " <> inspect([args, invalid])
      System.halt(1)
    end

    cond do
      Keyword.get(opts, :help) -> :help
      Keyword.get(opts, :model) -> Keyword.get(opts, :model)
      true -> :help
    end
  end

  def do_process(:help) do
    IO.puts("""
    optional arguments:
    -h, --help          show this help message and exit
    -m MODEL, --model MODEL
                        [yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|yolov4-csp|yolov4x-mish]-[{dimension}],
                        where {dimension} could be either a single number (e.g. 288, 416, 608) or 2 numbers, WxH (e.g.
                        416x256)
    """)

    System.halt(0)
  end

  def do_process(model) do
    [_mname | rest] = String.split(Path.basename(model), "-", parts: 3)

    if String.match?(Enum.at(rest, -1), ~r/^[[:digit:]]+/u) == False or
         !String.starts_with?(
           Path.basename(model),
           ["yolov3-tiny", "yolov3", "yolov3-spp", "yolov4-tiny", "yolov4", "yolov4-csp", "yolov4x-mish"]
         ) do
      raise "Wrong model. It muse be one of: [yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|yolov4-csp|yolov4x-mish]-{dimension}, where {dimension} could be either a single number (e.g. 288, 416, 608) or 2 numbers, WxH (e.g. 416x256)"
    else
      if File.exists?(model<>".cfg") and File.exists?(model<>".weights") do
        if File.exists?(model<>".onnx") do
          IO.puts("Overwriting previous ONNX model file: "<>model<>".onnx")
        end
        DarknetToOnnx.darknet_to_onnx(model, model <> ".cfg", model <> ".weights", model <> ".onnx")
      else
        raise "Model files doesn't exists: "<>model<>"[.cfg, .model]"
      end
    end
  end
end
