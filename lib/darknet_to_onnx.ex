defmodule DarknetToOnnx do
  @moduledoc """
  Documentation for `DarknetToOnnx`.
  """

  @max_batch_size 1
  @model "yolov3-tiny"

  def main() do
    {:ok, parser_pid} = DarknetToOnnx.ParseDarknet.start_link([cfg_file_path: "../yolov3-tiny.cfg"], [])
    
    parser_state = DarknetToOnnx.ParseDarknet.get_state()
    layer_configs = parser_state.parse_result
    output_tensor_names = DarknetToOnnx.ParseDarknet.get_output_convs(parser_state)
    category_num = DarknetToOnnx.ParseDarknet.get_category_num(parser_state)
    c = (category_num + 5) * 3
    [h, w]=DarknetToOnnx.ParseDarknet.get_h_and_w(parser_state)
    output_tensor_shapes = case Enum.count(output_tensor_names) do
      2 -> [[c, div(h, 32), div(w, 32)], [c, div(h, 16), div(w, 16)]]
      3 -> [[c, div(h, 32), div(w, 32)], [c, div(h, 16), div(w, 16)],
                                   [c, div(h, 8), div(w, 8) ]]
      4 -> [[c, div(h, 64), div(w, 64)], [c, div(h, 32), div(w, 32)],
                                   [c, div(h, 16), div(w, 16)], [c, div(h, 8), div(w, 8)]]
    end
    if DarknetToOnnx.ParseDarknet.is_pan_arch?(parser_state) do
      output_tensor_shapes=Enum.reverse(output_tensor_shapes)
    end
    output_tensor_dims=Enum.zip(output_tensor_names, output_tensor_shapes)

    IO.puts "Building ONNX graph..."
    #builder = DarknetToOnnx.GraphBuilderONNX(
    #  @model, output_tensor_dims, @max_batch_size)
    #yolo_model_def = builder.build_onnx_graph(
    #    layer_configs=layer_configs,
    #    weights_file_path=weights_file_path,
    #    verbose=True)
  end
end
