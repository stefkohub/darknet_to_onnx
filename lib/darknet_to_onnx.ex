defmodule DarknetToOnnx do
  @moduledoc """
  Documentation for `DarknetToOnnx`.
  """

  @max_batch_size 1
  # @model "yolov3-tiny-416"
  # @weights_file_path "../yolov3-tiny-416.weights"
  # @cfg_file_path "../yolov3-tiny-416.cfg"
  # @output_path "yolov3-tiny-416.onnx"

  alias DarknetToOnnx.Helper, as: Helper

  def darknet_to_onnx(model, cfg_file_path, weights_file_path, output_path) do
    {result, parser_pid} = DarknetToOnnx.ParseDarknet.start_link([cfg_file_path: cfg_file_path])

    if result != :ok do
      {_msg_already_started, old_pid} = parser_pid
      Agent.stop(old_pid)
      {:ok, _parser_pid} = DarknetToOnnx.ParseDarknet.start_link([cfg_file_path: cfg_file_path])
    end

    parser_state = DarknetToOnnx.ParseDarknet.get_state()
    %{parse_result: layer_configs, keys: layer_configs_keys} = parser_state
    output_tensor_names = DarknetToOnnx.ParseDarknet.get_output_convs(parser_state)
    category_num = DarknetToOnnx.ParseDarknet.get_category_num(parser_state)
    c = (category_num + 5) * 3
    [h, w] = DarknetToOnnx.ParseDarknet.get_h_and_w(parser_state)

    output_tensor_shapes =
      case Enum.count(output_tensor_names) do
        2 -> [[c, div(h, 32), div(w, 32)], [c, div(h, 16), div(w, 16)]]
        3 -> [[c, div(h, 32), div(w, 32)], [c, div(h, 16), div(w, 16)], [c, div(h, 8), div(w, 8)]]
        4 -> [[c, div(h, 64), div(w, 64)], [c, div(h, 32), div(w, 32)], [c, div(h, 16), div(w, 16)], [c, div(h, 8), div(w, 8)]]
      end

    output_tensor_shapes =
      if DarknetToOnnx.ParseDarknet.is_pan_arch?(parser_state) do
        Enum.reverse(output_tensor_shapes)
      else
        output_tensor_shapes
      end

    output_tensor_dims = Map.new(Enum.zip(output_tensor_names, output_tensor_shapes))

    IO.puts("Building ONNX graph...")

    {result, gb_pid} =
      DarknetToOnnx.GraphBuilderONNX.start_link(model_name: model, output_tensors: output_tensor_dims, batch_size: @max_batch_size)

    if result != :ok do
      {_msg_already_started, old_pid} = gb_pid
      Agent.stop(old_pid)

      {:ok, _gb_pid} =
        DarknetToOnnx.GraphBuilderONNX.start_link(model_name: model, output_tensors: output_tensor_dims, batch_size: @max_batch_size)
    end

    builder = DarknetToOnnx.GraphBuilderONNX.get_state(gb_pid)

    yolo_model_def =
      DarknetToOnnx.GraphBuilderONNX.build_onnx_graph(
        builder,
        layer_configs,
        layer_configs_keys,
        weights_file_path,
        True
      )

    IO.puts("Saving ONNX file...")

    Helper.save_model(yolo_model_def, output_path)
    yolo_model_def
  end
end
