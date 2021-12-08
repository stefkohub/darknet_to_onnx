defmodule DarknetToOnnx.WeightLoader do
  @moduledoc """
      Helper class used for loading the serialized weights of a binary file stream
    and returning the initializers and the input tensors required for populating
    the ONNX graph with weights.
  """

  use Agent, restart: :temporary

  alias DarknetToOnnx.Learning, as: Utils
  alias DarknetToOnnx.Helper, as: Helper

  @doc """
        Initialized with a path to the YOLO .weights file.
        Keyword argument:
        weights_file_path -- path to the weights file.
  """
  def start_link(opts) do
    initial_state = %{
      weights_file: open_weights_file(Keyword.fetch!(opts, :weights_file))
    }

    {:ok, _pid} = Agent.start_link(fn -> initial_state end, name: __MODULE__)
    initial_state
  end

  def get_state() do
    Agent.get(__MODULE__, fn state -> state end)
  end

  def update_state(key, value) do
    Agent.update(__MODULE__, fn state -> %{state | key => value} end)
  end

  def stop_link() do
    Agent.stop(__MODULE__)
  end

  @doc """
        Opens a YOLO DarkNet file stream and skips the header.
        Keyword argument:
        weights_file_path -- path to the weights file.
  """
  def open_weights_file(weights_file_path) do
    # Here we skip the first 20 bytes of this file. I don't know why they are assigning this header to an
    # np.array since they are not using it anymore...

    {:ok, weights_file} = File.open(weights_file_path, [:read, :binary])
    # Move the file pointer after the header
    IO.binread(weights_file, 5 * 4)
    # param_data = [Nx.from_binary(header, {:s, 32})
    weights_file
  end

  @doc """
        Deserializes the weights from a file stream in the DarkNet order.
        Keyword arguments:
        conv_params -- a ConvParams object
        param_category -- the category of parameters to be created ('bn' or 'conv')
        suffix -- a string determining the sub-type of above param_category (e.g.,
        'weights' or 'bias')
  """
  def load_one_param_type(conv_params, param_category, suffix) do
    param_name = DarknetToOnnx.ConvParams.generate_param_name(conv_params.node_name, param_category, suffix)
    [channels_out, channels_in, filter_h, filter_w] = conv_params.conv_weight_dims

    param_shape =
      case param_category do
        "bn" ->
          {channels_out}
        "conv" ->
          case suffix do
            "weights" -> {channels_out, channels_in, filter_h, filter_w}
            "bias" -> {channels_out}
          end
      end

    param_size = Enum.reduce(Tuple.to_list(param_shape), 1, fn val, acc -> acc * val end)
    %{weights_file: weights_file} = get_state()
    param_data = IO.binread(weights_file, param_size * 4)
    update_state(:weights_file, weights_file)
    [param_name, param_data, param_shape]
  end

  @doc """
        Creates the initializers with weights from the weights file together with
        the input tensors.
        Keyword arguments:
        conv_params -- a ConvParams object
        param_category -- the category of parameters to be created ('bn' or 'conv')
        suffix -- a string determining the sub-type of above param_category (e.g.,
        'weights' or 'bias')
  """
  def create_param_tensors(conv_params, param_category, suffix) do
    [param_name, param_data, param_data_shape] = load_one_param_type(conv_params, param_category, suffix)

    initializer_tensor =
      DarknetToOnnx.Helper.make_tensor(
        param_name,
        1,
        param_data_shape,
        param_data
      )

    input_tensor =
      Helper.make_tensor_value_info(
        param_name,
        1,
        param_data_shape
      )

    [initializer_tensor, input_tensor]
  end

  @doc """
        Returns the initializers with weights from the weights file and
        the input tensors of a convolutional layer for all corresponding ONNX nodes.
        Keyword argument:
        conv_params -- a ConvParams object
  """
  def load_conv_weights(conv_params) do
    [init, input] =
      if conv_params.batch_normalize != nil and conv_params.batch_normalize == True do
        [bias_init, bias_input] = create_param_tensors(conv_params, "bn", "bias")
        [bn_scale_init, bn_scale_input] = create_param_tensors(conv_params, "bn", "scale")
        [bn_mean_init, bn_mean_input] = create_param_tensors(conv_params, "bn", "mean")
        [bn_var_init, bn_var_input] = create_param_tensors(conv_params, "bn", "var")

        [
          [bn_scale_init, bias_init, bn_mean_init, bn_var_init],
          [bn_scale_input, bias_input, bn_mean_input, bn_var_input]
        ]
      else
        [bias_init, bias_input] = create_param_tensors(conv_params, "conv", "bias")
        [
          [bias_init],
          [bias_input]
        ]
      end

    [conv_init, conv_input] = create_param_tensors(conv_params, "conv", "weights")
    [Utils.cfl(init, conv_init), Utils.cfl(input, conv_input)]
  end

  @doc """
        Returns the initializers with the value of the scale input
        tensor given by upsample_params.
        Keyword argument:
        upsample_params -- a UpsampleParams object
  """
  def load_upsample_scales(upsample_params) do
    upsample_state = DarknetToOnnx.UpsampleParams.get_state(upsample_params.node_name)
    name = DarknetToOnnx.UpsampleParams.generate_param_name(upsample_state)

    scale_init =
      Helper.make_tensor(
        name,
        1,
        upsample_state.value.shape,
        upsample_state.value.data.state
      )

    scale_input =
      Helper.make_tensor_value_info(
        name,
        1,
        upsample_state.value.shape
      )

    [[scale_init], [scale_input]]
  end
end
