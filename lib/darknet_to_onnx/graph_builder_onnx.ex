defmodule DarknetToOnnx.GraphBuilderONNX do
  @moduledoc """
      Class for creating an ONNX graph from a previously generated list of layer dictionaries.
  """

  use Agent, restart: :transient

  alias DarknetToOnnx.Learning, as: Utils
  alias DarknetToOnnx.Helper, as: Helper
  # alias Onnx.ModelProto, as: Model
  # alias Onnx.GraphProto, as: Graph
  # alias Onnx.NodeProto, as: Node
  # alias Onnx.ValueInfoProto, as: Value
  # alias Onnx.AttributeProto, as: Attribute
  # alias Onnx.OperatorSetIdProto, as: Opset
  # alias Onnx.TypeProto, as: Type
  # NON MI PIACE alias Onnx.TypeProto.Tensor, as: Placeholder
  # alias Onnx.TypeProto.Tensor, as: TensorProto
  # alias Onnx.TensorShapeProto, as: Shape
  # alias Onnx.TensorShapeProto.Dimension, as: Dimension

  @doc """
        Initialize with all DarkNet default parameters used creating
        YOLO, and specify the output tensors as an OrderedDict for their
        output dimensions with their names as keys.
        Keyword argument:
        output_tensors -- the output tensors as an OrderedDict containing the keys'
        output dimensions
  """
  def start_link(opts) do
    initial_state = %{
      model_name: Keyword.fetch!(opts, :model_name),
      output_tensors: Keyword.fetch!(opts, :output_tensors),
      nodes: [],
      graph_def: nil,
      input_tensor: nil,
      epsilon_bn: 1.0e-5,
      momentum_bn: 0.99,
      alpha_lrelu: 0.1,
      param_dict: [],
      major_node_specs: [],
      batch_size: Keyword.fetch!(opts, :batch_size),
      route_spec: 0
    }

    Agent.start_link(fn -> initial_state end, name: __MODULE__)
  end

  def get_state(_pid) do
    Agent.get(__MODULE__, fn state -> state end)
  end

  @doc """
        Create an ONNX input tensor from a 'net' layer and store the batch size.
        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
  """

  def make_input_tensor(state, layer_name, layer_dict) do
    channels = layer_dict["channels"]
    height = layer_dict["height"]
    width = layer_dict["width"]

    # input_tensor =
    #  Nx.tensor(
    #    [state.batch_size, channels, height, width],
    #    names: [String.to_atom(layer_name)],
    #    type: {:f, 64}
    #  )

    input_tensor =
      Helper.make_tensor_value_info(
        layer_name,
        # {:f, 32},
        1,
        [state.batch_size, channels, height, width]
      )

    state = %{state | input_tensor: input_tensor}
    [state, layer_name, channels]
  end

  @doc """
    mah...
  """
  def majorNodeSpecs(name, channels) do
    %{
      :name => name,
      :channels => channels,
      :created_onnx_node =>
        if name != nil and is_integer(channels) and channels > 0 do
          true
        else
          false
        end
    }
  end

  @doc """
    Get a previously ONNX node.
    Target index can be passed for jumping to a specific index.
    Keyword arguments:
    target_index -- optional for jumping to a specific index,
                    default: 0 for the previous element, while
                    taking 'route' spec into account
  """
  def get_previous_node_specs(state, target_index \\ 0) do
    if target_index == 0 do
      if state.route_spec != 0 do
        # TODO: assert 'dummy' not in previous_node.name
        [%{state | route_spec: 0}, Enum.at(state.major_node_specs, state.route_spec)]
      else
        [state, Enum.at(state.major_node_specs, -1)]
      end
    else
      [state, Enum.at(state.major_node_specs, target_index)]
    end

    # TODO: assert previous_node.created_onnx_node
  end

  defp make_conv_node_batch_normalize(state, inputs, conv_params_state, layer_name_output)
       when conv_params_state.batch_normalize == true do
    inputs =
      Utils.cfl(
        inputs,
        for suffix <- ["scale", "bias", "mean", "var"] do
          DarknetToOnnx.ConvParams.generate_param_name(conv_params_state, "bn", suffix)
        end
      )

    Helper.make_node("BatchNormalization", inputs, [layer_name_output <> "_bn"], layer_name_output <> "_bn", %{
      epsilon: state.epsilon_bn,
      momentum: state.momentum_bn
    })
  end

  def make_conv_node_leaky_relu(state, inputs, _conv_params_state, layer_name) do
    Helper.make_node("LeakyRelu", inputs, [layer_name <> "_lrelu"], layer_name <> "_lrelu", %{
      alpha: state.alpha_lrelu
    })
  end

  def make_conv_node_mish(_state, inputs, _conv_params_state, layer_name) do
    layer_name_softplus = layer_name <> "_softplus"
    layer_name_tanh = layer_name <> "_tanh"
    layer_name_mish = layer_name <> "_mish"

    [
      Helper.make_node("Softplus", inputs, [layer_name_softplus], layer_name_softplus),
      Helper.make_node("Tanh", [layer_name_softplus], [layer_name_tanh], layer_name_tanh),
      Helper.make_node("Mul", Utils.cfl(inputs, layer_name_tanh), [layer_name_mish], layer_name_mish)
    ]
  end

  def make_conv_node_swish(_state, inputs, _conv_params_state, layer_name) do
    layer_name_sigmoid = layer_name <> "_sigmoid"
    layer_name_swish = layer_name <> "_swish"

    [
      Helper.make_node("Sigmoid", inputs, [layer_name_sigmoid], layer_name_sigmoid),
      Helper.make_node("Mul", Utils.cfl(inputs, layer_name_sigmoid), [layer_name_swish], layer_name_swish)
    ]
  end

  def make_conv_node_logistic(_state, inputs, _conv_params_state, layer_name) do
    Helper.make_node("Sigmoid", inputs, [layer_name <> "_lgx"], layer_name <> "_lgx")
  end

  @doc """
        Create an ONNX Conv node with optional batch normalization and
        activation nodes.
        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
  """
  def make_conv_node(state, layer_name, layer_dict) do
    [state, previous_node_specs] = get_previous_node_specs(state)
    # inputs = [previous_node_specs.name]
    kernel_shape = [layer_dict["size"], layer_dict["size"]]
    weights_shape = Utils.cfl([layer_dict["filters"], previous_node_specs.channels], kernel_shape)

    conv_params_state =
      DarknetToOnnx.ConvParams.start_link(
        node_name: layer_name,
        batch_normalize:
          if layer_dict["batch_normalize"] != nil and layer_dict["batch_normalize"] > 0 do
            true
          else
            false
          end,
        conv_weight_dims: weights_shape
      )

    strides = [layer_dict["stride"], layer_dict["stride"]]
    dilations = [1, 1]
    weights_name = DarknetToOnnx.ConvParams.generate_param_name(conv_params_state, "conv", "weights")

    inputs =
      if conv_params_state.batch_normalize != true do
        Utils.cfl([], [
          previous_node_specs.name,
          weights_name,
          DarknetToOnnx.ConvParams.generate_param_name(conv_params_state, "conv", "bias")
        ])
      else
        Utils.cfl([], [previous_node_specs.name, weights_name])
      end

    conv_node =
      Helper.make_node("Conv", inputs, [layer_name], layer_name, %{
        kernel_shape: kernel_shape,
        strides: strides,
        auto_pad: "SAME_LOWER",
        dilations: dilations
      })

    state = %{state | nodes: Utils.cfl(state.nodes, conv_node)}
    layer_name_output = layer_name
    inputs = [layer_name]

    [state, layer_name_output, inputs] =
      if conv_params_state.batch_normalize == true do
        new_nodes = make_conv_node_batch_normalize(state, inputs, conv_params_state, layer_name)

        [
          %{state | nodes: Utils.cfl(state.nodes, new_nodes)},
          layer_name <> "_bn",
          [layer_name <> "_bn"]
        ]
      else
        [state, layer_name_output, inputs]
      end

    [state, layer_name_output] =
      case layer_dict["activation"] do
        "leaky" ->
          [
            %{state | nodes: Utils.cfl(state.nodes, [make_conv_node_leaky_relu(state, inputs, conv_params_state, layer_name)])},
            layer_name <> "_lrelu"
          ]

        "mish" ->
          [
            %{state | nodes: Utils.cfl(state.nodes, [make_conv_node_mish(state, inputs, conv_params_state, layer_name)])},
            layer_name <> "_mish"
          ]

        "swish" ->
          [
            %{state | nodes: Utils.cfl(state.nodes, [make_conv_node_swish(state, inputs, conv_params_state, layer_name)])},
            layer_name <> "_swish"
          ]

        "logistic" ->
          [
            %{state | nodes: Utils.cfl(state.nodes, [make_conv_node_logistic(state, inputs, conv_params_state, layer_name)])},
            layer_name <> "_lgx"
          ]

        "linear" ->
          [state, layer_name_output]

        _ ->
          raise "Unsupported layer_dict activation type: " <> inspect(layer_dict["activation"])
      end

    state = %{state | param_dict: Utils.cfl(state.param_dict, %{layer_name => conv_params_state})}
    [state, layer_name_output, layer_dict["filters"]]
  end

  @doc """
        Create an ONNX Maxpool node with the properties from
        the DarkNet-based graph.
        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
  """
  def make_maxpool_node(state, layer_name, layer_dict) do
    [state, previous_node_specs] = get_previous_node_specs(state)

    [
      %{
        state
        | nodes:
            Utils.cfl(
              state.nodes,
              [
                Helper.make_node("MaxPool", [previous_node_specs.name], [layer_name], layer_name, %{
                  kernel_shape: [layer_dict["size"], layer_dict["size"]],
                  auto_pad: "SAME_UPPER",
                  strides: [layer_dict["stride"], layer_dict["stride"]]
                })
              ]
            )
      },
      layer_name,
      previous_node_specs.channels
    ]
  end

  @doc """
        Create an ONNX Add node with the shortcut properties from
        the DarkNet-based graph.
        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)

  """
  def make_shortcut_node(state, layer_name, layer_dict) do
    if !layer_dict["activation"] === "linear" do
      raise "Wrong activation for shortcut node"
    end

    [state, first_node_specs] = get_previous_node_specs(state)
    [state, second_node_specs] = get_previous_node_specs(state, layer_dict["from"])
    # if first_node_specs != second_node_specs do
    #  raise "Wrong shortcut node configuration: "<>inspect([first_node_specs.name, second_node_specs.name])
    # end

    [
      %{
        state
        | nodes:
            Utils.cfl(
              state.nodes,
              [
                Helper.make_node(
                  "Add",
                  [first_node_specs.name, second_node_specs.name],
                  [layer_name],
                  layer_name
                )
              ]
            )
      },
      layer_name,
      first_node_specs.channels
    ]
  end

  def inner_make_route_node(state, layer_name, layer_dict, layer_dict_layers) when layer_dict_layers == 1 do
    if "groups" in Map.keys(layer_dict) do
      if "group_id" not in Map.keys(layer_dict) do
        raise "Wrong group layer. Missing group_id"
      end

      index =
        if hd(layer_dict["layers"]) > 0 do
          hd(layer_dict["layers"]) + 1
        else
          hd(layer_dict["layers"])
        end

      [state, route_node_specs] = get_previous_node_specs(state, index)
      groups = layer_dict["groups"]
      group_id = layer_dict["group_id"]

      if group_id >= groups do
        raise "Wrong groups and group_id in route node"
      end

      if rem(route_node_specs.channels, groups) != 0 do
        raise "Wrong channels and groups number in route node"
      end

      channels = div(route_node_specs.channels, groups)

      [
        %{
          state
          | nodes:
              Utils.cfl(
                state.nodes,
                Helper.make_node(
                  "Split",
                  [route_node_specs.name],
                  for nn <- 0..(groups - 1) do
                    nn == group_id && layer_name || "#{layer_name}_dummy#{nn}"
                  end,
                  layer_name,
                  %{
                    axis: 1,
                    split: [channels] |> List.duplicate(groups) |> List.flatten()
                  }
                )
              )
        },
        layer_name,
        channels
      ]
    else
      # if there is no "groups" into the layer_dict
      [
        %{
          state
          | route_spec:
              if hd(layer_dict["layers"]) < 0 do
                hd(layer_dict["layers"]) - 1
              else
                if hd(layer_dict["layers"]) > 0 do
                  hd(layer_dict["layers"]) + 1
                end
              end
        },
        layer_name <> "_dummy",
        1
      ]
    end
  end

  def inner_make_route_node(state, layer_name, layer_dict, layer_dict_layers) when layer_dict_layers != 1 do
    if "groups" in Map.keys(layer_dict) do
      raise "groups not implemented for multiple-input route layer!" <> inspect(Map.keys(layer_dict))
    end

    layers = layer_dict["layers"]

    %{inputs: inputs, channels: channels} =
      layers
      |> Enum.reduce(%{inputs: [], channels: 0}, fn index, acc ->
        index = (index > 0 && index + 1) || index
        [_state, previous_node_specs] = get_previous_node_specs(state, index)

        %{
          acc
          | inputs: Utils.cfl(acc.inputs, previous_node_specs.name),
            channels: acc.channels + previous_node_specs.channels
        }
      end)

    [
      %{
        state
        | nodes:
            Utils.cfl(
              state.nodes,
              Helper.make_node("Concat", inputs, [layer_name], layer_name, %{axis: 1})
            )
      },
      layer_name,
      channels
    ]
  end

  @doc """
        If the 'layers' parameter from the DarkNet configuration is only one index, continue
        node creation at the indicated (negative) index. Otherwise, create an ONNX Concat node
        with the route properties from the DarkNet-based graph.
        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
  """
  def make_route_node(state, layer_name, layer_dict) do
    inner_make_route_node(state, layer_name, layer_dict, Enum.count(layer_dict["layers"]))
    # [state, layer_name, channels]
  end

  @doc """
        Create an ONNX Upsample node with the properties from
        the DarkNet-based graph.
        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
  """
  def make_upsample_node(state, layer_name, layer_dict) do
    ## Converting an integer to a float
    upsample_factor = layer_dict["stride"] / 1
    scales = Nx.tensor([1.0, 1.0, upsample_factor, upsample_factor], type: {:f, 32})
    [state, previous_node_specs] = get_previous_node_specs(state)

    if previous_node_specs.channels <= 0 do
      raise "Error in upsample node. Channels cannot be <=0"
    end

    upsample_params = DarknetToOnnx.UpsampleParams.start_link(node_name: layer_name, value: scales)
    inputs = Utils.cfl(previous_node_specs.name, DarknetToOnnx.UpsampleParams.generate_param_name(upsample_params))

    [
      %{
        state
        | nodes:
            Utils.cfl(
              state.nodes,
              Helper.make_node("Upsample", inputs, [layer_name], layer_name, %{mode: "nearest"})
            ),
          param_dict: Utils.cfl(state.param_dict, %{layer_name => upsample_params})
      },
      layer_name,
      previous_node_specs.channels
    ]
  end

  @doc """
        Create an ONNX Yolo node.
        These are dummy nodes which would be removed in the end.
  """
  def make_yolo_node(state, layer_name, _layer_dict) do
    [state, layer_name <> "_dummy", 1]
  end

  @doc """
        Take in a layer parameter dictionary, choose the correct function for
        creating an ONNX node and store the information important to graph creation
        as a MajorNodeSpec object.
        Keyword arguments:
        layer_name -- the layer's name (also the corresponding key in layer_configs)
        layer_dict -- a layer parameter dictionary (one element of layer_configs)
  """
  def make_onnx_node(state, layer_name, layer_dict) do
    layer_type = layer_dict["type"]
    # major_node_specs = %{}
    # major_node_output_name = ""
    # major_node_output_channels = 0

    [state, major_node_specs] =
      if state.input_tensor == nil do
        if layer_type == "net" do
          [state, major_node_output_name, major_node_output_channels] =
            make_input_tensor(
              state,
              layer_name,
              layer_dict
            )

          major_node_specs =
            majorNodeSpecs(
              major_node_output_name,
              major_node_output_channels
            )

          [state, major_node_specs]
        else
          raise "First node must be type net, got: " <> layer_type
        end
      else
        node_creators = %{
          "convolutional" => &make_conv_node/3,
          "maxpool" => &make_maxpool_node/3,
          "shortcut" => &make_shortcut_node/3,
          "route" => &make_route_node/3,
          "upsample" => &make_upsample_node/3,
          "yolo" => &make_yolo_node/3
        }

        [state, major_node_specs] =
          if layer_type in Map.keys(node_creators) do
            [state, major_node_output_name, major_node_output_channels] = node_creators[layer_type].(state, layer_name, layer_dict)
            major_node_specs = majorNodeSpecs(major_node_output_name, major_node_output_channels)
            [state, major_node_specs]
          else
            raise "Layer of type " <> layer_type <> " not supported"
          end

        [state, major_node_specs]
      end

    [state, major_node_specs]
  end

  @doc """
        Iterate over all layer configs (parsed from the DarkNet
        representation of YOLO), create an ONNX graph, populate it with
        weights from the weights file and return the graph definition.
        Keyword arguments:
        layer_configs -- an OrderedDict object with all parsed layers' configurations
        weights_file_path -- location of the weights file
        verbose -- toggles if the graph is printed after creation (default: true)
  """
  def make_initializer_inputs(state, force_raw \\ false) do
    Enum.map(state.param_dict, fn data ->
      [layer_name] = Map.keys(data)
      [_, layer_type] = String.split(layer_name, "_")
      params = data[layer_name]

      case layer_type do
        "convolutional" ->
          DarknetToOnnx.WeightLoader.load_conv_weights(params, force_raw)

        "upsample" ->
          DarknetToOnnx.WeightLoader.load_upsample_scales(params)

        true ->
          raise "Unexpected layer_name"
      end
    end)
    |> Enum.reduce([[], []], fn value, acc ->
      [newv, newi] = value
      [val, inp] = acc
      [Utils.cfl(val, newv), Utils.cfl(inp, newi)]
    end)
  end

  def create_major_node_specs(state, layer_configs, layer_configs_keys) do
    Enum.reduce(layer_configs_keys, {state, []}, fn layer_name, acc ->
      # layer_configs[layer_name]
      %{name: _, data: layer_dict} = Enum.find(layer_configs, fn x -> x.name === layer_name end)
      {state, major_node_specs} = acc
      [state, new_major_node_specs] = make_onnx_node(state, layer_name, layer_dict)
      state = %{state | major_node_specs: Utils.cfl(major_node_specs, new_major_node_specs)}
      {state, state.major_node_specs}
    end)
  end

  def build_onnx_graph(state, layer_configs, layer_configs_keys, weights_file_path, verbose \\ true, force_raw \\ false) do
    {state, major_node_specs} = create_major_node_specs(state, layer_configs, layer_configs_keys)

    state = %{
      state
      | major_node_specs: Utils.cfl(state.major_node_specs, major_node_specs) |> Enum.filter(fn x -> !String.contains?(x.name, "dummy") end)
    }

    outputs =
      Enum.map(state.output_tensors, fn {tensor_name, tensor_data} ->
        Helper.make_tensor_value_info(tensor_name, 1, Utils.cfl(state.batch_size, tensor_data))
      end)

    [state, outputs]
    DarknetToOnnx.WeightLoader.start_link(weights_file: weights_file_path)
    [initializer, inputs] = make_initializer_inputs(state, force_raw)
    DarknetToOnnx.WeightLoader.stop_link()

    state = %{
      state
      | graph_def:
          Helper.make_graph(
            state.nodes,
            state.model_name,
            Utils.cfl(state.input_tensor, inputs),
            outputs,
            initializer
          )
    }

    if verbose == true do
      Helper.printable_graph(state.graph_def)
    end

    Helper.make_model(state.graph_def, producer_name: "Elixir Nx Example")
  end
end
