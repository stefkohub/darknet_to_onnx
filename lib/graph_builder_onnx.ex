defmodule DarknetToOnnx.GraphBuilderONNX do
  @moduledoc """
      Class for creating an ONNX graph from a previously generated list of layer dictionaries.
  """

  use Agent
  require Logger

  @doc """
        Initialize with all DarkNet default parameters used creating
        YOLO, and specify the output tensors as an OrderedDict for their
        output dimensions with their names as keys.
        Keyword argument:
        output_tensors -- the output tensors as an OrderedDict containing the keys'
        output dimensions
  """
  def start_link(opts) do
    initial_state=%{
            model_name: Keyword.fetch!(opts, :model_name) ,
        output_tensors: Keyword.fetch!(opts, :output_tensors),
                 nodes: [],
             graph_def: nil,
          input_tensor: nil,
            epsilon_bn: 1.0e-5,
           momentum_bn: 0.99,
           alpha_lrelu: 0.1,
            param_dict: %{},
      major_node_specs: [],
             batch_size: Keyword.fetch!(opts, :batch_size),
            route_spec: 0
    }
    Agent.start_link(fn -> initial_state end, name: __MODULE__)
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
    input_tensor = Nx.tensor(
      [state.batch_size, channels, height, width], 
      names: [String.to_atom(layer_name)], 
      type: {:f, 64})
    #input_tensor = helper.make_tensor_value_info(
    #  layer_name, TensorProto.FLOAT, [
    #            self.batch_size, channels, height, width])
    state = %{state|input_tensor: input_tensor}
    [state, layer_name, channels]
  end


  @doc """
    mah...
  """
  def majorNodeSpecs(name, channels) do
    %{               :name => name, 
                 :channels => channels, 
        :created_onnx_node => if (name != nil and is_integer(channels) and channels>0) do True else False end
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
        #TODO: assert 'dummy' not in previous_node.name
        r=[%{state | route_spec: 0}, Enum.at(state.major_node_specs,state.route_spec)]
        IO.puts "get_previous_node_specs ritorna 1: "<>inspect([state.route_spec, state.major_node_specs,r])
        r
      else
        r=[state, Enum.at(state.major_node_specs, -1)]
        IO.puts "get_previous_node_specs ritorna 2: "<>inspect(r)
        r
      end
    else
      r=[state, Enum.at(state.major_node_specs,target_index)]
      IO.puts "get_previous_node_specs ritorna 3: "<>inspect(r)
      r
    end
    #TODO: assert previous_node.created_onnx_node
end

  defp make_conv_node_batch_normalize(state, conv_params_state, layer_name_output) when 
       conv_params_state.batch_normalize == True do
    %Onnx.NodeProto{
         op_type: "BatchNormalization",
           input: [layer_name_output] ++ for suffix <- ["scale", "bias", "mean", "var"] do 
                    DarknetToOnnx.ConvParams.generate_param_name(conv_params_state, "bn", suffix)
                  end,
          output: [layer_name_output <>"_bn"],
            name: [layer_name_output<>"_bn"],
       attribute: %{
         epsilon: state.epsilon_bn,
        momentum: state.momentum_bn
      }
    }
  end

  def make_conv_node_leaky_relu(state, conv_params_state, layer_name) do
    %Onnx.NodeProto{
      op_type: "LeakyRelu",
      input: [layer_name<>"_bn"],
      output: [layer_name <>"_lrelu"],
      name: [layer_name<>"_lrelu"],
      attribute: %{
        alpha: state.alpha_lrelu
      }
    }
  end

  def make_conv_node_mish(state, conv_params_state, layer_name) do
    layer_name_softplus=layer_name<>"_softplus"
    [ %Onnx.NodeProto{
        op_type: "Softplus",
        input: [layer_name],
        output: [layer_name_softplus],
        name: [layer_name_softplus],
    },
      %Onnx.NodeProto{
        op_type: "Tanh",
        input: [layer_name_softplus],
        output: [layer_name <>"_tanh"],
        name: [layer_name<>"_tanh"],
      },
      %Onnx.NodeProto{
        op_type: "Mul",
        input: [layer_name],
        output: [layer_name <>"_mish"],
        name: [layer_name<>"_mish"],
      }
    ]
  end

  def make_conv_node_swish(state, conv_params_state, layer_name) do
    [ %Onnx.NodeProto{
        op_type: "Sigmoid",
        input: [layer_name],
        output: [layer_name<>"_sigmoid"],
        name: layer_name<>"_sigmoid"
    },
      %Onnx.NodeProto{
        op_type: "Mul",
        input: [layer_name],
        output: [layer_name<>"_swish"],
        name: layer_name<>"_swish"
      }
    ]
  end

  def make_conv_node_logistic(state, conv_params_state, layer_name) do
    %Onnx.NodeProto{
      op_type: "Sigmoid",
      input: [layer_name],
      output: [layer_name <>"_lgx"],
      name: layer_name <>"_lgx"
    }
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
    weights_shape = [layer_dict["filters"], previous_node_specs.channels] 
    conv_params_state = DarknetToOnnx.ConvParams.start_link([
      node_name: layer_name, 
      batch_normalize: if layer_dict["batch_normalize"]>0 do True else False end,
      conv_weight_dims: weights_shape
    ])
    strides = [layer_dict["stride"], layer_dict["stride"]]
    dilations = [1, 1]
    weights_name = DarknetToOnnx.ConvParams.generate_param_name(conv_params_state, "conv", "weights")
    inputs = if !conv_params_state.batch_normalize do
      [previous_node_specs.name] ++ [weights_name] ++ DarknetToOnnx.ConvParams.generate_param_name(conv_params_state, "conv", "bias")
    else
      [previous_node_specs.name] ++ [weights_name]
    end
    conv_node = %Onnx.NodeProto{
      op_type: "Conv",
      input: inputs,
      output: [layer_name],
      name: layer_name,
      attribute: %{
        kernel_shape: kernel_shape,
        strides: strides,
        auto_pad: 'SAME_LOWER',
        dilations: dilations
      }
    }
    # IO.puts "DEVO SOMMARE"
    # IO.puts inspect(state.nodes)
    # IO.puts "CON"
    # IO.puts inspect(conv_node)
    state=%{state | nodes: state.nodes ++ [conv_node]}
    layer_name_output=layer_name
    [state, layer_name_output] = if conv_params_state.batch_normalize == True do 
      [
        %{state | nodes: state.nodes ++ [make_conv_node_batch_normalize(state, conv_params_state, layer_name)]}, 
        layer_name<>"_bn"
      ]
    else
      [state, layer_name_output]
    end
    [state, layer_name_output] = case layer_dict["activation"] do
      "leaky" ->
        [
          %{state | nodes: state.nodes ++ [make_conv_node_leaky_relu(state, conv_params_state, layer_name)]},
          layer_name<>"_leaky"
        ]
      "mish" ->
        [
          %{state | nodes: state.nodes ++ make_conv_node_mish(state, conv_params_state, layer_name)},
          layer_name<>"_mish"
        ]
      "swish" ->
        [
          %{state | nodes: state.nodes ++ make_conv_node_swish(state, conv_params_state, layer_name)},
          layer_name<>"_swish"
        ]
      "logistic" ->
        [
          %{state | nodes: state.nodes ++ [make_conv_node_logistic(state, conv_params_state, layer_name)]},
          layer_name<>"_logistic"
        ]
      _ ->
        [state, layer_name_output]
    end
    state=%{state | param_dict: Map.merge(state.param_dict, %{layer_name => conv_params_state})}
    IO.puts "++++++++++++++++++++++++++++++++++++++++++++++++++"
    IO.puts "make_conv_node state="<>inspect(state)
    IO.puts "make_conv_node layer_name="<>inspect(layer_name)
    IO.puts "make_conv_node conv_params_state="<>inspect(conv_params_state)
    IO.puts "--------------------------------------------------"
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
      %{state | nodes: state.nodes ++ [%Onnx.NodeProto{
            op_type: "MaxPool",
            input: [previous_node_specs.name],
            output: [layer_name],
            name: [layer_name],
            attribute: %{
              kernel_shape: [layer_dict["size"], layer_dict["size"]],
              auto_pad: "SAME_UPPER",
              strides: [layer_dict["stride"], layer_dict["stride"]]
            }
          }]
      },
      layer_name,
      previous_node_specs.channels
    ]
  end

  def make_shortcut_node(state, layer_name, layer_dict) do
    [state, "shortcut", 2 ]
  end

  def inner_make_route_node(state, layer_name, layer_dict, layer_dict_layers) when layer_dict_layers == 1 do
    if "groups" in Map.keys(layer_dict) do
      index = if hd(layer_dict["layers"]) > 0 do hd(layer_dict["layers"])+1 else hd(layer_dict["layers"]) end
      [state, route_node_specs] = get_previous_node_specs(state, target_index=index)
      groups=layer_dict["groups"]
      [%{state | nodes: state.nodes ++ %Onnx.NodeProto{
            op_type: "Split",
            input: [route_node_specs.name],
            output: for nn <- 0..groups, into: [] do 
              if nn==groups do 
                layer_name 
              else 
                layer_name<>"_dummy"<>Integer.to_string(nn) 
              end 
            end,
            name: layer_name,
            attribute: %{
              axis: 1,
              split: [route_node_specs.channels]|>List.duplicate(groups)|>List.flatten 
            }
          }
        },
        layer_name,
        div(route_node_specs.channels, groups)
       ]
    else
      # if there is no "groups" into the layer_dict
      [
        %{state | route_spec: if hd(layer_dict["layers"]) <0 do
          hd(layer_dict["layers"]) - 1
        else if hd(layer_dict["layers"]) > 0 do
            hd(layer_dict["layers"]) + 1
        end
        end},
        layer_name <> "_dummy",
        1
      ]
    end
  end

  def inner_make_route_node(state, layer_name, layer_dict, layer_dict_layers) when layer_dict_layers != 1 do
    # TODO Check for "groups" NOT IN Map.keys(layer_dict)
    if "groups" not in Map.keys(layer_dict) do
      raise "groups not implemented for multiple-input route layer!"<>inspect(Map.keys(layer_dict))
    end
    # Puoi mettere i valori direttamente nella chiamata alla funzione togliendo assegnazioni inutili
    inputs=[]
    channels=0
    layers=layer_dict["layers"]
    [inputs, channels] = inner_make_route_node_recursive(state, inputs, channels, layers)
    # IO.puts "HO OTTENUTO [inputs, channels]="<>inspect([inputs, channels])
    [ 
      %{state | nodes: state.nodes ++ %Onnx.NodeProto{
            op_type: "Concat",
            input: inputs,
            output: [layer_name],
            name: layer_name,
            attribute: %{
              axis: 1,
            }
          }
      },
      layer_name,
      channels
    ]
  end

  def inner_make_route_node_recursive(state, inputs, channels, []) do
    [inputs, channels]
  end

  def inner_make_route_node_recursive(state, inputs, channels, layers) do
    index=hd(layers)
    index=if index>0 do index+1 else index end
    [state, previous_node_specs] = get_previous_node_specs(state, target_index=index)
    inputs = inputs ++ previous_node_specs.name
    channels = channels + previous_node_specs.channels
    inner_make_route_node_recursive(state, inputs, channels, tl(layers))
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
    [state, layer_name, channels] = inner_make_route_node(state, layer_name, layer_dict, Enum.count(layer_dict["layers"]))
    [state, layer_name, channels]
  end

  def make_upsample_node(state, layer_name, layer_dict) do
    [state, "upsample", 4]
  end

  @doc """
        Create an ONNX Yolo node.
        These are dummy nodes which would be removed in the end.
  """
  def make_yolo_node(state, layer_name, layer_dict) do
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
    major_node_specs = %{}
    major_node_output_name = ""
    major_node_output_channels = 0
    [state, major_node_specs ] = if state.input_tensor == nil do
      if layer_type == "net" do
        [state, major_node_output_name, major_node_output_channels] = make_input_tensor(
                    state, layer_name, layer_dict)
        major_node_specs = majorNodeSpecs(major_node_output_name,
                                          major_node_output_channels)
        [state, major_node_specs]
      else
        raise "First node must be type net"
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
      [state, major_node_specs] = if layer_type in Map.keys(node_creators) do
        [ state, major_node_output_name, major_node_output_channels ] = 
              node_creators[layer_type].(state, layer_name, layer_dict)
        major_node_specs = majorNodeSpecs(major_node_output_name, major_node_output_channels)
        [state, major_node_specs]
      else
        raise "Layer of type "<>layer_type<>" not supported"
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
        verbose -- toggles if the graph is printed after creation (default: True)
  """
  defp inner_create_major_node_specs(state, _layer_configs, []) do
    state
  end

  defp inner_create_major_node_specs(state, layer_configs, acc) do
    layer_name = hd(acc)
    layer_dict = layer_configs[layer_name]
    [state, major_node_specs] = make_onnx_node(state, layer_name, layer_dict)
    state = try do
      # state = if major_node_specs != nil and major_node_specs.name != nil do
      %{state | major_node_specs: state.major_node_specs ++ [major_node_specs]}
    rescue
      e in RuntimeError -> e
    end
    inner_create_major_node_specs(state, layer_configs, tl(acc))
  end

  def build_onnx_graph(state, layer_configs, weights_file_path, verbose \\ True) do
    # Enum.each(layer_configs, fn({layer_name, layer_dict}) -> 
    state = %{state | major_node_specs: inner_create_major_node_specs(state, layer_configs, Map.keys(layer_configs))}
    #end)
  end

end