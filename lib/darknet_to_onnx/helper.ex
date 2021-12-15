defmodule DarknetToOnnx.Helper do
  @moduledoc """
    Helper class used for creating tensors
    (partially ported from: https://github.com/onnx/onnx/blob/master/onnx/helper.py)
  """

  @onnx_opset_version 15
  @onnx_ir_version 8
  @output_file_name "yolov3-tiny-416-SF.onnx"

  alias DarknetToOnnx.Learning, as: Utils
  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  #  alias Onnx.NodeProto, as: Node
  alias Onnx.ValueInfoProto, as: Value
  alias Onnx.AttributeProto, as: Attribute
  alias Onnx.OperatorSetIdProto, as: Opset
  alias Onnx.TypeProto, as: Type
  # NON MI PIACE alias Onnx.TypeProto.Tensor, as: Placeholder
  alias Onnx.TensorProto, as: TensorProto
  alias Onnx.TypeProto.Tensor, as: TensorTypeProto
  alias Onnx.TensorShapeProto, as: Shape
  alias Onnx.TensorShapeProto.Dimension, as: Dimension

  @doc """
    Make a TensorProto with specified arguments.  If raw is False, this
    function will choose the corresponding proto field to store the
    values based on data_type. If raw is True, use "raw_data" proto
    field to store the values, and values should be of type bytes in
    this case.
  """
  def make_tensor(name, data_type, dims, vals, raw \\ False) do
    # TODO: add a zillion of checks on sizes and so on...
    # expected_size = Enum.reduce(Tuple.to_list(dims), 1, fn val, acc -> acc * val end)
    # TODO add support for complex and float 16 values...

    tensor = %TensorProto{
      data_type: data_type,
      name: name,
      raw_data: (raw && vals) || "",
      float_data: (!raw && vals) || [],
      dims: Tuple.to_list(dims)
    }

    tensor
  end

  def make_tensor_value_info(name, elem_type, shape, doc_string \\ "", shape_denotation \\ "") do
    the_type = make_tensor_type_proto(elem_type, shape, shape_denotation)

    %Value{
      name: name,
      doc_string: (doc_string !== "" && doc_string) || "",
      type: the_type
    }
  end

  def make_tensor_type_proto(elem_type, shape, shape_denotation \\ []) do
    %Type{
      value:
        {:tensor_type,
         %TensorTypeProto{
           elem_type: elem_type,
           shape:
             if shape != nil do
               if Utils.is_enum?(shape_denotation) == True and Enum.count(shape_denotation) != 0 and
                    Enum.count(shape_denotation) != Enum.count(shape) do
                 raise "Invalid shape_denotation. Must be the same length as shape."
               end

               %Shape{dim: create_dimensions(shape, shape_denotation)}
             else
               %Shape{}
             end
         }}
    }
  end

  defp create_dimensions(shape, shape_denotation) do
    list_shape = (is_tuple(shape) && Tuple.to_list(shape)) || shape

    list_shape
    |> Enum.with_index()
    |> Enum.map(fn {acc, index} ->
      cond do
        is_integer(acc) ->
          %Dimension{
            value: {:dim_value, acc},
            denotation:
              if shape_denotation != "" do
                Enum.at(shape_denotation, index)
              else
                ""
              end
          }

        is_binary(acc) ->
          %Dimension{
            value: {:dim_param, acc},
            denotation:
              if shape_denotation != "" do
                Enum.at(shape_denotation, index)
              else
                ""
              end
          }

        [] ->
          _ = IO.puts("Empty acc")

        true ->
          raise "Invalid item in shape: " <> inspect(acc) <> ". Needs to be integer or text type."
      end
    end)
    |> List.flatten()
  end

  def make_graph(nodes, name, inputs, outputs, initializer \\ [], doc_string \\ "", value_info \\ [], sparse_initializer \\ []) do
    %Graph{
      doc_string: doc_string,
      initializer: initializer,
      input: inputs,
      name: name,
      node: nodes,
      output: outputs,
      quantization_annotation: [],
      sparse_initializer: sparse_initializer,
      value_info: value_info
    }
  end

  def make_model(graph, kwargs) do
    %Model{
      doc_string: Keyword.get(kwargs, :doc_string, ""),
      domain: Keyword.get(kwargs, :domain, ""),
      graph: graph,
      ir_version: @onnx_ir_version,
      metadata_props: Keyword.get(kwargs, :metadata_props, []),
      model_version: Keyword.get(kwargs, :model_version, 1),
      opset_import: Keyword.get(kwargs, :opset_imports, [%Opset{domain: "", version: @onnx_opset_version}]),
      producer_name: Keyword.get(kwargs, :producer_name, ""),
      producer_version: Keyword.get(kwargs, :producer_version, "0.0.1-sf"),
      training_info: Keyword.get(kwargs, :training_info, [])
    }
  end

  def printable_graph(graph, prefix \\ "") do
    indent = prefix <> "  "
    header = ["graph", graph.name]
    # initializers = 
  end

  def get_all_tensors(model) do
    #  TODO:, attributes from model.graph.node
    model.graph.initializer
  end

  def set_external_data(tensor, location, offset \\ nil, length \\ nil, checksum \\ nil, basepath \\ nil) do
    if !Map.has_key?(tensor, "raw_data") do
      raise "raw_data field doesn't exist."
    end

    %{
      tensor
      | data_location: :EXTERNAL,
        external_data:
          Utils.cfl(tensor.external_data, [
            %{
              "location" => location,
              "offset" => offset,
              "length" => length,
              "checksum" => checksum,
              "basepath" => basepath
            }
          ])
    }
  end

  def save_external_data(tensor, filepath) do
    info = ExternalDataInfo.start_link(tensor)
    external_data_file_path = Path.join(filepath, info.location)

    if !Map.has_key?(tensor, "raw_data") do
      raise "raw_data field doesn't exist."
    end

    {:ok, data_file} = File.open(external_data_file_path, [:binary, :write])
    # TODO: Should ensure appending to the file
    # TODO: if info.offset ....
    IO.binwrite(data_file, tensor.raw_data.data.state)
    # return a new tensor
    File.close(data_file)
    {:ok, %{size: offset}} = File.stat!(external_data_file_path)
    set_external_data(tensor, info.location, offset, byte_size(tensor.raw_data.data.state))
  end

  def uses_external_data(tensor) do
    Map.has_key?(tensor, "data_location") and tensor.data_location == :EXTERNAL
  end

  def write_external_data_tensors(model, filepath) do
    Enum.each(get_all_tensors(model), fn tensor ->
      if uses_external_data(tensor) do
        save_external_data(tensor, filepath)
        # TODO: tensor.ClearField("raw_data")
      end
    end)
  end

  def save_model(proto, f) do
    encoded_model = Onnx.ModelProto.encode!(proto)
    {:ok, file} = File.open(f, [:write])
    IO.binwrite(file, encoded_model)
    File.close(file)
  end

  def is_TensorProto(val) do
    is_map(val) and Map.has_key?(val, :__struct__) and
      val.__struct__ === Onnx.TensorProto
  end

  def is_SparseTensorProto(val) do
    is_map(val) and Map.has_key?(val, :__struct__) and
      val.__struct__ === Onnx.SparseTensorProto
  end

  def is_GraphProto(val) do
    is_map(val) and Map.has_key?(val, :__struct__) and
      val.__struct__ === Onnx.GraphProto
  end

  def is_TypeProto(val) do
    is_map(val) and Map.has_key?(val, :__struct__) and
      val.__struct__ === Onnx.TypeProto
  end

  def inner_make_attribute(key, val) do
    newAttr = %Attribute{
      name: Atom.to_string(key)
    }

    to_add =
      cond do
        is_float(val) ->
          %{f: val, type: :FLOAT}

        is_integer(val) ->
          %{i: val, type: :INT}

        is_binary(val) or is_boolean(val) ->
          %{s: val, type: :STRING}

        is_TensorProto(val) ->
          %{t: val, type: :TENSOR}

        is_SparseTensorProto(val) ->
          %{sparse_tensor: val, type: :SPARSE_TENSOR}

        is_GraphProto(val) ->
          %{g: val, type: :GRAPH}

        is_TypeProto(val) ->
          %{tp: val, type: :TYPE_PROTO}

        Utils.is_enum?(val) && Enum.all?(val, fn x -> is_integer(x) end) ->
          %{ints: val, type: :INTS}

        Utils.is_enum?(val) and Enum.all?(val, fn x -> is_float(x) or is_integer(x) end) ->
          # Convert all the numbers to float
          %{floats: Enum.map(val, fn v -> v / 1 end), type: :FLOATS}

        Utils.is_enum?(val) and Enum.all?(val, fn x -> is_binary(x) end) ->
          %{strings: val, type: :STRINGS}

        Utils.is_enum?(val) and Enum.all?(val, fn x -> is_TensorProto(x) end) ->
          %{tensors: val, type: :TENSORS}

        Utils.is_enum?(val) and Enum.all?(val, fn x -> is_SparseTensorProto(x) end) ->
          %{sparse_tensors: val, type: :SPARSE_TENSORS}

        Utils.is_enum?(val) and Enum.all?(val, fn x -> is_GraphProto(x) end) ->
          %{graphs: val, type: :GRAPHS}

        Utils.is_enum?(val) and Enum.all?(val, fn x -> is_TypeProto(x) end) ->
          %{type_protos: val, type: :TYPE_PROTOS}
      end

    Map.merge(newAttr, to_add)
  end

  def make_attribute(kwargs) do
    sortedargs = for {k, v} <- Enum.sort(kwargs), v != "", do: {k, v}

    sortedargs
    |> Enum.reduce([], fn {key, val}, acc ->
      [inner_make_attribute(key, val) | acc]
    end)
  end

  @doc """
        Construct a NodeProto.
        Arguments:
        op_type (string): The name of the operator to construct
        inputs (list of string): list of input names
        outputs (list of string): list of output names
        name (string, default None): optional unique identifier for NodeProto
        doc_string (string, default None): optional documentation string for NodeProto
        domain (string, default None): optional domain for NodeProto.
            If it's None, we will just use default domain (which is empty)
        kwargs (dict): the attributes of the node.  The acceptable values
            are documented in :func:`make_attribute`.
  """
  def make_node(op_type, inputs, outputs, name \\ "", kwargs \\ [], doc_string \\ "", domain \\ "") do
    %Onnx.NodeProto{
      op_type: op_type,
      input: inputs,
      output: outputs,
      name: name,
      domain: domain,
      doc_string: doc_string,
      attribute: make_attribute(kwargs)
    }
  end
end
