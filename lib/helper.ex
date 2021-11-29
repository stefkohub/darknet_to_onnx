defmodule DarknetToOnnx.Helper do
  @moduledoc """
    Helper class used for creating tensors
    (ported from: https://github.com/onnx/onnx/blob/master/onnx/helper.py)
  """

  alias DarknetToOnnx.Learning, as: Utils
  alias Onnx.ModelProto, as: Model
  alias Onnx.GraphProto, as: Graph
  alias Onnx.NodeProto, as: Node
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
    expected_size = Enum.reduce(Tuple.to_list(dims), 1, fn val, acc -> acc*val end)
    # TODO add support for complex and float 16 values...
    IO.puts("DENTRO MAKE_TENSOR HO: "<>inspect([name, data_type, dims, vals, raw]))
    tensor = %TensorProto{
      data_type: data_type,
      raw_data: raw && vals || "",
      float_data: vals,
      dims: dims
    }
    IO.puts("DENTRO MAKE_TENSOR tensor= "<>inspect(tensor))
    tensor
  end

  def make_tensor_value_info(name, elem_type, shape, doc_string \\ "", shape_denotation \\ []) do
    the_type = make_tensor_type_proto(elem_type, shape, shape_denotation)

    %Onnx.ValueInfoProto{
      name: name,
      doc_string:
        if doc_string do
          doc_string
        else
          ""
        end,
      type: the_type
    }
  end

  def make_tensor_type_proto(elem_type, shape, shape_denotation \\ []) do
    tensor_shape_proto = %Shape{dim: %Dimension{}}

    type_proto = %Type{
      value:
        {:tensor_type,
         %TensorTypeProto{
           elem_type: elem_type,
           shape:
             if shape != nil do
               if Enum.count(shape_denotation) != 0 and Enum.count(shape_denotation) != Enum.count(shape) do
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
    list_shape = is_tuple(shape) && Tuple.to_list(shape) || shape
    Enum.reduce(list_shape |> Enum.with_index(), [], fn {value, index}, acc -> create_dimensions_reduce(shape_denotation, value, index, acc) end)
  end

  defp create_dimensions_reduce(shape_denotation, value, index, acc) do
    [value, acc]
    |> Enum.map(fn acc ->
      cond do
        is_integer(acc) ->
          %Dimension{
            value: {:dim_value, acc},
            denotation:
              if shape_denotation do
                Enum.at(shape_denotation, index)
              else
                ""
              end
          }

        is_binary(acc) ->
          %Dimension{
            value: {:dim_param, acc},
            denotation:
              if shape_denotation do
                Enum.at(shape_denotation, index)
              else
                ""
              end
          }

        true ->
          acc
      end
    end)
    |> List.flatten()
  end

  def make_graph(nodes, name, inputs, outputs, initializer \\ [], doc_string \\ "", value_info \\ [], sparse_initializer \\ []) do
    %Graph{
      doc_string: doc_string,
      initializer: [],
      input: inputs,
      name: name,
      node: node,
      output: outputs,
      quantization_annotation: [],
      sparse_initializer: sparse_initializer,
      value_info: value_info
    }
  end
end
