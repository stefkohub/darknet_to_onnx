defmodule DarknetToOnnx.ConvParams do
  @moduledoc """
    Helper class to store the hyper parameters of a Conv layer,
    including its prefix name in the ONNX graph and the expected dimensions
    of weights for convolution, bias, and batch normalization.
    Additionally acts as a wrapper for generating safe names for all
    weights, checking on feasible combinations.
  """
  use Agent, restart: :transient

  @doc """
        Constructor based on the base node name (e.g. 101_convolutional), the batch
        normalization setting, and the convolutional weights shape.
        Keyword arguments:
        node_name -- base name of this YOLO convolutional layer
        batch_normalize -- bool value if batch normalization is used
        conv_weight_dims -- the dimensions of this layer's convolutional weights
  """
  # node_name, batch_normalize, conv_weight_dims) do
  def start_link(opts) do
    initial_state = %{
      node_name: Keyword.fetch!(opts, :node_name),
      batch_normalize: Keyword.fetch!(opts, :batch_normalize),
      # TODO: assert len(conv_weight_dims) == 4
      conv_weight_dims: Keyword.fetch!(opts, :conv_weight_dims)
    }

    Agent.start_link(fn -> initial_state end, name: String.to_atom(initial_state.node_name))
    initial_state
  end

  def get_state(node_name) do
    Agent.get(String.to_atom(node_name), fn state -> state end)
  end

  @doc """
        Generates a name based on two string inputs,
        and checks if the combination is valid.
  """
  # The first def is a header needed to use default values in parameter list.
  # In Elixir it is not possible to use default values in more than 1 function declaration.
  def generate_param_name(node_name, param_category, suffix \\ nil)

  def generate_param_name(node_name, param_category, suffix)
      when suffix != nil and param_category in ["bn", "conv"] and
             suffix in ["scale", "mean", "var", "weights", "bias"] and
             is_binary(node_name) do
    generate_param_name(get_state(node_name), param_category, suffix)
  end

  def generate_param_name(state, param_category, suffix)
      when suffix != nil and param_category in ["bn", "conv"] and
             suffix in ["scale", "mean", "var", "weights", "bias"] do
    cond do
      param_category == "bn" and state.batch_normalize != True and
          suffix not in ["scale", "bias", "mean", "var"] ->
        raise "Error in generate_param_name: wrong suffix " <> suffix <> " for bn category"

      param_category == "conv" and suffix not in ["weights", "bias"] ->
        if suffix == "bias" and state.batch_normalize != True do
          raise "Error in generate_param_name: wrong suffix " <> suffix <> " for conv category"
        else
          raise "Error in generate_param_name: wrong suffix " <> suffix <> " for conv category"
        end

      true ->
        state.node_name <> "_" <> param_category <> "_" <> suffix
    end
  end
end
