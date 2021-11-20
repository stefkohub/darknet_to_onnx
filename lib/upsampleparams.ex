defmodule DarknetToOnnx.UpsampleParams do
  @moduledoc """
    Helper class to store the scale parameter for an Upsample node.
  """

  @doc """
        Constructor based on the base node name (e.g. 86_Upsample),
        and the value of the scale input tensor.
        Keyword arguments:
        node_name -- base name of this YOLO Upsample layer
        value -- the value of the scale input to the Upsample layer as a numpy array
  """
  # node_name, batch_normalize, conv_weight_dims) do
  def start_link(opts) do
    initial_state = %{
      node_name: Keyword.fetch!(opts, :node_name),
      value: Keyword.fetch!(opts, :value),
    }
    Agent.start_link(fn -> initial_state end, name: String.to_atom(initial_state.node_name))
    initial_state
  end

  @doc """
        Generates the scale parameter name for the Upsample node.
  """
  def generate_param_name(state) do
    state.node_name <> "_" <> "scale"
  end
end