defmodule  DarknetToOnnx.ConvParams do

  @moduledoc """
    Helper class to store the hyper parameters of a Conv layer,
    including its prefix name in the ONNX graph and the expected dimensions
    of weights for convolution, bias, and batch normalization.
    Additionally acts as a wrapper for generating safe names for all
    weights, checking on feasible combinations.
  """

  @doc """
        Constructor based on the base node name (e.g. 101_convolutional), the batch
        normalization setting, and the convolutional weights shape.
        Keyword arguments:
        node_name -- base name of this YOLO convolutional layer
        batch_normalize -- bool value if batch normalization is used
        conv_weight_dims -- the dimensions of this layer's convolutional weights
  """
  def start_link(opts) do #node_name, batch_normalize, conv_weight_dims) do
    initial_state = %{
      node_name: Keyword.fetch!(opts, :node_name),
      batch_normalize: Keyword.fetch!(opts, :batch_normalize),
        # TODO: assert len(conv_weight_dims) == 4
      conv_weight_dims: Keyword.fetch!(opts, :conv_weight_dims)
    }
    Agent.start_link(fn -> initial_state end, name: String.to_atom(initial_state.node_name))
    initial_state
  end

  @doc """
        Generates a name based on two string inputs,
        and checks if the combination is valid.
  """
  def generate_param_name(state, param_category, suffix) when
    param_category in ["bn", "conv"] and suffix in ["scale", "mean", "var", "weights", "bias"]  do
    #TODO (considering use of fetch! in start_link)
    if param_category == "bn" and suffix not in ["scale", "bias", "mean", "var"] do
      raise "Error in generate_param_name: wrong suffix "<>suffix<>" for bn category"
    end
    if param_category == "conv" and suffix not in ["weights", "bias"] do
      raise "Error in generate_param_name: wrong suffix "<>suffix<>" for bias category"
    end
    state.node_name <> "_" <> param_category <> "_" <> suffix
        #if param_category == 'bn':
        #    assert self.batch_normalize
        #    assert suffix in ['scale', 'bias', 'mean', 'var']
        #elif param_category == 'conv':
        #    assert suffix in ['weights', 'bias']
        #    if suffix == 'bias':
        #        assert not self.batch_normalize
  end

end
