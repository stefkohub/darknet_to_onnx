defmodule DarknetToOnnx.MajorNodeSpecs do
  @moduledoc """
    Helper class used to store the names of ONNX output names,
    corresponding to the output of a DarkNet layer and its output channels.
    Some DarkNet layers are not created and there is no corresponding ONNX node,
    but we still need to track them in order to set up skip connections.
  """

  use Agent
  require Logger

  

end
