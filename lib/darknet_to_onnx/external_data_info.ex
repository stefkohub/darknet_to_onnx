defmodule DarknetToOnnx.ExternalDataInfo do
  def start_link(tensor) do
    initialState = %{
      location: "",
      offset: nil,
      length: nil,
      checksum: nil,
      basepath: ""
    }

    if tensor.external_data != [] do
      Map.merge(initialState, tensor.external_data)
    else
      initialState
    end
  end
end
