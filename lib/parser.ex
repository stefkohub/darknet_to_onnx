defmodule DarknetToOnnx.ParseDarknet do
  @moduledoc """
  The Darknet parser (from: https://github.com/jkjung-avt/tensorrt_demos/blob/master/yolo/yolo_to_onnx.py)
  """

  use Agent
  require Logger

  @doc """
        Initializes a DarkNetParser object.
        Keyword argument:
        supported_layers -- a string list of supported layers in DarkNet naming convention,
        parameters are only added to the class dictionary if a parsed layer is included.
  """
  def start_link(
        opts,
        args \\ [
          'net',
          'convolutional',
          'maxpool',
          'shortcut',
          'route',
          'upsample',
          'yolo'
        ]
      ) do
    cfg_file_path = Keyword.fetch!(opts, :cfg_file_path)

    initial_state = %{
      parse_result: [],
      keys: [],
      output_convs: []
    }

    Agent.start_link(fn -> parse_cfg_file(initial_state, cfg_file_path) end, name: __MODULE__)
  end

  @doc """
      Identifies the parameters contained in one of the cfg file and returns
        them in the required format for each parameter type, e.g. as a list, an int or a float.
        Keyword argument:
        param_line -- one parsed line within a layer block
  """
  def parse_params(params, skip_params \\ ["steps", "scales", "mask"]) do
    # [param_type, param_value_raw] = param |> String.replace(~r/\s+/, "")|>String.split("=")
    {param_type, param_value_raw} = params
    param_value_raw = param_value_raw |> String.replace(~r/\s+/, "")
    # IO.puts("Analizzo params: " <> inspect([params, param_type, param_value_raw]))

    cond do
      skip_params != nil and param_type in skip_params ->
        [nil, param_value_raw]

      param_type == "layers" or param_type == "anchors" ->
        [
          param_type,
          param_value_raw
          |> String.split(",")
          |> Enum.map(fn x -> String.to_integer(String.trim(x)) end)
        ]

      !String.match?(param_value_raw, ~r/^[[:alpha:]]+$/u) and String.match?(param_value_raw, ~r/\./) ->
        # It is a float number
        [zero, decimali] = param_value_raw |> String.split(".")

        if zero === "" do
          [param_type, String.to_float("0" <> param_value_raw)]
        else
          [param_type, param_value_raw]
        end

      !String.match?(param_value_raw, ~r/^[[:alpha:]]+$/u) ->
        # Otherwise it is integer
        [param_type, String.to_integer(param_value_raw)]

      true ->
        [param_type, param_value_raw]
    end
  end

  @doc """
    Takes the yolov?.cfg file and parses it layer by layer,
        appending each layer's parameters as a dictionary to layer_configs.
        Keyword argument:
        cfg_file_path
  """
  def old_parse_cfg_file(state, cfg_file_path) do
    # TODO: Add checks on file parsing
    {:ok, parse_result} = ConfigParser.parse_file(cfg_file_path)
    keys = Map.keys(parse_result)
    # Adding the type keyword... That's just for learning. The type can ne obtained on the fly with a match
    # like: { _, type } = String.split(key, "_")
    parse_result =
      DarknetToOnnx.Learning.update_map(parse_result, fn state, k, v ->
        # [_ptype, pvalue]=parse_params(state, state[k])
        # IO.puts("STATE K=" <> inspect(state[k]))

        Enum.map(state[k], fn {k, v} ->
          # IO.puts("CHIAMO parse_param CON: " <> inspect({k, v}))
          [_ptype, pvalue] = parse_params({k, v})
          {k, pvalue}
        end)
        |> Map.new()
      end)

    parse_result = DarknetToOnnx.Learning.update_map_adding_type(parse_result)
    # IO.puts("Qui parse_result=" <> inspect(parse_result))
    %{state | :parse_result => parse_result, :keys => keys}
  end

  def parse_cfg_file(state, cfg_file_path) do
    {:ok, parse_result} = ConfigParser.parse_file(cfg_file_path)

    parse_result =
      Enum.map(parse_result, fn {name, datamap} ->
        [_, new_type] = String.split(name, "_")

        {name,
         (Enum.map(datamap, fn {k, v} ->
            [_ptype, pvalue] = DarknetToOnnx.ParseDarknet.parse_params({k, v})
            {k, pvalue}
          end) ++ [{"type", new_type}])
         |> Map.new()}
      end)
      |> Map.new()

    %{state | :parse_result => parse_result, :keys => Map.keys(parse_result)}
  end

  def get_state() do
    Agent.get(__MODULE__, fn state -> state end)
  end

  def is_pan_arch?(state) do
    yolos = Enum.filter(state.keys, fn k -> Regex.run(~r{.*yolo}, k) end)
    upsamples = Enum.filter(state.keys, fn k -> Regex.run(~r{.*upsample}, k) end)
    yolo_count = Enum.count(yolos)
    upsample_count = Enum.count(upsamples)

    try do
      yolo_count in [2, 3, 4]
      upsample_count == yolo_count - 1 or upsample_count == 0
    rescue
      e ->
        Logger.error(Exception.format(:error, e, __STACKTRACE__))
        nil
    end

    # the model is with PAN if an upsample layer appears before the 1st yolo
    [yindex, _] = String.split(hd(yolos), "_")
    [uindex, _] = String.split(hd(upsamples), "_")
    uindex < yindex
  end

  @doc """
  Find output conv layer names from layer configs.
    The output conv layers are those conv layers immediately proceeding
    the yolo layers.
    # Arguments
        layer_configs: output of the DarkNetParser, i.e. a OrderedDict of
                       the yolo layers.
  """
  def get_output_convs(state) do
    yolos = Enum.filter(state.keys, fn k -> Regex.run(~r{.*yolo}, k) end)
    # convs = Enum.filter(state.keys, fn(k) -> Regex.run(~r{.*convolutional}, k) end)
    state = %{state | :output_convs => []}
    output_convs = inner_get_output_convs(state, yolos)
    Agent.update(__MODULE__, fn _s -> %{state | output_convs: output_convs} end)
    output_convs
  end

  defp inner_get_output_convs(state, yolos) when yolos != [] do
    Enum.map(yolos, fn y ->
      [yindex, _] = String.split(y, "_")

      layer_to_find =
        (String.to_integer(yindex) - 1)
        |> Integer.to_string()
        |> String.pad_leading(3, "0")

      previous_layer = layer_to_find <> "_convolutional"

      if Enum.member?(state.keys, previous_layer) do
        case state.parse_result[previous_layer]["activation"] do
          "linear" -> previous_layer
          "logistic" -> previous_layer <> "_lgx"
          _ -> raise("Unexpected activation: " <> state.parse_result[previous_layer]["activation"])
        end
      else
        y
      end
    end)
  end

  @doc """
    Find number of output classes of the yolo model.
  """
  def get_category_num(state) do
    values = get_in(Map.values(state.parse_result), [Access.all(), "classes"])
    [cn] = Enum.uniq(Enum.filter(values, fn x -> x != nil end))
    cn
  end

  @doc """
  Find input height and width of the yolo model from layer configs.
  """
  def get_h_and_w(state) do
    [state.parse_result["000_net"]["height"], state.parse_result["000_net"]["width"]]
  end
end
