defmodule DarknetToOnnx.Learning do
  def update_map(parse_result, f) do
    inner_update_map(parse_result, Enum.count(parse_result) - 1, f)
  end

  defp inner_update_map(state, -1, _f) do
    state
  end

  defp inner_update_map(state, n_section, f) do
    {k, v} = Enum.at(state, n_section)
    new_kv = f.(state, k, v)
    inner_update_map(Map.put(state, k, new_kv), n_section - 1, f)
  end

  def add_type_key(state, k, _v) do
    [_, type] = String.split(k, "_")
    put_in(state[k], [Access.key("type", %{})], type)
  end

  def update_map_adding_type(parse_result) do
    update_map(parse_result, &add_type_key/3)
  end

  @doc """
       Concatenate and flatten
       Returns a list obtained from the concatenation of the two input lists.
       The result is then flattened so the result is a plain list.
  """
  def cfl(srclist, newarray) do
    Enum.concat([srclist], [newarray]) |> List.flatten()
  end

  @doc """
        Check whether a variable is enumerable
  """
  def is_enum?(var) do
    if is_list(var) or
         (is_map(var) and not Map.has_key?(var, :__struct__)) or
         is_tuple(var) do
      true
    else
      false
    end
  end
end
