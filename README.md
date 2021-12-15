# DarknetToOnnx

This is a partial port of yolo_to_onnx from jkjung-avt/tensorrt_demos/tensorrt_demos/yolo.
At the moment I tested it with yolov3-tiny and yolov3 models (cfg and weights files).

## Usage
I have some problem in converting this app to escript. It seems there is some problem with axon library. I am still working on it.
So at the moment the only way to use it as cli-like is:

`mix run -e 'DarknetToOnnx.CLI.main(["-m", model_path_and_model_name])'`

The model_path_and_prefix must contain the full path (absolute or relative) to the directory containing the config and weights files.
The model name must be in the form: `<yolo_model>-<dimension>`.

For more details on allowed parameters, please use:
`mix run -e 'DarknetToOnnx.CLI.main(["-h"])'`

The output will be a file with same `model_path_and_model_name` with onnx extension.

### Usage example
```bash
$ ls YoloWeights
coco.names          yolov3-tiny-416.weights      yolov3-tiny-416.cfg
$ mix run -e 'DarknetToOnnx.CLI.main(["-m", "./YoloWeights/yolov3-tiny-416"])'
[some outputs...]
$ ls YoloWeights
coco.names          yolov3-tiny-416.weights      yolov3-tiny-416.cfg      yolov3-tiny-416.onnx 
$
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/darknet_to_onnx](https://hexdocs.pm/darknet_to_onnx).

