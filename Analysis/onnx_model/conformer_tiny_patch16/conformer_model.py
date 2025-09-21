import onnx
import torch
from conformer_implementation import Conformer  # adjust import to match the repo
from onnx import helper
from onnxsim import simplify


def simplify_model(model_path: str):
    # Simplify
    model_simplified, check = simplify(model_path)

    assert check, "Simplified ONNX model could not be validated"

    out_path = model_path.replace(".onnx", "_simplified.onnx")
    # Save simplified model
    onnx.save(model_simplified, out_path)

    print(f"✅ Simplified model saved as {out_path}")


def extract_shape(tensor):
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        dim_str = str(dim)
        if not any(ch.isdigit() for ch in dim_str):
            shape.append(-1)
        else:
            shape.append(dim.dim_value)
    return shape


def analyze_model(model_path: str):
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    print("Number of Nodes >> ", len(model.graph.node))
    print("Number of Inputs >> ", len(model.graph.input))
    for input in model.graph.input:
        print("\t Input Name >> ", input.name)
        print("\t Input Shape >> ", extract_shape(input))
    print("Number of Outputs >> ", len(model.graph.output))
    for output in model.graph.output:
        print("\t Output Name >> ", output.name)
        print("\t Output Shape >> ", extract_shape(output))


def duplicate_shared_initializers(model_path, output_path):
    model = onnx.load(model_path)
    graph = model.graph

    # Count how many times each initializer is used
    usage_count = {}
    for node in graph.node:
        for inp in node.input:
            usage_count[inp] = usage_count.get(inp, 0) + 1

    # Map name -> initializer
    initializer_map = {init.name: init for init in graph.initializer}

    # For each node, if its input is shared, create a copy
    for node in graph.node:
        new_inputs = []
        for inp in node.input:
            if inp in initializer_map and usage_count[inp] > 1:
                # Clone initializer with a new name
                new_name = inp + "_for_" + node.name
                new_init = helper.make_tensor(
                    name=new_name,
                    data_type=initializer_map[inp].data_type,
                    dims=initializer_map[inp].dims,
                    vals=onnx.numpy_helper.to_array(initializer_map[inp])
                    .flatten()
                    .tolist(),
                )
                graph.initializer.append(new_init)
                new_inputs.append(new_name)

                # Decrease usage count for the original
                usage_count[inp] -= 1
            else:
                new_inputs.append(inp)
        node.input[:] = new_inputs

    onnx.save(model, output_path)
    print(f"Saved de-duplicated ONNX model to {output_path}")


model = Conformer()  # variant depends on repo
model.eval()  # no checkpoint loaded

# Dummy input (ImageNet default 224x224, RGB)
dummy_input = torch.randn(1, 3, 224, 224)

# Export ONNX
torch.onnx.export(
    model,
    dummy_input,
    "conformer_untrained.onnx",
    export_params=True,  # will export the (random) parameters
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)


print("Exported conformer_untrained.onnx successfully!")

duplicate_shared_initializers(
    "./conformer_untrained.onnx", "./conformer_untrained.onnx"
)

simplify_model("./conformer_untrained.onnx")
analyze_model("./conformer_untrained_simplified.onnx")
