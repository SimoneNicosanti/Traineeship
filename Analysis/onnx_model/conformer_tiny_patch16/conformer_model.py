import onnx
import torch
from conformer_implementation import Conformer  # adjust import to match the repo
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

simplify_model("./conformer_untrained.onnx")
analyze_model("./conformer_untrained_simplified.onnx")
