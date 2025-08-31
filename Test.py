from urllib.request import urlopen
from PIL import Image
import timm
import torch
img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('hrnet_w18.ms_aug_in1k', pretrained=True)
# model.save_pretrained('./hrnet_w18.ms_aug_in1k')
img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)
dummy_input = transforms(img).unsqueeze(0)

# Export to ONNX
onnx_path = "hrnet_w18.onnx"
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print(f"✅ ONNX model saved to: {onnx_path}")
