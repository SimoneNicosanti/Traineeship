import onnx
import torch
from torchvision import transforms
from PIL import Image
from urllib.request import urlopen
import timm

import os

import model_analyze_utils

os.makedirs("./vitbase_patch_16_240", exist_ok=True)



# --- Load Image ---


# onnx_model = onnx.load("crossvit_9_240.onnx")





def export_model() :
    # --- Load image ---
    img = Image.open(urlopen(
        'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    ))

    # --- Load model ---
    model = timm.create_model('vit_base_patch16_clip_384.openai_ft_in1k', pretrained=True)
    model.eval()

    # --- Get model-specific transforms ---
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # --- Preprocess image and create dummy input ---
    img_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, H, W)

    # --- Export to ONNX ---
    torch.onnx.export(
        model,
        img_tensor,
        "./vitbase_patch_16_240/vit_base_clip384.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17
    )

    print("✅ Exported model to vit_base_clip384.onnx")

    



def main() :
    export_model()
    model_analyze_utils.simplify_model("./vitbase_patch_16_240/vit_base_clip384.onnx")
    model_analyze_utils.analyze_model("./vitbase_patch_16_240/vit_base_clip384_simplified.onnx")

if __name__ == "__main__" :
    main()
