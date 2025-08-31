import onnx
import torch
from torchvision import transforms
from PIL import Image
from urllib.request import urlopen
import timm

import os

import model_analyze_utils

os.makedirs("./crossvit_9_240", exist_ok=True)



# --- Load Image ---


# onnx_model = onnx.load("crossvit_9_240.onnx")





def export_model() :
    img = Image.open(urlopen(
        'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
    ))

    # --- Preprocessing (same as ImageNet models expect) ---
    transform = transforms.Compose([
        transforms.Resize((240, 240)),       # CrossViT-9-240 expects 240x240
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    img_tensor = transform(img).unsqueeze(0)  # shape: (1, 3, 240, 240)

    # --- Load pretrained CrossViT ---
    model = timm.create_model('crossvit_9_240.in1k', pretrained=True)

    # --- Export to ONNX ---
    torch.onnx.export(
        model,
        img_tensor,                     # dummy input
        "./crossvit_9_240/crossvit_9_240.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17
    )

    print("✅ Exported to crossvit_9_240.onnx")




def main() :
    export_model()
    model_analyze_utils.simplify_model("./crossvit_9_240/crossvit_9_240.onnx")
    model_analyze_utils.analyze_model("./crossvit_9_240/crossvit_9_240_simplified.onnx")

if __name__ == "__main__" :
    main()
