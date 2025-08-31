import onnx
import torch
from torchvision import transforms
from PIL import Image
from urllib.request import urlopen
import timm

import os

import model_analyze_utils

os.makedirs("./coatnet_1_rw_224", exist_ok=True)



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


    # --- Load pretrained CrossViT ---
    model = timm.create_model('coat_mini.in1k', pretrained=True)

    input_tensor = transform(img).unsqueeze(0)  # shape: [1, 3, 224, 224]

    # --- Export to ONNX ---
    torch.onnx.export(
        model, 
        torch.randn(1, 3, 224, 224),              # example input
        "./coatnet_1_rw_224/coatnet_1_rw_224.onnx",                 # where to save
        export_params=True,        # store trained weights inside the model
        opset_version=17,          # ONNX opset (17 recommended for new models)
        do_constant_folding=True,  # optimize constants
        input_names=['input'],     # name of input tensor
        output_names=['output'],   # name of output tensor
        dynamic_axes={             # allow variable batch size
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )

    print("✅ Exported to coatnet_1_rw_224.onnx")




def main() :
    export_model()
    model_analyze_utils.simplify_model("./coatnet_1_rw_224/coatnet_1_rw_224.onnx")
    model_analyze_utils.analyze_model("./coatnet_1_rw_224/coatnet_1_rw_224_simplified.onnx")

if __name__ == "__main__" :
    main()
