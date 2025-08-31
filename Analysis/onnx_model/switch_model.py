import os
import torch
from transformers import AutoTokenizer, SwitchTransformersForConditionalGeneration
import onnx
import model_analyze_utils

os.makedirs("./switch-base-8", exist_ok=True)


def export_model() :


    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    model = SwitchTransformersForConditionalGeneration.from_pretrained("google/switch-base-8")
    model.eval()

    # --- Example input ---
    input_text = "A <extra_id_0> walks into a bar."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    # --- Create dummy decoder input (e.g., start token) ---
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

    # --- ONNX export wrapper ---
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids, decoder_input_ids):
            # return logits
            outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
            return outputs.logits

    wrapper = OnnxWrapper(model)

    # --- Export ---
    torch.onnx.export(
        wrapper,
        (input_ids, decoder_input_ids),
        "./switch-base-8/switch_base8.onnx",
        input_names=["input_ids", "decoder_input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "decoder_input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"}
        },
        opset_version=17
    )

    onnx_model = onnx.load("./switch-base-8/switch_base8.onnx")
    onnx.save_model(onnx_model, "./switch-base-8/switch_base8_structure.onnx", save_as_external_data=True)




def main() :
    # export_model()
    model_analyze_utils.simplify_model("./switch-base-8/switch_base8_structure.onnx")
    model_analyze_utils.analyze_model("./switch-base-8/switch_base8_structure_simpl.onnx")

if __name__ == "__main__" :
    main()