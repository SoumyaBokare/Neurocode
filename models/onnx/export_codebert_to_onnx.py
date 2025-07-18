import torch
from transformers import AutoTokenizer, AutoModel
import os

def export_codebert_onnx(output_path):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    model.eval()
    dummy = tokenizer("def add(a, b): return a + b", return_tensors="pt", truncation=True, padding=True)
    input_names = ["input_ids", "attention_mask"]
    output_names = ["last_hidden_state"]
    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}, "attention_mask": {0: "batch_size", 1: "seq_len"}},
        opset_version=14
    )

if __name__ == "__main__":
    os.makedirs("models/onnx", exist_ok=True)
    export_codebert_onnx("models/onnx/codebert.onnx")
