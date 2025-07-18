from agents.onnx_inference.codebert_onnx import CodeBERTONNX
import numpy as np

# Load the ONNX model
model = CodeBERTONNX("models/onnx/codebert.onnx")

# Create dummy input (batch size 1, sequence length 10)
input_ids = np.ones((1, 10), dtype=np.int64)
attention_mask = np.ones((1, 10), dtype=np.int64)

# Run inference
output = model.infer(input_ids, attention_mask)
print("ONNX output shape:", output.shape)
