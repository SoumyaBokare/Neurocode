import onnxruntime as ort
import numpy as np

class CodeBERTONNX:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)

    def infer(self, input_ids, attention_mask):
        inputs = {
            'input_ids': input_ids.astype(np.int64),
            'attention_mask': attention_mask.astype(np.int64)
        }
        outputs = self.session.run(None, inputs)
        return outputs[0]  # last_hidden_state
