from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import time
from mlops.tracking import log_codebert_inference

class CodeAnalysisAgent:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base", output_attentions=True)
        self.model.eval()

    def analyze(self, code: str):
        start = time.time()
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        latency_ms = (time.time() - start) * 1000
        log_codebert_inference(len(code), latency_ms)  # Log inference details to MLflow
        return embedding

    def analyze_with_attention(self, code: str):
        """
        Analyze code and return both embedding and attention weights
        """
        start = time.time()
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
            embedding = last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            # Get attention weights from last layer
            attentions = outputs.attentions
            attention = attentions[-1][0].mean(0).cpu().numpy()  # Average across heads
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        latency_ms = (time.time() - start) * 1000
        log_codebert_inference(len(code), latency_ms)  # Log inference details to MLflow
        
        return {
            "embedding": embedding,
            "tokens": tokens,
            "attention": attention,
            "latency_ms": latency_ms
        }
