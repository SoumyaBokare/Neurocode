import numpy as np
import matplotlib.pyplot as plt
import io
import base64

from transformers import AutoTokenizer, AutoModel
import torch

def get_attention(code: str, layer_idx=-1, head_idx=None):
    """
    Get attention weights for a given code snippet
    
    Args:
        code: The code string to analyze
        layer_idx: Which layer to extract attention from (-1 for last layer)
        head_idx: Which attention head to use (None for mean across all heads)
    
    Returns:
        tokens: List of tokens
        attention: Attention matrix
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base", output_attentions=True)
        model.eval()
        
        inputs = tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            attentions = outputs.attentions
            
            # Get specific layer attention
            attn_layer = attentions[layer_idx]  # Shape: (batch_size, num_heads, seq_len, seq_len)
            
            # Average across heads if head_idx is None
            if head_idx is None:
                attn = attn_layer[0].mean(0).cpu().numpy()  # Mean across heads
            else:
                attn = attn_layer[0][head_idx].cpu().numpy()  # Specific head
            
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            
        return tokens, attn
    
    except Exception as e:
        print(f"Error in get_attention: {e}")
        return [], np.array([])

def plot_attention(tokens, attention, title="Attention Map"):
    """
    Create a visualization of attention weights
    
    Args:
        tokens: List of tokens
        attention: Attention matrix
        title: Title for the plot
    
    Returns:
        Base64 encoded image string
    """
    try:
        if len(tokens) == 0 or attention.size == 0:
            return ""
            
        plt.figure(figsize=(12, 8))
        
        # Create heatmap using matplotlib
        plt.imshow(attention, cmap='viridis', aspect='auto')
        plt.colorbar(label='Attention Weight')
        
        # Set ticks and labels
        plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
        plt.yticks(range(len(tokens)), tokens)
        
        plt.title(title)
        plt.xlabel('Tokens')
        plt.ylabel('Tokens')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64
        
    except Exception as e:
        print(f"Error in plot_attention: {e}")
        return ""

def get_token_importance(tokens, attention):
    """
    Calculate importance scores for each token based on attention weights
    
    Args:
        tokens: List of tokens
        attention: Attention matrix
    
    Returns:
        Dictionary mapping tokens to importance scores
    """
    if len(tokens) == 0 or attention.size == 0:
        return {}
        
    # Sum attention weights received by each token
    token_importance = attention.sum(axis=0)
    
    # Normalize
    token_importance = token_importance / token_importance.sum()
    
    return {token: float(importance) for token, importance in zip(tokens, token_importance)}

def analyze_code_attention(code: str):
    """
    Comprehensive attention analysis for code
    
    Args:
        code: The code string to analyze
    
    Returns:
        Dictionary with tokens, attention matrix, importance scores, and visualization
    """
    tokens, attention = get_attention(code)
    
    if len(tokens) == 0:
        return {
            "tokens": [],
            "attention": [],
            "importance": {},
            "visualization": "",
            "error": "Failed to analyze attention"
        }
    
    importance = get_token_importance(tokens, attention)
    visualization = plot_attention(tokens, attention)
    
    return {
        "tokens": tokens,
        "attention": attention.tolist(),
        "importance": importance,
        "visualization": visualization,
        "error": None
    }
