#!/usr/bin/env python3
"""Test script to verify CodeAnalysisAgent MLflow integration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.code_analysis.agent import CodeAnalysisAgent

def test_agent_mlflow():
    """Test the CodeAnalysisAgent with MLflow logging."""
    print("Testing CodeAnalysisAgent with MLflow logging...")
    
    # Create agent
    agent = CodeAnalysisAgent()
    
    # Test code samples
    test_codes = [
        "def hello_world():\n    print('Hello, World!')",
        "import numpy as np\ndef calculate_mean(data):\n    return np.mean(data)",
        "class Calculator:\n    def add(self, a, b):\n        return a + b"
    ]
    
    # Analyze each code sample
    for i, code in enumerate(test_codes):
        print(f"\n--- Testing code sample {i+1} ---")
        print(f"Code: {code}")
        
        # This will trigger MLflow logging
        embedding = agent.analyze(code)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding sample: {embedding[:5]}")
    
    print("\n✅ All tests completed! Check MLflow UI at http://127.0.0.1:5000")

if __name__ == "__main__":
    test_agent_mlflow()
