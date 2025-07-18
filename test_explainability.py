#!/usr/bin/env python3
"""
Test script for explainability features in NeuroCode Assistant
Tests attention visualization, token importance, and integration with API endpoints.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import requests
import json
from explainability.attention_map import analyze_code_attention, get_attention, plot_attention
from agents.code_analysis.agent import CodeAnalysisAgent
from api.main import app
import uvicorn
import threading
import time

def test_attention_extraction():
    """Test basic attention extraction functionality"""
    print("=" * 50)
    print("Testing attention extraction...")
    
    test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""
    
    try:
        tokens, attention = get_attention(test_code)
        print(f"✓ Successfully extracted attention")
        print(f"  - Number of tokens: {len(tokens)}")
        print(f"  - Attention matrix shape: {attention.shape}")
        print(f"  - Sample tokens: {tokens[:10]}...")
        
        # Test token importance
        result = analyze_code_attention(test_code)
        print(f"✓ Comprehensive analysis completed")
        print(f"  - Importance scores calculated for {len(result['importance'])} tokens")
        
        return True
    except Exception as e:
        print(f"✗ Error in attention extraction: {e}")
        return False

def test_visualization():
    """Test attention visualization"""
    print("\n" + "=" * 50)
    print("Testing attention visualization...")
    
    test_code = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""
    
    try:
        tokens, attention = get_attention(test_code)
        
        if len(tokens) > 0:
            img_base64 = plot_attention(tokens, attention, "Quick Sort Attention")
            print(f"✓ Visualization generated successfully")
            print(f"  - Image encoded as base64: {len(img_base64)} characters")
            
            # Test comprehensive analysis
            result = analyze_code_attention(test_code)
            if result["visualization"]:
                print(f"✓ Comprehensive analysis with visualization completed")
                print(f"  - Top 5 important tokens:")
                sorted_importance = sorted(result["importance"].items(), key=lambda x: x[1], reverse=True)
                for token, importance in sorted_importance[:5]:
                    print(f"    {token}: {importance:.4f}")
            else:
                print(f"✗ Comprehensive analysis failed")
                return False
        else:
            print(f"✗ No tokens extracted")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Error in visualization: {e}")
        return False

def test_agent_integration():
    """Test integration with CodeAnalysisAgent"""
    print("\n" + "=" * 50)
    print("Testing CodeAnalysisAgent integration...")
    
    test_code = """
class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
    
    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)
"""
    
    try:
        agent = CodeAnalysisAgent()
        
        # Test regular analysis
        embedding = agent.analyze(test_code)
        print(f"✓ Regular analysis completed")
        print(f"  - Embedding shape: {embedding.shape}")
        
        # Test analysis with attention
        result = agent.analyze_with_attention(test_code)
        print(f"✓ Analysis with attention completed")
        print(f"  - Embedding shape: {result['embedding'].shape}")
        print(f"  - Number of tokens: {len(result['tokens'])}")
        print(f"  - Attention matrix shape: {result['attention'].shape}")
        print(f"  - Latency: {result['latency_ms']:.2f}ms")
        
        return True
    except Exception as e:
        print(f"✗ Error in agent integration: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints for explainability"""
    print("\n" + "=" * 50)
    print("Testing API endpoints...")
    
    test_code = """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
"""
    
    # Start API server in a separate thread
    def run_server():
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        base_url = "http://127.0.0.1:8000"
        
        # Test analyze endpoint
        response = requests.post(f"{base_url}/analyze", 
                               json={"code": test_code})
        if response.status_code == 200:
            print("✓ /analyze endpoint working")
            result = response.json()
            print(f"  - Embedding vector length: {len(result['vector'])}")
        else:
            print(f"✗ /analyze endpoint failed: {response.status_code}")
            return False
        
        # Test analyze-with-attention endpoint
        response = requests.post(f"{base_url}/analyze-with-attention", 
                               json={"code": test_code})
        if response.status_code == 200:
            print("✓ /analyze-with-attention endpoint working")
            result = response.json()
            print(f"  - Embedding vector length: {len(result['vector'])}")
            print(f"  - Number of tokens: {len(result['tokens'])}")
            print(f"  - Latency: {result['latency_ms']:.2f}ms")
        else:
            print(f"✗ /analyze-with-attention endpoint failed: {response.status_code}")
            return False
        
        # Test analyze-attention endpoint
        response = requests.post(f"{base_url}/analyze-attention", 
                               json={"code": test_code})
        if response.status_code == 200:
            print("✓ /analyze-attention endpoint working")
            result = response.json()
            print(f"  - Number of tokens: {len(result['tokens'])}")
            print(f"  - Attention matrix shape: {len(result['attention'])}x{len(result['attention'][0]) if result['attention'] else 0}")
            print(f"  - Visualization generated: {'Yes' if result['visualization'] else 'No'}")
            print(f"  - Token importance calculated: {len(result['importance'])} tokens")
        else:
            print(f"✗ /analyze-attention endpoint failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error in API testing: {e}")
        return False

def generate_test_report():
    """Generate a comprehensive test report"""
    print("\n" + "=" * 50)
    print("EXPLAINABILITY TEST REPORT")
    print("=" * 50)
    
    tests = [
        ("Attention Extraction", test_attention_extraction),
        ("Visualization", test_visualization),
        ("Agent Integration", test_agent_integration),
        ("API Endpoints", test_api_endpoints),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All explainability features are working correctly!")
        print("✓ Attention extraction implemented")
        print("✓ Visualization working")
        print("✓ Agent integration complete")
        print("✓ API endpoints functional")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    print("NeuroCode Assistant - Explainability Feature Test")
    print("=" * 50)
    
    success = generate_test_report()
    
    if success:
        print("\n🚀 Ready to move to the next development phase!")
        print("Consider integrating with:")
        print("- MLflow for attention weight logging")
        print("- Frontend visualization components")
        print("- Real-time attention analysis in the UI")
    else:
        print("\n🔧 Please fix the failing tests before proceeding.")
    
    sys.exit(0 if success else 1)
