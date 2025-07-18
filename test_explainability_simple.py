#!/usr/bin/env python3
"""
Simple test for explainability features in NeuroCode Assistant
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from explainability.attention_map import get_attention, plot_attention, analyze_code_attention
from agents.code_analysis.agent import CodeAnalysisAgent

def test_basic_attention():
    """Test basic attention extraction"""
    print("Testing basic attention extraction...")
    
    test_code = """
def add(a, b):
    return a + b
"""
    
    try:
        tokens, attention = get_attention(test_code)
        print(f"✓ Attention extracted successfully")
        print(f"  - Tokens: {len(tokens)}")
        print(f"  - Attention shape: {attention.shape}")
        print(f"  - Sample tokens: {tokens[:5]}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_visualization():
    """Test attention visualization"""
    print("\nTesting attention visualization...")
    
    test_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
    
    try:
        tokens, attention = get_attention(test_code)
        if len(tokens) > 0:
            img_base64 = plot_attention(tokens, attention)
            print(f"✓ Visualization created successfully")
            print(f"  - Image size: {len(img_base64)} chars")
            return True
        else:
            print("✗ No tokens to visualize")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_comprehensive_analysis():
    """Test comprehensive analysis"""
    print("\nTesting comprehensive analysis...")
    
    test_code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
    
    try:
        result = analyze_code_attention(test_code)
        print(f"✓ Comprehensive analysis completed")
        print(f"  - Tokens: {len(result['tokens'])}")
        print(f"  - Attention matrix: {len(result['attention'])}x{len(result['attention'][0]) if result['attention'] else 0}")
        print(f"  - Importance scores: {len(result['importance'])}")
        print(f"  - Visualization: {'✓' if result['visualization'] else '✗'}")
        
        # Show top important tokens
        if result['importance']:
            sorted_importance = sorted(result['importance'].items(), key=lambda x: x[1], reverse=True)
            print(f"  - Top 3 important tokens:")
            for token, importance in sorted_importance[:3]:
                print(f"    '{token}': {importance:.4f}")
        
        return result['error'] is None
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_agent_integration():
    """Test CodeAnalysisAgent integration"""
    print("\nTesting agent integration...")
    
    test_code = """
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        return self.items.pop()
"""
    
    try:
        agent = CodeAnalysisAgent()
        
        # Test regular analysis
        embedding = agent.analyze(test_code)
        print(f"✓ Regular analysis: {embedding.shape}")
        
        # Test analysis with attention
        result = agent.analyze_with_attention(test_code)
        print(f"✓ Analysis with attention:")
        print(f"  - Embedding: {result['embedding'].shape}")
        print(f"  - Tokens: {len(result['tokens'])}")
        print(f"  - Attention: {result['attention'].shape}")
        print(f"  - Latency: {result['latency_ms']:.2f}ms")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("NeuroCode Assistant - Explainability Simple Test")
    print("=" * 50)
    
    tests = [
        ("Basic Attention", test_basic_attention),
        ("Visualization", test_visualization),
        ("Comprehensive Analysis", test_comprehensive_analysis),
        ("Agent Integration", test_agent_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All explainability features working!")
        print("✓ Attention extraction ✓")
        print("✓ Visualization ✓")
        print("✓ Comprehensive analysis ✓")
        print("✓ Agent integration ✓")
        print("\nStep 14 completed successfully!")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
