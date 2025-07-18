#!/usr/bin/env python3
"""
Test API endpoints for explainability features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import json
import time

def test_api_endpoints():
    """Test the API endpoints"""
    print("Testing API endpoints for explainability...")
    
    # Test data
    test_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
"""
    
    base_url = "http://127.0.0.1:8001"
    
    # Test basic analyze endpoint
    try:
        print("Testing /analyze endpoint...")
        response = requests.post(f"{base_url}/analyze", 
                               json={"code": test_code},
                               timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ /analyze working - vector length: {len(result['vector'])}")
        else:
            print(f"✗ /analyze failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ /analyze error: {e}")
        return False
    
    # Test analyze-with-attention endpoint
    try:
        print("Testing /analyze-with-attention endpoint...")
        response = requests.post(f"{base_url}/analyze-with-attention", 
                               json={"code": test_code},
                               timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ /analyze-with-attention working:")
            print(f"  - Vector length: {len(result['vector'])}")
            print(f"  - Tokens: {len(result['tokens'])}")
            print(f"  - Attention shape: {len(result['attention'])}x{len(result['attention'][0]) if result['attention'] else 0}")
            print(f"  - Latency: {result['latency_ms']:.2f}ms")
        else:
            print(f"✗ /analyze-with-attention failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ /analyze-with-attention error: {e}")
        return False
    
    # Test comprehensive attention analysis
    try:
        print("Testing /analyze-attention endpoint...")
        response = requests.post(f"{base_url}/analyze-attention", 
                               json={"code": test_code},
                               timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ /analyze-attention working:")
            print(f"  - Tokens: {len(result['tokens'])}")
            print(f"  - Attention matrix: {len(result['attention'])}x{len(result['attention'][0]) if result['attention'] else 0}")
            print(f"  - Importance scores: {len(result['importance'])}")
            print(f"  - Visualization: {'✓' if result['visualization'] else '✗'}")
        else:
            print(f"✗ /analyze-attention failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ /analyze-attention error: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("API Explainability Test")
    print("=" * 30)
    print("Note: Make sure the API server is running on http://127.0.0.1:8000")
    print("Run: uvicorn api.main:app --host 127.0.0.1 --port 8000")
    print("=" * 30)
    
    # Wait a moment for user to read
    time.sleep(2)
    
    success = test_api_endpoints()
    
    if success:
        print("\n🎉 All API endpoints working!")
        print("✓ Basic analysis endpoint")
        print("✓ Analysis with attention endpoint")
        print("✓ Comprehensive attention analysis endpoint")
        print("\nStep 14 API integration completed successfully!")
    else:
        print("\n⚠️  Some API endpoints failed")
        print("Please check if the server is running and accessible")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
