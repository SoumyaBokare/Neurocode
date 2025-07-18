#!/usr/bin/env python3
"""Test script to verify all agents MLflow integration."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.code_analysis.agent import CodeAnalysisAgent
from agents.bug_detection.agent import BugDetectionAgent
from agents.documentation.agent import DocumentationAgent

def test_all_agents_mlflow():
    """Test all agents with MLflow logging."""
    print("🚀 Testing all agents with MLflow logging...")
    
    # Create agents
    code_agent = CodeAnalysisAgent()
    bug_agent = BugDetectionAgent()
    doc_agent = DocumentationAgent()
    
    # Test code samples with different characteristics
    test_codes = [
        {
            "name": "Simple function",
            "code": "def hello_world():\n    print('Hello, World!')"
        },
        {
            "name": "Function with potential security issues",
            "code": "def process_input():\n    user_input = input()\n    result = eval(user_input)\n    return result"
        },
        {
            "name": "Math function",
            "code": "def add(a, b):\n    return a + b"
        },
        {
            "name": "Complex function with exec",
            "code": "def dynamic_code():\n    code = 'print(\"Dynamic execution\")'\n    exec(code)\n    return True"
        }
    ]
    
    # Test each agent with each code sample
    for i, test_case in enumerate(test_codes):
        print(f"\n{'='*60}")
        print(f"🧪 Testing case {i+1}: {test_case['name']}")
        print(f"📝 Code: {test_case['code']}")
        print(f"{'='*60}")
        
        # Test CodeAnalysisAgent
        print("\n🔍 CodeAnalysisAgent:")
        try:
            embedding = code_agent.analyze(test_case['code'])
            print(f"✅ Embedding shape: {embedding.shape}")
            print(f"📊 Sample values: {embedding[:3]}")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test BugDetectionAgent
        print("\n🐛 BugDetectionAgent:")
        try:
            bugs = bug_agent.predict(test_case['code'])
            print(f"✅ Bugs found: {bugs['count']}")
            if bugs['bugs']:
                for bug in bugs['bugs']:
                    print(f"  - {bug}")
            else:
                print("  - No bugs detected")
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Test DocumentationAgent
        print("\n📚 DocumentationAgent:")
        try:
            docs = doc_agent.generate(test_case['code'])
            print(f"✅ Docstring: {docs['docstring']}")
            print(f"📝 Comments generated: {len(docs['comments'])}")
            for comment in docs['comments']:
                print(f"  - {comment}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print("🎉 All tests completed!")
    print("📊 Check MLflow UI at: http://127.0.0.1:5000")
    print("📈 You should see experiments for:")
    print("  - CodeAnalysisAgent")
    print("  - BugDetectionAgent") 
    print("  - DocumentationAgent")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_all_agents_mlflow()
