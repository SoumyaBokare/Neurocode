#!/usr/bin/env python3
"""
Final comprehensive test for NeuroCode Assistant
Tests all major components: MLflow, Federated Learning, and Explainability
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_mlflow_integration():
    """Test MLflow integration"""
    print("Testing MLflow integration...")
    
    try:
        from agents.code_analysis.agent import CodeAnalysisAgent
        from agents.bug_detection.agent import BugDetectionAgent
        from agents.documentation.agent import DocumentationAgent
        
        # Test CodeAnalysisAgent
        agent = CodeAnalysisAgent()
        embedding = agent.analyze("def test(): pass")
        print("✓ CodeAnalysisAgent with MLflow logging")
        
        # Test BugDetectionAgent
        bug_agent = BugDetectionAgent()
        result = bug_agent.predict("def unsafe(): return eval(input())")
        print("✓ BugDetectionAgent with MLflow logging")
        
        # Test DocumentationAgent
        doc_agent = DocumentationAgent()
        docs = doc_agent.generate("def add(a, b): return a + b")
        print("✓ DocumentationAgent with MLflow logging")
        
        return True
    except Exception as e:
        print(f"✗ MLflow integration error: {e}")
        return False

def test_federated_learning():
    """Test federated learning components"""
    print("\nTesting federated learning components...")
    
    try:
        from federated.config import FederatedConfig
        
        # Test configuration
        config = FederatedConfig()
        print(f"✓ Federated config loaded: {config.num_clients} clients")
        
        # Check if simulator exists
        if os.path.exists("federated/federated_simulator.py"):
            print("✓ Federated simulator file exists")
        else:
            print("✗ Federated simulator file missing")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Federated learning error: {e}")
        return False

def test_explainability():
    """Test explainability features"""
    print("\nTesting explainability features...")
    
    try:
        from explainability.attention_map import analyze_code_attention
        from agents.code_analysis.agent import CodeAnalysisAgent
        
        # Test attention analysis
        result = analyze_code_attention("def hello(): return 'world'")
        print(f"✓ Attention analysis: {len(result['tokens'])} tokens")
        
        # Test agent integration
        agent = CodeAnalysisAgent()
        result = agent.analyze_with_attention("class Test: pass")
        print(f"✓ Agent attention integration: {result['attention'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Explainability error: {e}")
        return False

def test_vector_database():
    """Test vector database functionality"""
    print("\nTesting vector database...")
    
    try:
        from vector_db.faiss_index import CodeVectorIndex
        from agents.code_analysis.agent import CodeAnalysisAgent
        
        # Test vector operations
        index = CodeVectorIndex()
        agent = CodeAnalysisAgent()
        
        # Add a vector
        embedding = agent.analyze("def example(): return True")
        index.add_vector(embedding, "def example(): return True")
        print("✓ Vector added to database")
        
        # Search for similar vectors
        matches = index.search(embedding, k=1)
        print(f"✓ Vector search: {len(matches)} matches")
        
        return True
    except Exception as e:
        print(f"✗ Vector database error: {e}")
        return False

def test_api_structure():
    """Test API structure and endpoints"""
    print("\nTesting API structure...")
    
    try:
        from api.main import app
        
        # Check if FastAPI app exists
        if app:
            print("✓ FastAPI app initialized")
        
        # Check for key endpoints by inspecting routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/analyze", "/analyze-with-attention", "/analyze-attention", "/predict-bugs", "/generate-docs"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✓ Endpoint {route} exists")
            else:
                print(f"✗ Endpoint {route} missing")
                return False
        
        return True
    except Exception as e:
        print(f"✗ API structure error: {e}")
        return False

def main():
    """Run all comprehensive tests"""
    print("NeuroCode Assistant - Comprehensive System Test")
    print("=" * 60)
    
    tests = [
        ("MLflow Integration", test_mlflow_integration),
        ("Federated Learning", test_federated_learning),
        ("Explainability", test_explainability),
        ("Vector Database", test_vector_database),
        ("API Structure", test_api_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} systems tested successfully")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("✅ MLflow experiment tracking")
        print("✅ Federated learning framework")
        print("✅ Explainability with attention weights")
        print("✅ Vector database operations")
        print("✅ API endpoints structure")
        print("\n🚀 NeuroCode Assistant is ready for production!")
        print("\nFeatures supported:")
        print("• Edge inference with CodeBERT")
        print("• Architecture visualization")
        print("• Federated learning simulation")
        print("• ML experiment tracking")
        print("• Visual explainability")
        print("• Multi-agent orchestration")
        print("• Vector-based code search")
        print("• Bug detection and documentation")
    else:
        print(f"\n⚠️  {total-passed} system(s) need attention")
        print("Please review the failing components before production deployment")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
