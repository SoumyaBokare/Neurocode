#!/usr/bin/env python3
"""
Quick test for Code Search functionality
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_code_search():
    """Test the complete code search pipeline"""
    try:
        print("🔍 Testing Code Search...")
        
        # Import components
        from vector_db.faiss_index import CodeVectorIndex
        from agents.code_analysis.agent import CodeAnalysisAgent
        
        # Initialize
        vector_index = CodeVectorIndex()
        code_analyzer = CodeAnalysisAgent()
        
        # Test query
        query = '''def calculate_sum(numbers):
    """Calculate sum of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total'''
        
        print(f"📊 Vector DB Stats: {vector_index.get_stats()}")
        print(f"🔍 Query: {query[:50]}...")
        
        # Generate embedding
        embedding = code_analyzer.analyze(query)
        print(f"✅ Embedding generated: {embedding.shape}")
        
        # Search
        results = vector_index.search(embedding, k=3)
        print(f"✅ Found {len(results)} results")
        
        # Display results
        for i, result in enumerate(results):
            print(f"🎯 Result {i+1}:")
            print(f"   Similarity: {result.get('similarity', 'N/A'):.3f}")
            print(f"   Category: {result.get('metadata', {}).get('category', 'unknown')}")
            print(f"   Code: {result.get('code', 'No code')[:60]}...")
            print()
        
        print("✅ Code Search test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_code_search()
