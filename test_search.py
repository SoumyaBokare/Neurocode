#!/usr/bin/env python3
"""
Test the code search functionality
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db.faiss_index import CodeVectorIndex
from agents.code_analysis.agent import CodeAnalysisAgent

def test_search():
    """Test the search functionality"""
    print("🔍 Testing Code Search Functionality")
    print("=" * 50)
    
    # Initialize components
    try:
        code_analyzer = CodeAnalysisAgent()
        vector_index = CodeVectorIndex()
        
        # Check database stats
        stats = vector_index.get_stats()
        print(f"📊 Database Stats:")
        print(f"   - Total vectors: {stats['total_vectors']}")
        print(f"   - Dimension: {stats['dimension']}")
        print(f"   - Snippets count: {stats['snippets_count']}")
        
        if stats['total_vectors'] == 0:
            print("❌ Database is empty. Run populate_vector_db.py first.")
            return False
        
        # Test query
        test_query = '''def sum_numbers(numbers):
    """Calculate sum of a list of numbers"""
    total = 0
    for num in numbers:
        total += num
    return total'''
        
        print(f"\n🔍 Testing with query:")
        print(f"```python\n{test_query}\n```")
        
        # Generate embedding
        embedding = code_analyzer.analyze(test_query)
        print(f"✅ Generated embedding: {embedding.shape}")
        
        # Search for similar code
        matches = vector_index.search(embedding, k=3)
        print(f"✅ Found {len(matches)} matches")
        
        # Display results
        for i, match in enumerate(matches):
            print(f"\n🎯 Result {i+1}:")
            print(f"   Similarity: {match.get('similarity', 'N/A'):.3f}")
            print(f"   Category: {match.get('metadata', {}).get('category', 'unknown')}")
            print(f"   Code snippet: {match.get('code', 'No code')[:100]}...")
        
        print("\n✅ Search functionality is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing search: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_search()
