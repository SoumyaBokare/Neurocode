# vector_db/faiss_index.py
import faiss
import numpy as np
import os
import pickle

class CodeVectorIndex:
    def __init__(self, dim=768):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.snippets = []
        self.metadata = []
        self.index_file = "vector_db/code_index.faiss"
        self.snippets_file = "vector_db/code_snippets.pkl"
        
        # Load existing index if available
        self.load_index()
        
        # Add sample data if index is empty
        if self.index.ntotal == 0:
            self.add_sample_data()

    def add_vector(self, vector: np.ndarray, code: str, metadata=None):
        """Add a vector and its associated code to the index"""
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match expected {self.dim}")
        
        self.index.add(np.array([vector]))
        self.snippets.append(code)
        self.metadata.append(metadata or {})

    def search(self, vector: np.ndarray, k=5):
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return []
        
        if vector.shape[0] != self.dim:
            raise ValueError(f"Query vector dimension {vector.shape[0]} doesn't match expected {self.dim}")
        
        # Ensure k doesn't exceed available vectors
        k = min(k, self.index.ntotal)
        
        try:
            D, I = self.index.search(np.array([vector]), k)
            results = []
            
            for i, idx in enumerate(I[0]):
                if idx >= 0 and idx < len(self.snippets):
                    results.append({
                        'code': self.snippets[idx],
                        'similarity': float(1.0 / (1.0 + D[0][i])),  # Convert distance to similarity
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {}
                    })
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def save_index(self):
        """Save the index and snippets to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
            faiss.write_index(self.index, self.index_file)
            
            with open(self.snippets_file, 'wb') as f:
                pickle.dump({'snippets': self.snippets, 'metadata': self.metadata}, f)
        except Exception as e:
            print(f"Error saving index: {e}")

    def load_index(self):
        """Load the index and snippets from disk"""
        try:
            if os.path.exists(self.index_file) and os.path.exists(self.snippets_file):
                self.index = faiss.read_index(self.index_file)
                
                with open(self.snippets_file, 'rb') as f:
                    data = pickle.load(f)
                    self.snippets = data.get('snippets', [])
                    self.metadata = data.get('metadata', [])
                
                print(f"Loaded {self.index.ntotal} vectors from disk")
        except Exception as e:
            print(f"Error loading index: {e}")
            # Reset to empty index
            self.index = faiss.IndexFlatL2(self.dim)
            self.snippets = []
            self.metadata = []

    def add_sample_data(self):
        """Add sample code snippets to the index"""
        # We'll add this after we can generate embeddings
        pass

    def get_stats(self):
        """Get statistics about the index"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dim,
            'snippets_count': len(self.snippets)
        }
