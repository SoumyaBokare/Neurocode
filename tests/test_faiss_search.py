from vector_db.faiss_index import CodeVectorIndex
import numpy as np

def test_faiss_search():
    index = CodeVectorIndex()
    v1 = np.random.rand(768)
    v2 = np.random.rand(768)
    index.add_vector(v1, "code1")
    index.add_vector(v2, "code2")
    matches = index.search(v1, k=1)
    assert matches[0] == "code1"
