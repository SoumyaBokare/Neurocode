from agents.code_analysis.agent import CodeAnalysisAgent

def test_code_analysis():
    agent = CodeAnalysisAgent()
    code = "def add(a, b): return a + b"
    embedding = agent.analyze(code)
    assert embedding is not None
    assert len(embedding.shape) == 1
    assert embedding.shape[0] == 768
