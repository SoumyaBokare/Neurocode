from agents.bug_detection.agent import BugDetectionAgent

def test_bug_detection():
    agent = BugDetectionAgent()
    code = "eval('2+2')\ninput()\nprint('safe')"
    result = agent.predict(code)
    assert result['count'] == 2
    assert any('eval' in bug for bug in result['bugs'])
    assert any('input' in bug for bug in result['bugs'])
