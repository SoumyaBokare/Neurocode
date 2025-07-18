from agents.code_analysis.agent import CodeAnalysisAgent
from agents.bug_detection.agent import BugDetectionAgent

def route_code_task(code: str, task: str):
    if task == "analyze":
        return CodeAnalysisAgent().analyze(code)
    elif task == "predict_bugs":
        return BugDetectionAgent().predict(code)
    else:
        return {"error": "Unknown task"}
