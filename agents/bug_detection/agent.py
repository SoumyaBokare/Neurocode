import time
from mlops.tracking import log_bug_detection_metrics

class BugDetectionAgent:
    def predict(self, code: str) -> dict:
        start = time.time()
        bugs = []
        bug_types = []
        
        if "eval" in code:
            bugs.append("Usage of eval - potential injection risk")
            bug_types.append("injection_risk")
        if "exec" in code:
            bugs.append("Usage of exec - potential security issue")
            bug_types.append("security_issue")
        if "input()" in code:
            bugs.append("Unvalidated user input")
            bug_types.append("unvalidated_input")
        
        latency_ms = (time.time() - start) * 1000
        
        # Log to MLflow
        log_bug_detection_metrics(
            input_length=len(code),
            latency_ms=latency_ms,
            bugs_found=len(bugs),
            bug_types=bug_types,
            confidence_scores=[0.8, 0.9, 0.7][:len(bugs)]  # Mock confidence scores
        )
        
        return {"bugs": bugs, "count": len(bugs), "bug_types": bug_types}
