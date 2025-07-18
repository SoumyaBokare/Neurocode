import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import time
import sys

def log_codebert_inference(input_length, latency_ms, model_version="codebert-base"):
    """
    Log CodeBERT inference metrics to MLflow
    
    Args:
        input_length (int): Length of input code
        latency_ms (float): Inference latency in milliseconds
        model_version (str): Model version identifier
    """
    experiment_name = "CodeAnalysisAgent"
    
    try:
        # Set tracking URI (optional - defaults to local)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        # Create or get experiment
        client = MlflowClient()
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = client.create_experiment(experiment_name)
                print(f"Created new experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                print(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            print(f"Error with experiment: {e}")
            return False
            
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"codebert-inference-{int(time.time())}"):
            # Log parameters
            mlflow.log_param("model", model_version)
            mlflow.log_param("input_length", input_length)
            mlflow.log_param("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Log metrics
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("throughput_tokens_per_sec", input_length / (latency_ms / 1000))
            
            # Log tags
            mlflow.set_tag("environment", "production")
            mlflow.set_tag("agent", "CodeAnalysisAgent")
            mlflow.set_tag("model_type", "transformer")
            
            print(f"✅ Successfully logged to MLflow")
            print(f"📊 Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"🔬 Run ID: {mlflow.active_run().info.run_id}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error logging to MLflow: {e}")
        return False

# Enhanced version with more metrics
def log_advanced_codebert_metrics(
    input_length, 
    latency_ms, 
    memory_usage_mb=None,
    confidence_score=None,
    detected_issues=None,
    model_version="codebert-base"
):
    """
    Log comprehensive CodeBERT metrics to MLflow
    """
    experiment_name = "NeuroCodeAssistant"
    
    try:
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"codebert-analysis-{int(time.time())}"):
            # Basic parameters
            mlflow.log_param("model", model_version)
            mlflow.log_param("input_length", input_length)
            mlflow.log_param("python_version", sys.version.split()[0])
            
            # Performance metrics
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("throughput_tokens_per_sec", input_length / (latency_ms / 1000))
            
            if memory_usage_mb:
                mlflow.log_metric("memory_usage_mb", memory_usage_mb)
            
            if confidence_score:
                mlflow.log_metric("confidence_score", confidence_score)
            
            if detected_issues:
                mlflow.log_metric("detected_issues_count", len(detected_issues))
                # Log issue types as tags
                for i, issue in enumerate(detected_issues[:5]):  # Limit to 5 issues
                    mlflow.set_tag(f"issue_{i+1}", issue)
            
            # Environment tags
            mlflow.set_tag("environment", "development")
            mlflow.set_tag("agent", "CodeAnalysisAgent")
            mlflow.set_tag("model_type", "transformer")
            mlflow.set_tag("task", "code_analysis")
            
            print(f"✅ Advanced metrics logged successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error logging advanced metrics: {e}")
        return False

def log_bug_detection_metrics(
    input_length,
    latency_ms,
    bugs_found,
    bug_types,
    confidence_scores=None,
    agent_version="v1.0"
):
    """
    Log BugDetectionAgent metrics to MLflow
    
    Args:
        input_length (int): Length of input code
        latency_ms (float): Detection latency in milliseconds
        bugs_found (int): Number of bugs detected
        bug_types (list): List of bug types found
        confidence_scores (list): Confidence scores for each bug
        agent_version (str): Agent version identifier
    """
    experiment_name = "BugDetectionAgent"
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        # Create or get experiment
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"bug-detection-{int(time.time())}"):
            # Log parameters
            mlflow.log_param("agent_version", agent_version)
            mlflow.log_param("input_length", input_length)
            mlflow.log_param("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Log metrics
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("bugs_found", bugs_found)
            mlflow.log_metric("bug_detection_rate", bugs_found / max(1, input_length) * 1000)  # bugs per 1000 chars
            
            if confidence_scores:
                mlflow.log_metric("avg_confidence", sum(confidence_scores) / len(confidence_scores))
                mlflow.log_metric("min_confidence", min(confidence_scores))
                mlflow.log_metric("max_confidence", max(confidence_scores))
            
            # Log bug types as tags
            for i, bug_type in enumerate(bug_types[:5]):  # Limit to 5 bug types
                mlflow.set_tag(f"bug_type_{i+1}", bug_type)
            
            # Environment tags
            mlflow.set_tag("environment", "production")
            mlflow.set_tag("agent", "BugDetectionAgent")
            mlflow.set_tag("task", "bug_detection")
            
            print(f"✅ BugDetectionAgent metrics logged successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error logging BugDetectionAgent metrics: {e}")
        return False

def log_documentation_metrics(
    input_length,
    latency_ms,
    docstring_generated,
    comments_generated,
    quality_score=None,
    agent_version="v1.0"
):
    """
    Log DocumentationAgent metrics to MLflow
    
    Args:
        input_length (int): Length of input code
        latency_ms (float): Generation latency in milliseconds
        docstring_generated (bool): Whether docstring was generated
        comments_generated (int): Number of comments generated
        quality_score (float): Quality score of generated documentation
        agent_version (str): Agent version identifier
    """
    experiment_name = "DocumentationAgent"
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        
        # Create or get experiment
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
            print(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"doc-generation-{int(time.time())}"):
            # Log parameters
            mlflow.log_param("agent_version", agent_version)
            mlflow.log_param("input_length", input_length)
            mlflow.log_param("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
            
            # Log metrics
            mlflow.log_metric("latency_ms", latency_ms)
            mlflow.log_metric("docstring_generated", 1 if docstring_generated else 0)
            mlflow.log_metric("comments_generated", comments_generated)
            mlflow.log_metric("documentation_rate", comments_generated / max(1, input_length) * 100)  # comments per 100 chars
            
            if quality_score:
                mlflow.log_metric("quality_score", quality_score)
            
            # Environment tags
            mlflow.set_tag("environment", "production")
            mlflow.set_tag("agent", "DocumentationAgent")
            mlflow.set_tag("task", "documentation_generation")
            
            print(f"✅ DocumentationAgent metrics logged successfully")
            return True
            
    except Exception as e:
        print(f"❌ Error logging DocumentationAgent metrics: {e}")
        return False

# Example usage and testing
if __name__ == "__main__":
    # Test the logging function
    print("Testing MLflow logging...")
    
    # Basic logging
    success = log_codebert_inference(
        input_length=150,
        latency_ms=45.2,
        model_version="codebert-base-v1.0"
    )
    
    if success:
        print("✅ Basic logging test passed")
    
    # Advanced logging
    success = log_advanced_codebert_metrics(
        input_length=200,
        latency_ms=52.8,
        memory_usage_mb=128.5,
        confidence_score=0.94,
        detected_issues=["unused_variable", "potential_bug", "style_issue"],
        model_version="codebert-base-v1.0"
    )
    
    if success:
        print("✅ Advanced logging test passed")
    
    # Bug detection logging
    success = log_bug_detection_metrics(
        input_length=300,
        latency_ms=60.5,
        bugs_found=5,
        bug_types=["syntax_error", "runtime_error"],
        confidence_scores=[0.95, 0.89],
        agent_version="v1.0"
    )
    
    if success:
        print("✅ Bug detection logging test passed")
    
    # Documentation logging
    success = log_documentation_metrics(
        input_length=250,
        latency_ms=55.3,
        docstring_generated=True,
        comments_generated=10,
        quality_score=0.87,
        agent_version="v1.0"
    )
    
    if success:
        print("✅ Documentation logging test passed")
    
    print("\n🚀 MLflow UI should be running at: http://127.0.0.1:5000")