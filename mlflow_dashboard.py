#!/usr/bin/env python3
"""MLflow dashboard setup and management script."""

import mlflow
from mlflow.tracking import MlflowClient
import time
import pandas as pd

def setup_mlflow_dashboard():
    """Setup and display MLflow dashboard information."""
    print("🚀 MLflow Dashboard Setup")
    print("=" * 50)
    
    # Set tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = MlflowClient()
    
    # List all experiments
    experiments = client.search_experiments()
    
    print(f"📊 Found {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        # Get runs for this experiment
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        print(f"    📈 Runs: {len(runs)}")
        
        if runs:
            latest_run = runs[0]  # Most recent run
            print(f"    🕒 Latest run: {latest_run.info.start_time}")
            print(f"    📊 Metrics: {list(latest_run.data.metrics.keys())}")
            print(f"    🏷️  Tags: {list(latest_run.data.tags.keys())}")
        print()
    
    print("🌐 MLflow UI Access:")
    print(f"  URL: http://127.0.0.1:5000")
    print(f"  Status: {'✅ Running' if check_mlflow_server() else '❌ Not running'}")
    print()
    
    print("📈 Recommended Next Steps:")
    print("1. Open MLflow UI in browser: http://127.0.0.1:5000")
    print("2. Compare runs across different experiments")
    print("3. Set up alerts for performance degradation")
    print("4. Export models for deployment")
    print("5. Create automated ML pipelines")

def check_mlflow_server():
    """Check if MLflow server is running."""
    try:
        client = MlflowClient("http://127.0.0.1:5000")
        client.search_experiments()
        return True
    except:
        return False

def export_experiment_summary():
    """Export experiment summary to CSV."""
    print("📄 Exporting experiment summary...")
    
    client = MlflowClient("http://127.0.0.1:5000")
    experiments = client.search_experiments()
    
    summary_data = []
    
    for exp in experiments:
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        
        for run in runs:
            summary_data.append({
                'experiment_name': exp.name,
                'run_id': run.info.run_id,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'status': run.info.status,
                'latency_ms': run.data.metrics.get('latency_ms', 0),
                'input_length': run.data.params.get('input_length', 0),
                'agent': run.data.tags.get('agent', 'unknown')
            })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        df.to_csv('mlflow_experiment_summary.csv', index=False)
        print(f"✅ Exported {len(summary_data)} runs to 'mlflow_experiment_summary.csv'")
    else:
        print("❌ No runs found to export")

def performance_analysis():
    """Analyze performance across all agents."""
    print("📊 Performance Analysis")
    print("=" * 30)
    
    client = MlflowClient("http://127.0.0.1:5000")
    experiments = client.search_experiments()
    
    agent_performance = {}
    
    for exp in experiments:
        if exp.name in ['CodeAnalysisAgent', 'BugDetectionAgent', 'DocumentationAgent']:
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            
            latencies = []
            for run in runs:
                if 'latency_ms' in run.data.metrics:
                    latencies.append(run.data.metrics['latency_ms'])
            
            if latencies:
                agent_performance[exp.name] = {
                    'avg_latency': sum(latencies) / len(latencies),
                    'min_latency': min(latencies),
                    'max_latency': max(latencies),
                    'total_runs': len(latencies)
                }
    
    for agent, stats in agent_performance.items():
        print(f"\n🤖 {agent}:")
        print(f"  📊 Total runs: {stats['total_runs']}")
        print(f"  ⚡ Avg latency: {stats['avg_latency']:.2f} ms")
        print(f"  🚀 Min latency: {stats['min_latency']:.2f} ms")
        print(f"  🐌 Max latency: {stats['max_latency']:.2f} ms")

if __name__ == "__main__":
    setup_mlflow_dashboard()
    print("\n" + "=" * 50)
    performance_analysis()
    print("\n" + "=" * 50)
    export_experiment_summary()
