#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Benchmarking Suite for NeuroCode Assistant
Tests all agents with clean and buggy code samples, measures performance and accuracy
"""

import time
import json
import csv
import os
import sys
import traceback
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.test_samples import get_all_test_samples, BENCHMARK_CONFIG
from agents.code_analysis.agent import CodeAnalysisAgent
from agents.bug_detection.agent import BugDetectionAgent
from agents.documentation.agent import DocumentationAgent
from explainability.attention_map import analyze_code_attention
from mlops.tracking import log_benchmark_results
import mlflow

class BenchmarkRunner:
    """Comprehensive benchmark runner for NeuroCode Assistant"""
    
    def __init__(self, output_dir="benchmark_results"):
        self.output_dir = output_dir
        self.results = []
        self.agents = {}
        self.start_time = None
        self.end_time = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_experiment("NeuroCode_Benchmark")
        
        print(f"🔧 Benchmark output directory: {output_dir}")
    
    def initialize_agents(self):
        """Initialize all agents for testing"""
        print("🤖 Initializing agents...")
        
        try:
            self.agents = {
                'code_analysis': CodeAnalysisAgent(),
                'bug_detection': BugDetectionAgent(),
                'documentation': DocumentationAgent()
            }
            print("✅ All agents initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize agents: {e}")
            return False
    
    def measure_system_resources(self):
        """Measure current system resource usage"""
        process = psutil.Process()
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'memory_percent': process.memory_percent(),
            'threads': process.num_threads()
        }
    
    def benchmark_code_analysis(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark code analysis agent"""
        agent = self.agents['code_analysis']
        code = sample['code']
        
        # Measure resources before
        resources_before = self.measure_system_resources()
        
        # Run analysis
        start_time = time.time()
        try:
            embedding = agent.analyze(code)
            analysis_time = time.time() - start_time
            
            # Measure resources after
            resources_after = self.measure_system_resources()
            
            # Test with attention if possible
            try:
                attention_result = agent.analyze_with_attention(code)
                attention_time = time.time() - start_time - analysis_time
            except Exception:
                attention_result = None
                attention_time = None
            
            return {
                'success': True,
                'analysis_time': analysis_time,
                'embedding_dimensions': embedding.shape[0] if embedding is not None else 0,
                'attention_time': attention_time,
                'attention_tokens': len(attention_result['tokens']) if attention_result else 0,
                'memory_usage': resources_after['memory_mb'] - resources_before['memory_mb'],
                'cpu_usage': resources_after['cpu_percent'],
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'analysis_time': time.time() - start_time,
                'embedding_dimensions': 0,
                'attention_time': None,
                'attention_tokens': 0,
                'memory_usage': 0,
                'cpu_usage': 0,
                'error': str(e)
            }
    
    def benchmark_bug_detection(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark bug detection agent"""
        agent = self.agents['bug_detection']
        code = sample['code']
        expected_bugs = sample.get('expected_bugs', 0)
        
        # Measure resources before
        resources_before = self.measure_system_resources()
        
        # Run bug detection
        start_time = time.time()
        try:
            result = agent.predict(code)
            detection_time = time.time() - start_time
            
            # Measure resources after
            resources_after = self.measure_system_resources()
            
            # Analyze results
            if isinstance(result, dict):
                prediction = result.get('prediction', 'unknown')
                confidence = result.get('confidence', 0)
                bugs_found = len(result.get('bugs', []))
            else:
                prediction = str(result)
                confidence = 0
                bugs_found = 1 if 'bug' in prediction.lower() else 0
            
            # Calculate accuracy
            if expected_bugs > 0:
                # For buggy code, check if bugs were detected
                accuracy = 1.0 if bugs_found > 0 or prediction == 'bug' else 0.0
            else:
                # For clean code, check if no bugs were detected
                accuracy = 1.0 if bugs_found == 0 and prediction != 'bug' else 0.0
            
            return {
                'success': True,
                'detection_time': detection_time,
                'prediction': prediction,
                'confidence': confidence,
                'bugs_found': bugs_found,
                'expected_bugs': expected_bugs,
                'accuracy': accuracy,
                'memory_usage': resources_after['memory_mb'] - resources_before['memory_mb'],
                'cpu_usage': resources_after['cpu_percent'],
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'detection_time': time.time() - start_time,
                'prediction': 'error',
                'confidence': 0,
                'bugs_found': 0,
                'expected_bugs': expected_bugs,
                'accuracy': 0.0,
                'memory_usage': 0,
                'cpu_usage': 0,
                'error': str(e)
            }
    
    def benchmark_documentation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark documentation generation agent"""
        agent = self.agents['documentation']
        code = sample['code']
        
        # Measure resources before
        resources_before = self.measure_system_resources()
        
        # Run documentation generation
        start_time = time.time()
        try:
            result = agent.generate(code)
            generation_time = time.time() - start_time
            
            # Measure resources after
            resources_after = self.measure_system_resources()
            
            # Analyze documentation quality
            doc_length = len(result) if isinstance(result, str) else 0
            has_docstring = '"""' in str(result) or "'''" in str(result)
            has_parameters = 'param' in str(result).lower() or 'arg' in str(result).lower()
            has_returns = 'return' in str(result).lower()
            
            # Simple quality score
            quality_score = sum([
                1 if doc_length > 50 else 0,
                1 if has_docstring else 0,
                1 if has_parameters else 0,
                1 if has_returns else 0
            ]) / 4
            
            return {
                'success': True,
                'generation_time': generation_time,
                'doc_length': doc_length,
                'quality_score': quality_score,
                'has_docstring': has_docstring,
                'has_parameters': has_parameters,
                'has_returns': has_returns,
                'memory_usage': resources_after['memory_mb'] - resources_before['memory_mb'],
                'cpu_usage': resources_after['cpu_percent'],
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'generation_time': time.time() - start_time,
                'doc_length': 0,
                'quality_score': 0.0,
                'has_docstring': False,
                'has_parameters': False,
                'has_returns': False,
                'memory_usage': 0,
                'cpu_usage': 0,
                'error': str(e)
            }
    
    def benchmark_attention_analysis(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark attention analysis"""
        code = sample['code']
        
        # Measure resources before
        resources_before = self.measure_system_resources()
        
        # Run attention analysis
        start_time = time.time()
        try:
            result = analyze_code_attention(code)
            analysis_time = time.time() - start_time
            
            # Measure resources after
            resources_after = self.measure_system_resources()
            
            # Analyze attention results
            tokens_count = len(result.get('tokens', []))
            attention_shape = len(result.get('attention', []))
            importance_scores = len(result.get('importance', {}))
            has_visualization = bool(result.get('visualization'))
            
            # Calculate attention coherence (simple metric)
            attention_coherence = 1.0 if tokens_count > 0 and attention_shape > 0 else 0.0
            
            return {
                'success': True,
                'analysis_time': analysis_time,
                'tokens_count': tokens_count,
                'attention_shape': attention_shape,
                'importance_scores': importance_scores,
                'has_visualization': has_visualization,
                'attention_coherence': attention_coherence,
                'memory_usage': resources_after['memory_mb'] - resources_before['memory_mb'],
                'cpu_usage': resources_after['cpu_percent'],
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'analysis_time': time.time() - start_time,
                'tokens_count': 0,
                'attention_shape': 0,
                'importance_scores': 0,
                'has_visualization': False,
                'attention_coherence': 0.0,
                'memory_usage': 0,
                'cpu_usage': 0,
                'error': str(e)
            }
    
    def run_benchmark(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Run complete benchmark for a single sample"""
        sample_name = sample.get('name', 'unknown')
        sample_type = 'clean' if sample.get('expected_bugs', 0) == 0 else 'buggy'
        
        print(f"📊 Benchmarking: {sample_name} ({sample_type})")
        
        # Run all benchmarks
        results = {
            'sample_name': sample_name,
            'sample_type': sample_type,
            'category': sample.get('category', 'unknown'),
            'complexity': sample.get('complexity', 'unknown'),
            'code_length': len(sample['code']),
            'timestamp': datetime.now().isoformat()
        }
        
        # Code analysis benchmark
        print("  🔍 Testing code analysis...")
        results['code_analysis'] = self.benchmark_code_analysis(sample)
        
        # Bug detection benchmark
        print("  🐛 Testing bug detection...")
        results['bug_detection'] = self.benchmark_bug_detection(sample)
        
        # Documentation benchmark
        print("  📖 Testing documentation...")
        results['documentation'] = self.benchmark_documentation(sample)
        
        # Attention analysis benchmark
        print("  🧠 Testing attention analysis...")
        results['attention'] = self.benchmark_attention_analysis(sample)
        
        return results
    
    def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("🚀 Starting comprehensive benchmark...")
        
        # Initialize agents
        if not self.initialize_agents():
            return False
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            self.start_time = time.time()
            
            # Get all test samples
            samples = get_all_test_samples()
            all_samples = samples['clean'] + samples['buggy']
            
            print(f"📋 Total samples to test: {len(all_samples)}")
            print(f"   - Clean samples: {len(samples['clean'])}")
            print(f"   - Buggy samples: {len(samples['buggy'])}")
            
            # Run benchmarks
            for i, sample in enumerate(all_samples):
                print(f"\n[{i+1}/{len(all_samples)}] " + "="*50)
                
                try:
                    result = self.run_benchmark(sample)
                    self.results.append(result)
                    
                    # Log to MLflow
                    self.log_sample_to_mlflow(result, i)
                    
                except Exception as e:
                    print(f"❌ Error benchmarking {sample.get('name', 'unknown')}: {e}")
                    traceback.print_exc()
            
            self.end_time = time.time()
            
            # Generate reports
            self.generate_reports()
            
            # Log overall metrics to MLflow
            self.log_overall_metrics()
            
            print(f"\n🎉 Benchmark completed in {self.end_time - self.start_time:.2f} seconds")
            print(f"📊 Results saved to: {self.output_dir}")
            
            return True
    
    def log_sample_to_mlflow(self, result: Dict[str, Any], sample_index: int):
        """Log individual sample results to MLflow"""
        prefix = f"sample_{sample_index}"
        
        # Log basic info
        mlflow.log_param(f"{prefix}_name", result['sample_name'])
        mlflow.log_param(f"{prefix}_type", result['sample_type'])
        mlflow.log_param(f"{prefix}_category", result['category'])
        
        # Log code analysis metrics
        if result['code_analysis']['success']:
            mlflow.log_metric(f"{prefix}_analysis_time", result['code_analysis']['analysis_time'])
            mlflow.log_metric(f"{prefix}_embedding_dims", result['code_analysis']['embedding_dimensions'])
        
        # Log bug detection metrics
        if result['bug_detection']['success']:
            mlflow.log_metric(f"{prefix}_detection_time", result['bug_detection']['detection_time'])
            mlflow.log_metric(f"{prefix}_bug_accuracy", result['bug_detection']['accuracy'])
            mlflow.log_metric(f"{prefix}_confidence", result['bug_detection']['confidence'])
        
        # Log documentation metrics
        if result['documentation']['success']:
            mlflow.log_metric(f"{prefix}_doc_time", result['documentation']['generation_time'])
            mlflow.log_metric(f"{prefix}_doc_quality", result['documentation']['quality_score'])
        
        # Log attention metrics
        if result['attention']['success']:
            mlflow.log_metric(f"{prefix}_attention_time", result['attention']['analysis_time'])
            mlflow.log_metric(f"{prefix}_attention_coherence", result['attention']['attention_coherence'])
    
    def log_overall_metrics(self):
        """Log overall benchmark metrics to MLflow"""
        if not self.results:
            return
        
        # Calculate aggregate metrics
        total_time = self.end_time - self.start_time
        total_samples = len(self.results)
        successful_samples = sum(1 for r in self.results if all([
            r['code_analysis']['success'],
            r['bug_detection']['success'],
            r['documentation']['success'],
            r['attention']['success']
        ]))
        
        # Analysis times
        analysis_times = [r['code_analysis']['analysis_time'] for r in self.results if r['code_analysis']['success']]
        bug_detection_times = [r['bug_detection']['detection_time'] for r in self.results if r['bug_detection']['success']]
        doc_times = [r['documentation']['generation_time'] for r in self.results if r['documentation']['success']]
        attention_times = [r['attention']['analysis_time'] for r in self.results if r['attention']['success']]
        
        # Bug detection accuracy
        bug_accuracies = [r['bug_detection']['accuracy'] for r in self.results if r['bug_detection']['success']]
        
        # Log overall metrics
        mlflow.log_metric("total_benchmark_time", total_time)
        mlflow.log_metric("total_samples", total_samples)
        mlflow.log_metric("successful_samples", successful_samples)
        mlflow.log_metric("success_rate", successful_samples / total_samples if total_samples > 0 else 0)
        
        if analysis_times:
            mlflow.log_metric("avg_analysis_time", sum(analysis_times) / len(analysis_times))
            mlflow.log_metric("max_analysis_time", max(analysis_times))
            mlflow.log_metric("min_analysis_time", min(analysis_times))
        
        if bug_detection_times:
            mlflow.log_metric("avg_bug_detection_time", sum(bug_detection_times) / len(bug_detection_times))
        
        if doc_times:
            mlflow.log_metric("avg_documentation_time", sum(doc_times) / len(doc_times))
        
        if attention_times:
            mlflow.log_metric("avg_attention_time", sum(attention_times) / len(attention_times))
        
        if bug_accuracies:
            mlflow.log_metric("avg_bug_detection_accuracy", sum(bug_accuracies) / len(bug_accuracies))
    
    def generate_reports(self):
        """Generate comprehensive reports"""
        print("📄 Generating reports...")
        
        # Generate CSV report
        self.generate_csv_report()
        
        # Generate JSON report
        self.generate_json_report()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_csv_report(self):
        """Generate CSV report"""
        csv_file = os.path.join(self.output_dir, "benchmark_results.csv")
        
        # Flatten results for CSV
        flattened_results = []
        for result in self.results:
            flat_result = {
                'sample_name': result['sample_name'],
                'sample_type': result['sample_type'],
                'category': result['category'],
                'complexity': result['complexity'],
                'code_length': result['code_length'],
                'timestamp': result['timestamp']
            }
            
            # Add all metrics
            for agent_name, agent_results in result.items():
                if isinstance(agent_results, dict) and agent_name not in ['sample_name', 'sample_type', 'category', 'complexity', 'code_length', 'timestamp']:
                    for metric, value in agent_results.items():
                        flat_result[f"{agent_name}_{metric}"] = value
            
            flattened_results.append(flat_result)
        
        # Write CSV
        if flattened_results:
            df = pd.DataFrame(flattened_results)
            df.to_csv(csv_file, index=False)
            print(f"✅ CSV report saved: {csv_file}")
    
    def generate_json_report(self):
        """Generate JSON report"""
        json_file = os.path.join(self.output_dir, "benchmark_results.json")
        
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(self.results),
                'total_time': self.end_time - self.start_time if self.end_time and self.start_time else 0,
                'config': BENCHMARK_CONFIG
            },
            'results': self.results
        }
        
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ JSON report saved: {json_file}")
    
    def generate_visualizations(self):
        """Generate visualization plots"""
        if not self.results:
            return
        
        print("📊 Generating visualizations...")
        
        # Performance comparison chart
        self.create_performance_chart()
        
        # Accuracy chart
        self.create_accuracy_chart()
        
        # Time distribution chart
        self.create_time_distribution_chart()
        
        # Success rate chart
        self.create_success_rate_chart()
    
    def create_performance_chart(self):
        """Create performance comparison chart"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NeuroCode Assistant - Performance Benchmark', fontsize=16)
        
        # Extract data
        clean_samples = [r for r in self.results if r['sample_type'] == 'clean']
        buggy_samples = [r for r in self.results if r['sample_type'] == 'buggy']
        
        # Analysis times
        clean_times = [r['code_analysis']['analysis_time'] for r in clean_samples if r['code_analysis']['success']]
        buggy_times = [r['code_analysis']['analysis_time'] for r in buggy_samples if r['code_analysis']['success']]
        
        axes[0, 0].hist([clean_times, buggy_times], bins=10, alpha=0.7, label=['Clean', 'Buggy'])
        axes[0, 0].set_title('Code Analysis Time Distribution')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Bug detection accuracy
        clean_accuracies = [r['bug_detection']['accuracy'] for r in clean_samples if r['bug_detection']['success']]
        buggy_accuracies = [r['bug_detection']['accuracy'] for r in buggy_samples if r['bug_detection']['success']]
        
        axes[0, 1].hist([clean_accuracies, buggy_accuracies], bins=10, alpha=0.7, label=['Clean', 'Buggy'])
        axes[0, 1].set_title('Bug Detection Accuracy Distribution')
        axes[0, 1].set_xlabel('Accuracy')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Documentation quality
        clean_quality = [r['documentation']['quality_score'] for r in clean_samples if r['documentation']['success']]
        buggy_quality = [r['documentation']['quality_score'] for r in buggy_samples if r['documentation']['success']]
        
        axes[1, 0].hist([clean_quality, buggy_quality], bins=10, alpha=0.7, label=['Clean', 'Buggy'])
        axes[1, 0].set_title('Documentation Quality Distribution')
        axes[1, 0].set_xlabel('Quality Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Memory usage
        clean_memory = [r['code_analysis']['memory_usage'] for r in clean_samples if r['code_analysis']['success']]
        buggy_memory = [r['code_analysis']['memory_usage'] for r in buggy_samples if r['code_analysis']['success']]
        
        axes[1, 1].hist([clean_memory, buggy_memory], bins=10, alpha=0.7, label=['Clean', 'Buggy'])
        axes[1, 1].set_title('Memory Usage Distribution')
        axes[1, 1].set_xlabel('Memory (MB)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_benchmark.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Performance chart saved")
    
    def create_accuracy_chart(self):
        """Create accuracy comparison chart"""
        categories = {}
        
        for result in self.results:
            category = result['category']
            if category not in categories:
                categories[category] = {'clean': [], 'buggy': []}
            
            if result['bug_detection']['success']:
                categories[category][result['sample_type']].append(result['bug_detection']['accuracy'])
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        category_names = list(categories.keys())
        clean_accuracies = [sum(categories[cat]['clean']) / len(categories[cat]['clean']) if categories[cat]['clean'] else 0 for cat in category_names]
        buggy_accuracies = [sum(categories[cat]['buggy']) / len(categories[cat]['buggy']) if categories[cat]['buggy'] else 0 for cat in category_names]
        
        x = range(len(category_names))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], clean_accuracies, width, label='Clean Code', alpha=0.8)
        ax.bar([i + width/2 for i in x], buggy_accuracies, width, label='Buggy Code', alpha=0.8)
        
        ax.set_xlabel('Code Category')
        ax.set_ylabel('Average Accuracy')
        ax.set_title('Bug Detection Accuracy by Category')
        ax.set_xticks(x)
        ax.set_xticklabels(category_names, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_by_category.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Accuracy chart saved")
    
    def create_time_distribution_chart(self):
        """Create time distribution chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract times for each agent
        analysis_times = [r['code_analysis']['analysis_time'] for r in self.results if r['code_analysis']['success']]
        bug_times = [r['bug_detection']['detection_time'] for r in self.results if r['bug_detection']['success']]
        doc_times = [r['documentation']['generation_time'] for r in self.results if r['documentation']['success']]
        attention_times = [r['attention']['analysis_time'] for r in self.results if r['attention']['success']]
        
        # Create box plot
        data = [analysis_times, bug_times, doc_times, attention_times]
        labels = ['Code Analysis', 'Bug Detection', 'Documentation', 'Attention']
        
        ax.boxplot(data, labels=labels)
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Processing Time Distribution by Agent')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Time distribution chart saved")
    
    def create_success_rate_chart(self):
        """Create success rate chart"""
        agents = ['code_analysis', 'bug_detection', 'documentation', 'attention']
        success_rates = []
        
        for agent in agents:
            successful = sum(1 for r in self.results if r[agent]['success'])
            total = len(self.results)
            success_rates.append(successful / total * 100 if total > 0 else 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(agents, success_rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Agent Success Rates')
        ax.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'success_rates.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Success rate chart saved")
    
    def generate_summary_report(self):
        """Generate summary report"""
        summary_file = os.path.join(self.output_dir, "benchmark_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("NeuroCode Assistant - Benchmark Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Runtime: {self.end_time - self.start_time:.2f} seconds\n")
            f.write(f"Total Samples: {len(self.results)}\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            
            # Success rates
            agents = ['code_analysis', 'bug_detection', 'documentation', 'attention']
            for agent in agents:
                successful = sum(1 for r in self.results if r[agent]['success'])
                total = len(self.results)
                success_rate = successful / total * 100 if total > 0 else 0
                f.write(f"{agent.replace('_', ' ').title()}: {success_rate:.1f}% ({successful}/{total})\n")
            
            f.write("\n")
            
            # Performance metrics
            f.write("Performance Metrics:\n")
            f.write("-" * 20 + "\n")
            
            # Average times
            analysis_times = [r['code_analysis']['analysis_time'] for r in self.results if r['code_analysis']['success']]
            if analysis_times:
                f.write(f"Average Analysis Time: {sum(analysis_times)/len(analysis_times):.3f}s\n")
            
            bug_times = [r['bug_detection']['detection_time'] for r in self.results if r['bug_detection']['success']]
            if bug_times:
                f.write(f"Average Bug Detection Time: {sum(bug_times)/len(bug_times):.3f}s\n")
            
            doc_times = [r['documentation']['generation_time'] for r in self.results if r['documentation']['success']]
            if doc_times:
                f.write(f"Average Documentation Time: {sum(doc_times)/len(doc_times):.3f}s\n")
            
            attention_times = [r['attention']['analysis_time'] for r in self.results if r['attention']['success']]
            if attention_times:
                f.write(f"Average Attention Time: {sum(attention_times)/len(attention_times):.3f}s\n")
            
            f.write("\n")
            
            # Bug detection accuracy
            f.write("Bug Detection Accuracy:\n")
            f.write("-" * 25 + "\n")
            
            clean_accuracies = [r['bug_detection']['accuracy'] for r in self.results if r['sample_type'] == 'clean' and r['bug_detection']['success']]
            buggy_accuracies = [r['bug_detection']['accuracy'] for r in self.results if r['sample_type'] == 'buggy' and r['bug_detection']['success']]
            
            if clean_accuracies:
                f.write(f"Clean Code Accuracy: {sum(clean_accuracies)/len(clean_accuracies):.3f}\n")
            if buggy_accuracies:
                f.write(f"Buggy Code Accuracy: {sum(buggy_accuracies)/len(buggy_accuracies):.3f}\n")
            
            f.write("\n")
            
            # Top performing samples
            f.write("Top Performing Samples:\n")
            f.write("-" * 25 + "\n")
            
            # Sort by overall performance
            sorted_results = sorted(self.results, key=lambda r: (
                r['code_analysis']['analysis_time'] if r['code_analysis']['success'] else 999,
                r['bug_detection']['accuracy'] if r['bug_detection']['success'] else 0
            ))
            
            for i, result in enumerate(sorted_results[:5]):
                f.write(f"{i+1}. {result['sample_name']} ({result['sample_type']})\n")
                if result['code_analysis']['success']:
                    f.write(f"   Analysis Time: {result['code_analysis']['analysis_time']:.3f}s\n")
                if result['bug_detection']['success']:
                    f.write(f"   Bug Detection Accuracy: {result['bug_detection']['accuracy']:.3f}\n")
                f.write("\n")
        
        print(f"✅ Summary report saved: {summary_file}")

def main():
    """Main function to run benchmark"""
    print("🧪 NeuroCode Assistant - Comprehensive Benchmark Suite")
    print("=" * 60)
    
    # Create benchmark runner
    runner = BenchmarkRunner()
    
    # Run benchmark
    success = runner.run_full_benchmark()
    
    if success:
        print("\n🎉 Benchmark completed successfully!")
        print("📊 Check the benchmark_results directory for detailed reports")
        print("📈 MLflow tracking available at: http://127.0.0.1:5000")
    else:
        print("\n❌ Benchmark failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
