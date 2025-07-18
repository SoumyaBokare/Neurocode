"""
Federated Learning Simulator using Flower Framework
Implements FedAvg with 3 clients for CodeBERT fine-tuning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import flwr as fl
from flwr.common import Metrics
from typing import Dict, List, Tuple, Optional
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import time
from collections import OrderedDict
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeBERTClassifier(nn.Module):
    """CodeBERT-based classifier for federated learning"""
    
    def __init__(self, num_classes=2, freeze_bert=True):
        super(CodeBERTClassifier, self).__init__()
        self.codebert = AutoModel.from_pretrained("microsoft/codebert-base")
        
        # Freeze BERT layers if specified
        if freeze_bert:
            for param in self.codebert.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        return self.classifier(pooled_output)

class FederatedDataset:
    """Creates federated datasets for simulation"""
    
    def __init__(self, num_clients=3, samples_per_client=100):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.num_clients = num_clients
        self.samples_per_client = samples_per_client
        
        # Generate synthetic code samples for demonstration
        self.code_samples = self._generate_synthetic_data()
        
    def _generate_synthetic_data(self):
        """Generate synthetic code samples with labels"""
        # Simple synthetic data for demonstration
        good_code_samples = [
            "def add(a, b):\\n    return a + b",
            "def multiply(x, y):\\n    return x * y",
            "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)",
            "def is_prime(n):\\n    if n < 2:\\n        return False\\n    for i in range(2, int(n**0.5) + 1):\\n        if n % i == 0:\\n            return False\\n    return True",
            "def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)",
        ]
        
        bad_code_samples = [
            "def unsafe_eval(code):\\n    return eval(code)",
            "def dangerous_exec(code):\\n    exec(code)",
            "def sql_injection(query):\\n    cursor.execute(query)",
            "def unsafe_pickle(data):\\n    return pickle.loads(data)",
            "def weak_random():\\n    return random.random()",
        ]
        
        # Create balanced dataset
        all_samples = []
        for _ in range(self.num_clients * self.samples_per_client // 2):
            all_samples.append((np.random.choice(good_code_samples), 0))  # 0 = good code
            all_samples.append((np.random.choice(bad_code_samples), 1))   # 1 = bad code
        
        return all_samples
    
    def get_client_data(self, client_id: int) -> Tuple[DataLoader, DataLoader]:
        """Get train and test data for a specific client"""
        start_idx = client_id * self.samples_per_client
        end_idx = (client_id + 1) * self.samples_per_client
        
        client_samples = self.code_samples[start_idx:end_idx]
        
        # Split into train/test (80/20)
        train_size = int(0.8 * len(client_samples))
        train_samples = client_samples[:train_size]
        test_samples = client_samples[train_size:]
        
        # Tokenize and create DataLoaders
        train_loader = self._create_dataloader(train_samples)
        test_loader = self._create_dataloader(test_samples)
        
        return train_loader, test_loader
    
    def _create_dataloader(self, samples: List[Tuple[str, int]]) -> DataLoader:
        """Create DataLoader from samples"""
        texts = [sample[0] for sample in samples]
        labels = [sample[1] for sample in samples]
        
        # Tokenize
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Create dataset
        dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels, dtype=torch.long)
        )
        
        return DataLoader(dataset, batch_size=8, shuffle=True)

class FlowerClient(fl.client.NumPyClient):
    """Flower client implementation"""
    
    def __init__(self, client_id: int, model: CodeBERTClassifier, train_loader: DataLoader, test_loader: DataLoader):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # MLflow setup
        self.experiment_name = f"FederatedLearning_Client_{client_id}"
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow for this client"""
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            
            # Create client-specific experiment
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                client.create_experiment(self.experiment_name)
                logger.info(f"Created experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
    
    def get_parameters(self, config):
        """Get model parameters"""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters"""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train model on client data"""
        start_time = time.time()
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Train model
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        total_samples = 0
        
        try:
            with mlflow.start_run(run_name=f"client_{self.client_id}_round_{config.get('round', 0)}"):
                for batch_idx, (input_ids, attention_mask, labels) in enumerate(self.train_loader):
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_samples += labels.size(0)
                
                avg_loss = total_loss / len(self.train_loader)
                training_time = time.time() - start_time
                
                # Log to MLflow
                mlflow.log_param("client_id", self.client_id)
                mlflow.log_param("round", config.get('round', 0))
                mlflow.log_param("local_epochs", 1)
                mlflow.log_param("batch_size", 8)
                mlflow.log_param("learning_rate", 1e-5)
                
                mlflow.log_metric("train_loss", avg_loss)
                mlflow.log_metric("training_time", training_time)
                mlflow.log_metric("samples_trained", total_samples)
                
                mlflow.set_tag("client_type", "federated")
                mlflow.set_tag("experiment_type", "federated_learning")
                
                logger.info(f"Client {self.client_id}: Round {config.get('round', 0)}, Loss: {avg_loss:.4f}")
        
        except Exception as e:
            logger.error(f"Training failed for client {self.client_id}: {e}")
            avg_loss = float('inf')
        
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"train_loss": avg_loss}
    
    def evaluate(self, parameters, config):
        """Evaluate model on client data"""
        # Set parameters
        self.set_parameters(parameters)
        
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for input_ids, attention_mask, labels in self.test_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(self.test_loader)
        
        return avg_loss, len(self.test_loader.dataset), {"accuracy": accuracy, "test_loss": avg_loss}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics using weighted average"""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["test_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    
    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "test_loss": sum(losses) / sum(examples)
    }

class FederatedServer:
    """Federated Learning Server with MLflow logging"""
    
    def __init__(self, num_rounds=5):
        self.num_rounds = num_rounds
        self.experiment_name = "FederatedLearning_Server"
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow for server"""
        try:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            
            # Create server experiment
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                client.create_experiment(self.experiment_name)
                logger.info(f"Created experiment: {self.experiment_name}")
            
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            logger.error(f"MLflow setup failed: {e}")
    
    def log_round_metrics(self, round_num: int, metrics: Dict):
        """Log server-side metrics for each round"""
        try:
            with mlflow.start_run(run_name=f"server_round_{round_num}"):
                mlflow.log_param("round", round_num)
                mlflow.log_param("aggregation_method", "FedAvg")
                
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
                
                mlflow.set_tag("server_type", "federated")
                mlflow.set_tag("experiment_type", "federated_learning")
                
                logger.info(f"Server: Round {round_num}, Metrics: {metrics}")
        except Exception as e:
            logger.error(f"Failed to log server metrics: {e}")

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client"""
    client_id = int(cid)
    
    # Create model
    model = CodeBERTClassifier(num_classes=2, freeze_bert=True)
    
    # Get client data
    dataset = FederatedDataset(num_clients=3, samples_per_client=100)
    train_loader, test_loader = dataset.get_client_data(client_id)
    
    return FlowerClient(client_id, model, train_loader, test_loader)

def run_federated_simulation():
    """Run federated learning simulation"""
    logger.info("🚀 Starting Federated Learning Simulation")
    
    # Create server
    server = FederatedServer(num_rounds=5)
    
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Use all available clients
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Custom strategy with MLflow logging
    class MLflowFedAvg(fl.server.strategy.FedAvg):
        def __init__(self, server_instance, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.server = server_instance
            self.round_num = 0
        
        def aggregate_evaluate(self, server_round, results, failures):
            """Aggregate evaluation results and log to MLflow"""
            self.round_num = server_round
            
            # Call parent method
            aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
            
            # Log to MLflow
            if aggregated_metrics is not None:
                metrics = aggregated_metrics[1]  # Get metrics dict
                self.server.log_round_metrics(server_round, metrics)
            
            return aggregated_metrics
    
    # Use custom strategy
    strategy = MLflowFedAvg(
        server,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    
    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    
    logger.info("✅ Federated Learning Simulation Completed")
    logger.info("📊 Check MLflow UI at: http://127.0.0.1:5000")

if __name__ == "__main__":
    run_federated_simulation()
