#!/usr/bin/env python3
"""
Test script for Federated Learning Simulation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from federated.federated_simulator import run_federated_simulation
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main function to run federated learning test"""
    print("=" * 60)
    print("🚀 FEDERATED LEARNING SIMULATION TEST")
    print("=" * 60)
    
    print("\n📋 Simulation Details:")
    print("- Framework: Flower (https://flower.dev)")
    print("- Model: CodeBERT with Classification Head")
    print("- Clients: 3 federated clients")
    print("- Strategy: FedAvg (Federated Averaging)")
    print("- Rounds: 5 training rounds")
    print("- Task: Code quality classification")
    print("- MLflow: Experiment tracking enabled")
    
    print("\n🔄 Starting simulation...")
    
    try:
        run_federated_simulation()
        
        print("\n" + "=" * 60)
        print("✅ FEDERATED LEARNING SIMULATION COMPLETED")
        print("=" * 60)
        
        print("\n📊 Check MLflow UI for results:")
        print("🌐 URL: http://127.0.0.1:5000")
        print("\n📈 Expected experiments:")
        print("- FederatedLearning_Server (server metrics)")
        print("- FederatedLearning_Client_0 (client 0 metrics)")
        print("- FederatedLearning_Client_1 (client 1 metrics)")
        print("- FederatedLearning_Client_2 (client 2 metrics)")
        
        print("\n🎯 Key Metrics to Monitor:")
        print("- train_loss: Training loss per client per round")
        print("- test_loss: Validation loss per client per round")
        print("- accuracy: Model accuracy per client per round")
        print("- training_time: Time taken for local training")
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        logger.error(f"Simulation failed: {e}")

if __name__ == "__main__":
    main()
