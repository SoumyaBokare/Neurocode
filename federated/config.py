"""
Configuration file for Federated Learning Setup
"""

class FederatedConfig:
    """Configuration class for federated learning"""
    
    def __init__(self):
        self.num_clients = 3
        self.num_rounds = 5
        self.min_fit_clients = 3
        self.min_evaluate_clients = 3
        self.samples_per_client = 100
        self.local_epochs = 1
        self.batch_size = 8
        self.learning_rate = 1e-5
        self.base_model = "microsoft/codebert-base"
        self.num_classes = 2
        self.max_length = 128
        self.tracking_uri = "http://127.0.0.1:5000"
        self.server_experiment = "FederatedLearning_Server"
        self.client_experiment_prefix = "FederatedLearning_Client_"

# Federated Learning Configuration
FEDERATED_CONFIG = {
    # Server Configuration
    "server": {
        "num_rounds": 5,
        "min_fit_clients": 3,
        "min_evaluate_clients": 3,
        "min_available_clients": 3,
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "strategy": "FedAvg"
    },
    
    # Client Configuration
    "client": {
        "num_clients": 3,
        "samples_per_client": 100,
        "local_epochs": 1,
        "batch_size": 8,
        "learning_rate": 1e-5,
        "train_test_split": 0.8
    },
    
    # Model Configuration
    "model": {
        "base_model": "microsoft/codebert-base",
        "num_classes": 2,
        "freeze_bert": True,
        "classifier_hidden_size": 256,
        "dropout_rate": 0.3,
        "max_length": 128
    },
    
    # MLflow Configuration
    "mlflow": {
        "tracking_uri": "http://127.0.0.1:5000",
        "server_experiment": "FederatedLearning_Server",
        "client_experiment_prefix": "FederatedLearning_Client_"
    },
    
    # Resource Configuration
    "resources": {
        "num_cpus": 1,
        "num_gpus": 0.0,  # Set to 0.33 if you have GPU available
        "use_cuda": True
    }
}

# Synthetic Dataset Configuration
DATASET_CONFIG = {
    "good_code_samples": [
        "def add(a, b):\n    return a + b",
        "def multiply(x, y):\n    return x * y",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
        "def validate_email(email):\n    import re\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return re.match(pattern, email) is not None",
        "def calculate_gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]"
    ],
    
    "bad_code_samples": [
        "def unsafe_eval(code):\n    return eval(code)",
        "def dangerous_exec(code):\n    exec(code)",
        "def sql_injection(query):\n    cursor.execute(query)",
        "def unsafe_pickle(data):\n    return pickle.loads(data)",
        "def weak_random():\n    return random.random()",
        "def path_traversal(filename):\n    return open('../../../etc/passwd', 'r').read()",
        "def command_injection(cmd):\n    import os\n    os.system(cmd)",
        "def buffer_overflow():\n    data = 'A' * 10000\n    return data",
        "def hardcoded_credentials():\n    username = 'admin'\n    password = 'password123'\n    return username, password",
        "def unsafe_deserialization(data):\n    import pickle\n    return pickle.loads(data)"
    ]
}

# MLflow Experiment Descriptions
EXPERIMENT_DESCRIPTIONS = {
    "FederatedLearning_Server": "Federated Learning Server - Aggregated metrics from all clients using FedAvg",
    "FederatedLearning_Client_0": "Federated Client 0 - Local training metrics and evaluation results",
    "FederatedLearning_Client_1": "Federated Client 1 - Local training metrics and evaluation results", 
    "FederatedLearning_Client_2": "Federated Client 2 - Local training metrics and evaluation results"
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    "min_accuracy": 0.6,
    "max_loss": 1.0,
    "max_training_time": 300,  # seconds
    "convergence_threshold": 0.01
}
