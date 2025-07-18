"""
Benchmark Test Files for NeuroCode Assistant
Contains clean and buggy code samples for testing
"""

# Clean code samples
CLEAN_CODE_SAMPLES = [
    {
        "name": "fibonacci_clean",
        "code": """
def fibonacci(n):
    \"\"\"Calculate the nth Fibonacci number using recursion.\"\"\"
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    \"\"\"Calculate the nth Fibonacci number using iteration.\"\"\"
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
        """,
        "expected_bugs": 0,
        "complexity": "medium",
        "category": "algorithms"
    },
    {
        "name": "binary_search_clean",
        "code": """
def binary_search(arr, target):
    \"\"\"Perform binary search on a sorted array.\"\"\"
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_recursive(arr, target, left=0, right=None):
    \"\"\"Recursive implementation of binary search.\"\"\"
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
        """,
        "expected_bugs": 0,
        "complexity": "medium",
        "category": "algorithms"
    },
    {
        "name": "class_design_clean",
        "code": """
class BankAccount:
    \"\"\"A simple bank account class with proper encapsulation.\"\"\"
    
    def __init__(self, account_number, initial_balance=0):
        self._account_number = account_number
        self._balance = initial_balance
        self._transaction_history = []
    
    @property
    def balance(self):
        return self._balance
    
    @property
    def account_number(self):
        return self._account_number
    
    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._balance += amount
        self._transaction_history.append(f"Deposited ${amount}")
        return self._balance
    
    def withdraw(self, amount):
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        if amount > self._balance:
            raise ValueError("Insufficient funds")
        self._balance -= amount
        self._transaction_history.append(f"Withdrew ${amount}")
        return self._balance
    
    def get_transaction_history(self):
        return self._transaction_history.copy()
        """,
        "expected_bugs": 0,
        "complexity": "medium",
        "category": "oop"
    },
    {
        "name": "data_processing_clean",
        "code": """
import json
from typing import List, Dict, Optional

class DataProcessor:
    \"\"\"Clean data processing with proper error handling.\"\"\"
    
    def __init__(self):
        self.processed_count = 0
    
    def process_json_data(self, json_string: str) -> Optional[Dict]:
        \"\"\"Process JSON data with error handling.\"\"\"
        try:
            data = json.loads(json_string)
            if not isinstance(data, dict):
                raise ValueError("JSON must be an object")
            return self._validate_and_clean(data)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"Processing error: {e}")
            return None
    
    def _validate_and_clean(self, data: Dict) -> Dict:
        \"\"\"Validate and clean the data.\"\"\"
        cleaned = {}
        for key, value in data.items():
            if isinstance(key, str) and key.strip():
                cleaned[key.strip()] = value
        self.processed_count += 1
        return cleaned
    
    def batch_process(self, json_list: List[str]) -> List[Dict]:
        \"\"\"Process multiple JSON strings.\"\"\"
        results = []
        for json_string in json_list:
            result = self.process_json_data(json_string)
            if result is not None:
                results.append(result)
        return results
        """,
        "expected_bugs": 0,
        "complexity": "medium",
        "category": "data_processing"
    },
    {
        "name": "async_operations_clean",
        "code": """
import asyncio
import aiohttp
from typing import List, Dict

class AsyncWebClient:
    \"\"\"Asynchronous web client with proper resource management.\"\"\"
    
    def __init__(self, timeout=30):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url: str) -> Dict:
        \"\"\"Fetch a single URL with error handling.\"\"\"
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    return {"url": url, "status": "success", "content": content}
                else:
                    return {"url": url, "status": "error", "error": f"HTTP {response.status}"}
        except Exception as e:
            return {"url": url, "status": "error", "error": str(e)}
    
    async def fetch_multiple(self, urls: List[str]) -> List[Dict]:
        \"\"\"Fetch multiple URLs concurrently.\"\"\"
        tasks = [self.fetch_url(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]
        """,
        "expected_bugs": 0,
        "complexity": "high",
        "category": "async"
    }
]

# Buggy code samples
BUGGY_CODE_SAMPLES = [
    {
        "name": "sql_injection_bug",
        "code": """
import sqlite3

def get_user_by_id(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result

def login_user(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # SQL injection vulnerability
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result is not None
        """,
        "expected_bugs": 2,
        "bug_types": ["sql_injection", "security"],
        "complexity": "high",
        "category": "security"
    },
    {
        "name": "buffer_overflow_simulation",
        "code": """
def unsafe_buffer_operation(data):
    # Simulated buffer overflow vulnerability
    buffer = bytearray(100)
    if len(data) > 100:
        # No bounds checking - potential overflow
        for i in range(len(data)):
            buffer[i] = data[i]
    return buffer

def process_user_input(user_input):
    # No input validation
    processed_data = user_input * 1000  # Potential memory exhaustion
    return unsafe_buffer_operation(processed_data)

def dangerous_eval(code_string):
    # Code injection vulnerability
    return eval(code_string)
        """,
        "expected_bugs": 3,
        "bug_types": ["buffer_overflow", "code_injection", "memory_exhaustion"],
        "complexity": "high",
        "category": "security"
    },
    {
        "name": "race_condition_bug",
        "code": """
import threading
import time

class UnsafeCounter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        # Race condition - not thread-safe
        temp = self.count
        time.sleep(0.001)  # Simulate processing time
        self.count = temp + 1
    
    def get_count(self):
        return self.count

def create_race_condition():
    counter = UnsafeCounter()
    threads = []
    
    for i in range(10):
        thread = threading.Thread(target=counter.increment)
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    return counter.get_count()  # Will likely be less than 10
        """,
        "expected_bugs": 1,
        "bug_types": ["race_condition", "concurrency"],
        "complexity": "high",
        "category": "concurrency"
    },
    {
        "name": "memory_leak_bug",
        "code": """
class LeakyResource:
    def __init__(self):
        self.data = []
        self.file_handle = open('temp.txt', 'w')
    
    def add_data(self, item):
        # Memory leak - data keeps growing
        self.data.append(item)
        self.data.append(item * 2)  # Unnecessary duplication
    
    def process_data(self):
        # File handle never closed - resource leak
        for item in self.data:
            self.file_handle.write(str(item) + '\\n')
    
    # Missing __del__ or context manager

def create_memory_leak():
    resources = []
    for i in range(1000):
        resource = LeakyResource()
        resource.add_data(f"data_{i}")
        resources.append(resource)  # References never released
    return resources
        """,
        "expected_bugs": 2,
        "bug_types": ["memory_leak", "resource_leak"],
        "complexity": "medium",
        "category": "resource_management"
    },
    {
        "name": "logic_error_bug",
        "code": """
def calculate_average(numbers):
    if len(numbers) == 0:
        return 0  # Should return None or raise exception
    
    total = 0
    for num in numbers:
        total += num
    
    # Logic error - should divide by len(numbers)
    return total / (len(numbers) - 1)

def find_maximum(arr):
    if not arr:
        return None
    
    max_val = arr[0]
    # Logic error - should start from index 1
    for i in range(len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
    
    return max_val

def factorial(n):
    if n < 0:
        return 1  # Should handle negative numbers properly
    if n == 0:
        return 1
    
    result = 1
    # Logic error - should be range(1, n+1)
    for i in range(1, n):
        result *= i
    
    return result
        """,
        "expected_bugs": 3,
        "bug_types": ["logic_error", "edge_case"],
        "complexity": "medium",
        "category": "logic"
    },
    {
        "name": "exception_handling_bug",
        "code": """
import json
import requests

def unsafe_json_parsing(json_string):
    # No exception handling
    data = json.loads(json_string)
    return data['required_field']

def unsafe_web_request(url):
    # No exception handling for network errors
    response = requests.get(url)
    return response.json()

def unsafe_file_operations(filename):
    # No exception handling for file operations
    with open(filename, 'r') as f:
        content = f.read()
    
    # Potential division by zero
    lines = content.split('\\n')
    return len(content) / len(lines)

def unsafe_type_conversion(value):
    # No validation before type conversion
    return int(value) + float(value)
        """,
        "expected_bugs": 4,
        "bug_types": ["exception_handling", "type_error", "division_by_zero"],
        "complexity": "medium",
        "category": "error_handling"
    },
    {
        "name": "authentication_bypass_bug",
        "code": """
def check_admin_access(user_role):
    # Logic error - should be == not =
    if user_role = "admin":
        return True
    return False

def validate_user_token(token):
    # Weak token validation
    if len(token) > 5:
        return True
    return False

def authenticate_user(username, password):
    # Hardcoded credentials
    if username == "admin" and password == "password123":
        return True
    
    # Weak password validation
    if len(password) > 3:
        return True
    
    return False

def authorize_action(user_id, action):
    # Missing authorization check
    return True  # Always allows access
        """,
        "expected_bugs": 4,
        "bug_types": ["authentication_bypass", "authorization_flaw", "weak_validation"],
        "complexity": "high",
        "category": "security"
    },
    {
        "name": "performance_bug",
        "code": """
def inefficient_search(data, target):
    # O(n²) when O(n) is possible
    for i in range(len(data)):
        for j in range(len(data)):
            if data[j] == target:
                return j
    return -1

def memory_inefficient_operation(n):
    # Creates unnecessary large data structures
    result = []
    for i in range(n):
        temp_list = list(range(i * 1000))  # Unnecessary memory allocation
        result.append(sum(temp_list))
    return result

def inefficient_string_concatenation(strings):
    # Inefficient string concatenation
    result = ""
    for s in strings:
        result = result + s + " "  # Creates new string each time
    return result

def unnecessary_recursion(n):
    # Exponential time complexity
    if n <= 1:
        return n
    return unnecessary_recursion(n-1) + unnecessary_recursion(n-2) + unnecessary_recursion(n-3)
        """,
        "expected_bugs": 4,
        "bug_types": ["performance", "algorithmic_complexity", "memory_inefficiency"],
        "complexity": "medium",
        "category": "performance"
    },
    {
        "name": "input_validation_bug",
        "code": """
def process_age(age_string):
    # No input validation
    age = int(age_string)
    if age > 150:
        return "Invalid age"
    return f"Age: {age}"

def calculate_discount(price, discount_percent):
    # No validation of inputs
    discount = price * (discount_percent / 100)
    return price - discount

def create_user_account(username, email, password):
    # No input validation
    if username and email and password:
        return {"username": username, "email": email, "password": password}
    return None

def process_file_upload(file_path, max_size):
    # No validation of file path or size
    with open(file_path, 'rb') as f:
        content = f.read()
    return len(content) <= max_size
        """,
        "expected_bugs": 4,
        "bug_types": ["input_validation", "path_traversal", "type_error"],
        "complexity": "medium",
        "category": "validation"
    },
    {
        "name": "crypto_vulnerability_bug",
        "code": """
import hashlib
import random

def weak_hash_function(password):
    # MD5 is cryptographically broken
    return hashlib.md5(password.encode()).hexdigest()

def generate_weak_random():
    # Weak random number generation
    return random.random()

def insecure_token_generation(length):
    # Predictable token generation
    token = ""
    for _ in range(length):
        token += str(random.randint(0, 9))
    return token

def store_password_insecurely(password):
    # Storing password in plaintext
    return {"password": password, "hash": weak_hash_function(password)}
        """,
        "expected_bugs": 4,
        "bug_types": ["weak_cryptography", "insecure_random", "password_storage"],
        "complexity": "high",
        "category": "cryptography"
    }
]

# Test configuration
BENCHMARK_CONFIG = {
    "timeout_seconds": 30,
    "max_retries": 3,
    "log_level": "INFO",
    "output_formats": ["csv", "json", "plots"],
    "metrics": [
        "analysis_time",
        "bug_detection_accuracy",
        "documentation_quality",
        "attention_coherence",
        "memory_usage",
        "cpu_usage"
    ]
}

def get_all_test_samples():
    """Get all test samples (clean and buggy)"""
    return {
        "clean": CLEAN_CODE_SAMPLES,
        "buggy": BUGGY_CODE_SAMPLES
    }

def get_samples_by_category(category):
    """Get samples filtered by category"""
    clean_filtered = [s for s in CLEAN_CODE_SAMPLES if s["category"] == category]
    buggy_filtered = [s for s in BUGGY_CODE_SAMPLES if s["category"] == category]
    return {"clean": clean_filtered, "buggy": buggy_filtered}

def get_samples_by_complexity(complexity):
    """Get samples filtered by complexity"""
    clean_filtered = [s for s in CLEAN_CODE_SAMPLES if s["complexity"] == complexity]
    buggy_filtered = [s for s in BUGGY_CODE_SAMPLES if s["complexity"] == complexity]
    return {"clean": clean_filtered, "buggy": buggy_filtered}
