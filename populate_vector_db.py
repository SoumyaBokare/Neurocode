#!/usr/bin/env python3
"""
Populate the vector database with sample code snippets
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_db.faiss_index import CodeVectorIndex
from agents.code_analysis.agent import CodeAnalysisAgent

def get_sample_code_snippets():
    """Get sample code snippets for the vector database"""
    samples = [
        {
            'code': '''def calculate_total(items):
    """Calculate total price of items"""
    total = 0
    for item in items:
        total += item.get('price', 0)
    return total''',
            'metadata': {'type': 'function', 'category': 'calculation'}
        },
        {
            'code': '''def process_user_data(data):
    """Process user data and return cleaned results"""
    if not data:
        return []
    
    cleaned_data = []
    for item in data:
        if isinstance(item, dict) and 'name' in item:
            cleaned_item = {
                'name': item['name'].strip(),
                'age': item.get('age', 0),
                'email': item.get('email', '').lower()
            }
            cleaned_data.append(cleaned_item)
    
    return cleaned_data''',
            'metadata': {'type': 'function', 'category': 'data_processing'}
        },
        {
            'code': '''def read_config_file(file_path):
    """Read configuration from a JSON file"""
    import json
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON in config file: {file_path}")
        return {}''',
            'metadata': {'type': 'function', 'category': 'file_io'}
        },
        {
            'code': '''def authenticate_user(username, password):
    """Authenticate user credentials"""
    import hashlib
    
    # Mock user database
    users = {
        'admin': 'hashed_admin_password',
        'user': 'hashed_user_password'
    }
    
    if username not in users:
        return False
    
    # Hash the provided password
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    
    return users[username] == hashed_password''',
            'metadata': {'type': 'function', 'category': 'authentication'}
        },
        {
            'code': '''def handle_api_request(request_data):
    """Handle API request and return response"""
    import requests
    
    url = request_data.get('url')
    method = request_data.get('method', 'GET')
    headers = request_data.get('headers', {})
    
    try:
        if method == 'GET':
            response = requests.get(url, headers=headers)
        elif method == 'POST':
            response = requests.post(url, json=request_data.get('data'), headers=headers)
        
        return {
            'status_code': response.status_code,
            'data': response.json() if response.content else None
        }
    except Exception as e:
        return {'error': str(e)}''',
            'metadata': {'type': 'function', 'category': 'api'}
        },
        {
            'code': '''def safe_divide(a, b):
    """Safely divide two numbers with error handling"""
    try:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        
        result = a / b
        return {'success': True, 'result': result}
        
    except ValueError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        return {'success': False, 'error': f"Unexpected error: {str(e)}"}''',
            'metadata': {'type': 'function', 'category': 'math'}
        },
        {
            'code': '''class DataProcessor:
    """A class for processing various data types"""
    
    def __init__(self, config):
        self.config = config
        self.processed_count = 0
    
    def process_item(self, item):
        """Process a single data item"""
        if not item:
            return None
        
        processed = {
            'id': item.get('id'),
            'timestamp': item.get('timestamp'),
            'value': self.normalize_value(item.get('value'))
        }
        
        self.processed_count += 1
        return processed
    
    def normalize_value(self, value):
        """Normalize a value based on configuration"""
        if isinstance(value, (int, float)):
            return value * self.config.get('multiplier', 1)
        return value''',
            'metadata': {'type': 'class', 'category': 'data_processing'}
        },
        {
            'code': '''def connect_to_database(db_config):
    """Connect to database and return connection object"""
    import sqlite3
    
    try:
        conn = sqlite3.connect(db_config['database_path'])
        cursor = conn.cursor()
        
        # Test connection
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        if result:
            print("Database connection successful")
            return conn
        else:
            raise Exception("Connection test failed")
            
    except Exception as e:
        print(f"Database connection failed: {e}")
        return None''',
            'metadata': {'type': 'function', 'category': 'database'}
        },
        {
            'code': '''def validate_email(email):
    """Validate email address format"""
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not email:
        return False
    
    if re.match(pattern, email):
        return True
    else:
        return False''',
            'metadata': {'type': 'function', 'category': 'validation'}
        },
        {
            'code': '''def log_message(message, level='INFO'):
    """Log a message with timestamp"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"[{timestamp}] {level}: {message}"
    
    print(formatted_message)
    
    # Also write to file
    with open('app.log', 'a') as f:
        f.write(formatted_message + '\\n')''',
            'metadata': {'type': 'function', 'category': 'logging'}
        }
    ]
    
    return samples

def populate_vector_database():
    """Populate the vector database with sample code"""
    print("🔄 Populating vector database with sample code...")
    
    # Initialize components
    try:
        code_analyzer = CodeAnalysisAgent()
        vector_index = CodeVectorIndex()
        
        # Get sample code snippets
        samples = get_sample_code_snippets()
        
        # Process each sample
        for i, sample in enumerate(samples):
            try:
                print(f"Processing sample {i+1}/{len(samples)}: {sample['metadata']['category']}")
                
                # Generate embedding
                embedding = code_analyzer.analyze(sample['code'])
                
                # Add to vector database
                vector_index.add_vector(embedding, sample['code'], sample['metadata'])
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
        
        # Save the index
        vector_index.save_index()
        
        # Print statistics
        stats = vector_index.get_stats()
        print(f"\n✅ Successfully populated vector database!")
        print(f"📊 Statistics:")
        print(f"   - Total vectors: {stats['total_vectors']}")
        print(f"   - Dimension: {stats['dimension']}")
        print(f"   - Snippets count: {stats['snippets_count']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error populating vector database: {e}")
        return False

if __name__ == "__main__":
    populate_vector_database()
