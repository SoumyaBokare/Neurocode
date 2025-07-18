# Sample Code Snippets for NeuroCode Assistant Code Search

## 1. Data Processing Function
```python
def process_user_data(data):
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
    
    return cleaned_data
```

## 2. File Operations
```python
def read_config_file(file_path):
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
        return {}
```

## 3. API Request Handler
```python
def handle_api_request(request_data):
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
        return {'error': str(e)}
```

## 4. Database Connection
```python
def connect_to_database(db_config):
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
        return None
```

## 5. Authentication Function
```python
def authenticate_user(username, password):
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
    
    return users[username] == hashed_password
```

## 6. Machine Learning Model
```python
def train_simple_model(X, y):
    """Train a simple machine learning model"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Create and train model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = model.score(X_test, y_test)
    
    return {
        'model': model,
        'accuracy': accuracy,
        'test_size': len(X_test)
    }
```

## 7. Error Handling Example
```python
def safe_divide(a, b):
    """Safely divide two numbers with error handling"""
    try:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        
        result = a / b
        return {'success': True, 'result': result}
        
    except ValueError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        return {'success': False, 'error': f"Unexpected error: {str(e)}"}
```

## 8. Web Scraping Function
```python
def scrape_website(url):
    """Scrape content from a website"""
    import requests
    from bs4 import BeautifulSoup
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title and text
        title = soup.find('title').text if soup.find('title') else 'No title'
        paragraphs = soup.find_all('p')
        text_content = ' '.join([p.text for p in paragraphs])
        
        return {
            'title': title,
            'content': text_content[:500],  # First 500 chars
            'url': url
        }
    except Exception as e:
        return {'error': str(e)}
```

## 9. Class Definition
```python
class DataProcessor:
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
        return value
```

## 10. Async Function Example
```python
async def fetch_data_async(url):
    """Asynchronously fetch data from URL"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {'success': True, 'data': data}
                else:
                    return {'success': False, 'error': f'HTTP {response.status}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
```

## How to Use These Samples:

1. **Copy any of the code snippets above**
2. **Open your NeuroCode Assistant at http://localhost:8501**
3. **Login with demo credentials** (admin/admin123, developer/dev123, or viewer/view123)
4. **Go to the "🔍 Code Search" tab**
5. **Paste the code in the search box**
6. **Adjust similarity threshold** (try 0.5-0.8 for good results)
7. **Click "🔍 Search Similar Code"**

The system will find similar code patterns based on semantic similarity using CodeBERT embeddings!
