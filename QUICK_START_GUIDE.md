# NeuroCode Assistant - Quick Start Guide

## 🚀 Production Deployment

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start MLflow Server
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### 3. Start API Server
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8001
```

### 4. Launch Professional UI
```bash
streamlit run ui/streamlit_app.py
```

### 5. Run Benchmarks (Optional)
```bash
python benchmark/benchmark_runner.py
```

## 🔐 Demo Users

- **Admin**: username: `admin`, password: `admin123`
- **Developer**: username: `developer`, password: `dev123`
- **Viewer**: username: `viewer`, password: `view123`

## 📊 Access Points

- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://127.0.0.1:8001/docs
- **MLflow Dashboard**: http://127.0.0.1:5000

## 🎯 Features Available

### UI Tabs
1. **Analyze Code** - Code analysis and insights
2. **Detect Bugs** - Security vulnerability detection
3. **Generate Docs** - Automated documentation
4. **Search Similar** - Vector-based code search
5. **Visualize Attention** - Attention weight heatmaps
6. **Architecture Graph** - GNN-based architecture visualization
7. **Analytics Dashboard** - Performance metrics and insights

### Security Features
- JWT-based authentication
- Role-based access control
- Rate limiting and audit logging
- Secure API endpoints

### Benchmarking
- Comprehensive test suite
- Performance measurement
- Accuracy testing
- MLflow integration
- Automated reporting

## 🎉 Success!

Your NeuroCode Assistant is now fully operational with:
- Professional web interface
- Secure authentication system
- Comprehensive benchmarking
- Production-ready deployment
