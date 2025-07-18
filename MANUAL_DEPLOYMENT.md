# Manual Deployment Guide - NeuroCode Assistant

## 🚀 **RECOMMENDED: Use 3 Separate Terminals**

### Terminal 1: MLflow Server
```bash
# Set environment variables
set PYTHONWARNINGS=ignore

# Start MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --host 127.0.0.1
```

### Terminal 2: FastAPI Server
```bash
# Set environment variables
set PYTHONWARNINGS=ignore
set TF_ENABLE_ONEDNN_OPTS=0

# Start FastAPI
uvicorn api.main:app --host 127.0.0.1 --port 8001 --log-level warning
```

### Terminal 3: Streamlit UI
```bash
# Set environment variables
set PYTHONWARNINGS=ignore
set STREAMLIT_EMAIL=
set TF_ENABLE_ONEDNN_OPTS=0

# Start Streamlit
streamlit run ui/streamlit_app.py --server.headless true --server.port 8501
```

## 🔧 **ALTERNATIVE: Use Python Launcher**

```bash
# Terminal 1: MLflow
python -c "import subprocess; subprocess.run(['mlflow', 'ui', '--backend-store-uri', 'sqlite:///mlflow.db', '--port', '5000'])"

# Terminal 2: FastAPI
python -c "import subprocess; subprocess.run(['uvicorn', 'api.main:app', '--host', '127.0.0.1', '--port', '8001'])"

# Terminal 3: Streamlit
python launch_streamlit.py
```

## 📋 **VERIFICATION STEPS**

1. **Check MLflow**: Open http://127.0.0.1:5000
2. **Check FastAPI**: Open http://127.0.0.1:8001/docs
3. **Check Streamlit**: Open http://localhost:8501
4. **Login**: Use admin/admin123, developer/dev123, or viewer/view123

## 🐛 **TROUBLESHOOTING**

### If Streamlit Won't Start:
1. **Check Python path**: `python -c "import sys; print(sys.path)"`
2. **Check imports**: `python -c "import streamlit; print('OK')"`
3. **Check agents**: `python -c "from agents.code_analysis.agent import CodeAnalysisAgent; print('OK')"`
4. **Run test**: `python test_streamlit.py`

### If Services Conflict:
1. **Kill all Python processes**: `taskkill /F /IM python.exe /T`
2. **Check ports**: `netstat -ano | findstr :8501`
3. **Use different ports**: Change 8501 to 8502, etc.

### If Warnings Appear:
- **Pydantic warnings**: Normal, ignore them
- **TensorFlow warnings**: Normal, ignore them
- **Streamlit warnings**: Normal, ignore them

## ✅ **SUCCESS INDICATORS**

- MLflow UI loads at http://127.0.0.1:5000
- FastAPI docs load at http://127.0.0.1:8001/docs
- Streamlit shows login page at http://localhost:8501
- Can login with demo credentials
- All tabs visible based on user role

## 🎯 **FINAL NOTES**

The warnings you see are **normal** and **harmless**. They don't affect functionality. The system is designed to work with these warnings present.

**Expected behavior**: 
- Warnings appear in console ✅
- Services start and run ✅
- UI is fully functional ✅
- All features work as expected ✅
