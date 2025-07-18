# NeuroCode Assistant - Troubleshooting Guide

## 🔧 **COMMON ISSUES & SOLUTIONS**

### 1. Port Conflict Errors

**Error**: `[Errno 10048] error while attempting to bind on address`

**Solutions**:
```bash
# Check what's using the port
netstat -ano | findstr :8001

# Kill specific process
taskkill /F /PID [process_id]

# Use improved deployment script
python deploy_improved.py
```

### 2. Pydantic Deprecation Warnings

**Error**: `'schema_extra' has been renamed to 'json_schema_extra'`

**Status**: ⚠️ **Warning only** - functionality not affected
**Cause**: MLflow using older Pydantic syntax
**Solution**: These warnings are harmless and can be ignored

### 3. TensorFlow oneDNN Warnings

**Error**: `oneDNN custom operations are on...`

**Solution**: Set environment variable:
```bash
# Windows
set TF_ENABLE_ONEDNN_OPTS=0

# Or in Python
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
```

### 4. Streamlit Context Warnings

**Error**: `missing ScriptRunContext!`

**Status**: ⚠️ **Warning only** - UI functions normally
**Cause**: Running Streamlit through subprocess
**Solution**: Use `streamlit run` directly or ignore warnings

### 5. TensorFlow Deprecated Functions

**Error**: `tf.reset_default_graph is deprecated`

**Status**: ⚠️ **Warning only** - functionality not affected
**Solution**: These warnings come from dependencies and are harmless

---

## 🚀 **RECOMMENDED DEPLOYMENT APPROACH**

### Option 1: Use Improved Script (Recommended)
```bash
python deploy_improved.py
```
**Features**:
- Automatic port cleanup
- Suppressed warnings
- Better error handling
- Service monitoring

### Option 2: Manual Deployment
```bash
# Terminal 1: MLflow
set PYTHONWARNINGS=ignore
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Terminal 2: FastAPI  
set TF_ENABLE_ONEDNN_OPTS=0
uvicorn api.main:app --host 127.0.0.1 --port 8001

# Terminal 3: Streamlit
set STREAMLIT_EMAIL=
streamlit run ui/streamlit_app.py
```

### Option 3: Windows Batch Script
```batch
deploy.bat
```

---

## 🔍 **DIAGNOSTIC COMMANDS**

### Check Service Status
```bash
# Check ports
netstat -ano | findstr :5000
netstat -ano | findstr :8001  
netstat -ano | findstr :8501

# Check processes
tasklist | findstr python
```

### Test Individual Components
```bash
# Test MLflow
curl http://127.0.0.1:5000

# Test FastAPI
curl http://127.0.0.1:8001/docs

# Test Streamlit
curl http://localhost:8501
```

---

## ✅ **VALIDATION CHECKLIST**

- [ ] **MLflow**: http://127.0.0.1:5000 accessible
- [ ] **FastAPI**: http://127.0.0.1:8001/docs shows API documentation
- [ ] **Streamlit**: http://localhost:8501 shows login page
- [ ] **Authentication**: Can login with demo credentials
- [ ] **Features**: All tabs accessible based on role
- [ ] **Warnings**: Only harmless warnings present

---

## 🎯 **QUICK FIXES**

### If Services Won't Start
1. **Restart PowerShell/Command Prompt**
2. **Run as Administrator** (if needed)
3. **Check antivirus** (may block local servers)
4. **Try different ports** (8002, 8502, 5001)

### If UI Looks Broken
1. **Clear browser cache**
2. **Try incognito/private mode**
3. **Check browser console** for errors
4. **Refresh page** after login

### If Performance is Slow
1. **Close unnecessary applications**
2. **Ensure adequate RAM** (4GB+ recommended)
3. **Check CPU usage**
4. **Restart services** if needed

---

## 📞 **SUPPORT INFORMATION**

**System Requirements**:
- Python 3.8+
- 4GB+ RAM
- 2GB+ disk space
- Modern web browser

**Working Ports**:
- MLflow: 5000
- FastAPI: 8001
- Streamlit: 8501

**Demo Credentials**:
- Admin: admin / admin123
- Developer: developer / dev123
- Viewer: viewer / view123

---

## 🏆 **STATUS SUMMARY**

**Current Status**: ✅ **OPERATIONAL WITH WARNINGS**

The NeuroCode Assistant is **fully functional** despite the warnings. The warnings are:
- **Harmless** and don't affect functionality
- **Common** in Python ML/AI applications
- **Expected** when running complex ML pipelines
- **Can be suppressed** using environment variables

**Bottom Line**: The system works perfectly! The warnings are just noise. 🎉
