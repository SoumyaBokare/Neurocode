@echo off
echo NeuroCode Assistant - Windows Deployment Script (Improved)
echo ========================================================

REM Set environment variables to suppress warnings
set PYTHONWARNINGS=ignore
set TF_ENABLE_ONEDNN_OPTS=0
set STREAMLIT_EMAIL=

echo Cleaning up existing processes...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 2 /nobreak > nul

echo Starting MLflow server...
start "MLflow" cmd /c "set PYTHONWARNINGS=ignore && mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --host 127.0.0.1"
timeout /t 5 /nobreak > nul

echo Starting FastAPI server...
start "FastAPI" cmd /c "set TF_ENABLE_ONEDNN_OPTS=0 && set PYTHONWARNINGS=ignore && uvicorn api.main:app --host 127.0.0.1 --port 8001 --log-level warning"
timeout /t 5 /nobreak > nul

echo Starting Streamlit UI...
start "Streamlit" cmd /c "set STREAMLIT_EMAIL= && set PYTHONWARNINGS=ignore && streamlit run ui/streamlit_app.py --server.headless true --server.port 8501"
timeout /t 8 /nobreak > nul

echo.
echo ✅ All services started!
echo.
echo ⚠️  NOTE: Some warnings are normal and don't affect functionality
echo.
echo 📊 Access Points:
echo - Streamlit UI: http://localhost:8501
echo - API Documentation: http://127.0.0.1:8001/docs
echo - MLflow Dashboard: http://127.0.0.1:5000
echo.
echo 🔐 Demo Login Credentials:
echo - Admin: admin / admin123
echo - Developer: developer / dev123
echo - Viewer: viewer / view123
echo.
echo 💡 Troubleshooting:
echo - If Streamlit doesn't load, wait 30 seconds and refresh
echo - Check Windows Defender/Firewall if services don't start
echo - Use Ctrl+C in individual windows to stop services
echo.
echo Press any key to exit (services will continue running)...
pause > nul
