#!/usr/bin/env python3
"""
NeuroCode Assistant - Deployment Script
Starts all required services for the NeuroCode Assistant platform
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

class DeploymentManager:
    def __init__(self):
        self.processes = []
        self.project_root = Path(__file__).parent
        
    def start_mlflow(self):
        """Start MLflow tracking server"""
        print("🚀 Starting MLflow tracking server...")
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "mlflow", "ui",
                "--backend-store-uri", "sqlite:///mlflow.db",
                "--port", "5000"
            ], cwd=self.project_root)
            self.processes.append(('MLflow', process))
            print("✅ MLflow server started on http://127.0.0.1:5000")
            return True
        except Exception as e:
            print(f"❌ Failed to start MLflow: {e}")
            return False
    
    def start_api(self):
        """Start FastAPI server"""
        print("🚀 Starting FastAPI server...")
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "api.main:app",
                "--host", "127.0.0.1",
                "--port", "8001"
            ], cwd=self.project_root)
            self.processes.append(('FastAPI', process))
            print("✅ FastAPI server started on http://127.0.0.1:8001")
            return True
        except Exception as e:
            print(f"❌ Failed to start FastAPI: {e}")
            return False
    
    def start_streamlit(self):
        """Start Streamlit UI"""
        print("🚀 Starting Streamlit UI...")
        try:
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py",
                "--server.headless", "true"
            ], cwd=self.project_root, env={**os.environ, 'STREAMLIT_EMAIL': ''})
            self.processes.append(('Streamlit', process))
            print("✅ Streamlit UI started on http://localhost:8501")
            return True
        except Exception as e:
            print(f"❌ Failed to start Streamlit: {e}")
            return False
    
    def check_services(self):
        """Check if all services are running"""
        print("\n🔍 Checking service status...")
        for name, process in self.processes:
            if process.poll() is None:
                print(f"✅ {name}: Running (PID: {process.pid})")
            else:
                print(f"❌ {name}: Stopped")
    
    def stop_all(self):
        """Stop all services"""
        print("\n🛑 Stopping all services...")
        for name, process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name}: Stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"🔥 {name}: Force killed")
            except Exception as e:
                print(f"❌ Failed to stop {name}: {e}")
    
    def deploy(self):
        """Deploy all services"""
        print("🎉 NeuroCode Assistant - Deployment Manager")
        print("=" * 50)
        
        # Start services
        mlflow_ok = self.start_mlflow()
        time.sleep(2)  # Give MLflow time to start
        
        api_ok = self.start_api()
        time.sleep(2)  # Give API time to start
        
        streamlit_ok = self.start_streamlit()
        time.sleep(3)  # Give Streamlit time to start
        
        # Check status
        self.check_services()
        
        if all([mlflow_ok, api_ok, streamlit_ok]):
            print("\n🎉 All services started successfully!")
            print("\n📊 Access Points:")
            print("- Streamlit UI: http://localhost:8501")
            print("- API Documentation: http://127.0.0.1:8001/docs")
            print("- MLflow Dashboard: http://127.0.0.1:5000")
            print("\n🔐 Demo Login Credentials:")
            print("- Admin: admin / admin123")
            print("- Developer: developer / dev123")
            print("- Viewer: viewer / view123")
            print("\n⚠️  Press Ctrl+C to stop all services")
            
            try:
                # Keep running until interrupted
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
                self.stop_all()
                print("✅ All services stopped successfully!")
        else:
            print("\n❌ Some services failed to start. Check the logs above.")
            self.stop_all()
            sys.exit(1)

if __name__ == "__main__":
    deployment = DeploymentManager()
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Shutdown signal received...")
        deployment.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        deployment.deploy()
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        deployment.stop_all()
        sys.exit(1)
