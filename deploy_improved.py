#!/usr/bin/env python3
"""
NeuroCode Assistant - Improved Deployment Script
Fixed version that handles port conflicts and warnings
"""

import subprocess
import sys
import time
import os
import psutil
import signal
from pathlib import Path

class ImprovedDeploymentManager:
    def __init__(self):
        self.processes = []
        self.project_root = Path(__file__).parent
        self.ports = {'mlflow': 5000, 'fastapi': 8001, 'streamlit': 8501}
        
    def kill_port(self, port):
        """Kill any process using the specified port"""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.pid:
                    try:
                        process = psutil.Process(conn.pid)
                        print(f"🔥 Killing process {conn.pid} on port {port}")
                        process.terminate()
                        process.wait(timeout=3)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        try:
                            process.kill()
                        except psutil.NoSuchProcess:
                            pass
        except Exception as e:
            print(f"⚠️  Warning: Could not clean port {port}: {e}")
    
    def cleanup_ports(self):
        """Clean up all required ports"""
        print("🧹 Cleaning up ports...")
        for service, port in self.ports.items():
            self.kill_port(port)
        time.sleep(1)
    
    def start_mlflow(self):
        """Start MLflow tracking server"""
        print("🚀 Starting MLflow tracking server...")
        try:
            env = os.environ.copy()
            env['PYTHONWARNINGS'] = 'ignore'  # Suppress warnings
            
            process = subprocess.Popen([
                sys.executable, "-m", "mlflow", "ui",
                "--backend-store-uri", "sqlite:///mlflow.db",
                "--port", "5000",
                "--host", "127.0.0.1"
            ], cwd=self.project_root, env=env, 
               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
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
            env = os.environ.copy()
            env['PYTHONWARNINGS'] = 'ignore'
            env['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable TensorFlow warnings
            
            process = subprocess.Popen([
                sys.executable, "-m", "uvicorn", "api.main:app",
                "--host", "127.0.0.1",
                "--port", "8001",
                "--log-level", "warning"
            ], cwd=self.project_root, env=env,
               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
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
            env = os.environ.copy()
            env['PYTHONWARNINGS'] = 'ignore'
            env['STREAMLIT_EMAIL'] = ''
            env['STREAMLIT_SERVER_HEADLESS'] = 'true'
            env['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
            
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py",
                "--server.headless", "true",
                "--server.port", "8501",
                "--server.address", "localhost",
                "--browser.gatherUsageStats", "false"
            ], cwd=self.project_root, env=env,
               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            
            self.processes.append(('Streamlit', process))
            print("✅ Streamlit UI started on http://localhost:8501")
            return True
        except Exception as e:
            print(f"❌ Failed to start Streamlit: {e}")
            return False
    
    def check_services(self):
        """Check if all services are running"""
        print("\n🔍 Checking service status...")
        all_running = True
        for name, process in self.processes:
            if process.poll() is None:
                print(f"✅ {name}: Running (PID: {process.pid})")
            else:
                print(f"❌ {name}: Stopped")
                self.get_process_output(name)  # Show output from crashed process
                all_running = False
        return all_running
    
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
    
    def get_process_output(self, process_name):
        """Get output from a specific process"""
        for name, process in self.processes:
            if name == process_name and process.poll() is not None:
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    if stdout:
                        print(f"📋 {name} stdout:")
                        print(stdout.decode('utf-8', errors='ignore')[:500])
                    if stderr:
                        print(f"📋 {name} stderr:")
                        print(stderr.decode('utf-8', errors='ignore')[:500])
                except:
                    pass
    
    def deploy(self):
        """Deploy all services with improved error handling"""
        print("🎉 NeuroCode Assistant - Improved Deployment")
        print("=" * 50)
        
        # Clean up existing processes
        self.cleanup_ports()
        
        # Start services with delays
        services_status = []
        
        mlflow_ok = self.start_mlflow()
        services_status.append(mlflow_ok)
        time.sleep(3)  # Give MLflow time to start
        
        api_ok = self.start_api()
        services_status.append(api_ok)
        time.sleep(3)  # Give API time to start
        
        streamlit_ok = self.start_streamlit()
        services_status.append(streamlit_ok)
        time.sleep(5)  # Give Streamlit time to start
        
        # Final status check
        all_running = self.check_services()
        
        if all_running and all(services_status):
            print("\n🎉 All services started successfully!")
            print("\n📊 Access Points:")
            print("- Streamlit UI: http://localhost:8501")
            print("- API Documentation: http://127.0.0.1:8001/docs")
            print("- MLflow Dashboard: http://127.0.0.1:5000")
            print("\n🔐 Demo Login Credentials:")
            print("- Admin: admin / admin123")
            print("- Developer: developer / dev123")
            print("- Viewer: viewer / view123")
            print("\n💡 Tips:")
            print("- Warnings are normal and don't affect functionality")
            print("- If ports are busy, the script will clean them up")
            print("- Press Ctrl+C to stop all services")
            
            try:
                # Keep running until interrupted
                while True:
                    time.sleep(10)
                    # Check if any process died
                    if not self.check_services():
                        print("\n⚠️  Some services stopped unexpectedly")
                        break
            except KeyboardInterrupt:
                print("\n🛑 Shutdown signal received...")
                self.stop_all()
                print("✅ All services stopped successfully!")
        else:
            print("\n❌ Some services failed to start or stopped unexpectedly")
            print("📋 Troubleshooting:")
            print("1. Check if ports 5000, 8001, 8501 are available")
            print("2. Ensure all dependencies are installed")
            print("3. Try running services individually")
            self.stop_all()
            sys.exit(1)

if __name__ == "__main__":
    deployment = ImprovedDeploymentManager()
    
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
