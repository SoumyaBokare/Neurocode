#!/usr/bin/env python3
"""
Robust Streamlit launcher for NeuroCode Assistant
"""
import os
import sys
import time

def setup_environment():
    """Setup environment variables"""
    os.environ['PYTHONWARNINGS'] = 'ignore'
    os.environ['STREAMLIT_EMAIL'] = ''
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} available")
        
        # Test basic imports
        import sys
        sys.path.append('.')
        
        from agents.code_analysis.agent import CodeAnalysisAgent
        print("✅ CodeAnalysisAgent available")
        
        from agents.bug_detection.agent import BugDetectionAgent
        print("✅ BugDetectionAgent available")
        
        from agents.documentation.agent import DocumentationAgent
        print("✅ DocumentationAgent available")
        
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Warning during dependency check: {e}")
        return True  # Continue anyway

def main():
    """Main launcher function"""
    print("🚀 Starting NeuroCode Assistant Streamlit UI")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages.")
        return False
    
    # Change to project root
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run Streamlit
    print("🎉 Launching Streamlit UI...")
    print("📍 URL: http://localhost:8501")
    print("⚠️  Note: Initial startup may take 30-60 seconds")
    print("🔐 Demo credentials: admin/admin123, developer/dev123, viewer/view123")
    
    # Import and run streamlit
    try:
        import streamlit.web.cli as stcli
        sys.argv = [
            "streamlit", "run", "ui/streamlit_app.py",
            "--server.headless", "true",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        stcli.main()
    except Exception as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return False

if __name__ == "__main__":
    main()
