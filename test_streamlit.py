#!/usr/bin/env python3
"""
Simple Streamlit test to check if the app can run without errors
"""
import streamlit as st
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_streamlit():
    """Test basic Streamlit functionality"""
    st.title("🧠 NeuroCode Assistant - Test")
    st.write("If you can see this, Streamlit is working!")
    
    # Test session state
    if 'test_counter' not in st.session_state:
        st.session_state.test_counter = 0
    
    if st.button("Test Button"):
        st.session_state.test_counter += 1
        st.success(f"Button clicked {st.session_state.test_counter} times!")
    
    # Test imports
    try:
        from agents.code_analysis.agent import CodeAnalysisAgent
        st.success("✅ CodeAnalysisAgent imported successfully")
    except Exception as e:
        st.error(f"❌ CodeAnalysisAgent import failed: {e}")
    
    try:
        from agents.bug_detection.agent import BugDetectionAgent
        st.success("✅ BugDetectionAgent imported successfully")
    except Exception as e:
        st.error(f"❌ BugDetectionAgent import failed: {e}")
    
    try:
        from agents.documentation.agent import DocumentationAgent
        st.success("✅ DocumentationAgent imported successfully")
    except Exception as e:
        st.error(f"❌ DocumentationAgent import failed: {e}")

if __name__ == "__main__":
    test_streamlit()
