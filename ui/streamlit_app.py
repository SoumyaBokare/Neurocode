#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeuroCode Assistant - Professional Streamlit UI
A comprehensive web interface for the NeuroCode Assistant system
"""

import streamlit as st
import sys
import os
import json
import base64
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import agents and utilities
from agents.code_analysis.agent import CodeAnalysisAgent
from agents.bug_detection.agent import BugDetectionAgent
from agents.documentation.agent import DocumentationAgent
from agents.architecture_gnn.agent import ArchitectureAgent
from explainability.attention_map import analyze_code_attention
from vector_db.faiss_index import CodeVectorIndex
from orchestration.agent_router import route_code_task

# Configure Streamlit page
st.set_page_config(
    page_title="NeuroCode Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 0.5rem;
}

.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}

.metric-card {
    background: #f8f9fa;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border-left: 4px solid #1f77b4;
}

.success-box {
    background: #d4edda;
    border: 1px solid #c3e6cb;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-box {
    background: #fff3cd;
    border: 1px solid #ffeaa7;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}

.error-box {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 5px;
    padding: 1rem;
    margin: 1rem 0;
}

.code-block {
    background: #f8f9fa;
    border-radius: 5px;
    padding: 1rem;
    border-left: 4px solid #28a745;
    font-family: 'Courier New', monospace;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Theme toggle
def theme_toggle():
    """Toggle between light and dark theme"""
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    col1, col2, col3 = st.columns([1, 1, 8])
    with col1:
        if st.button("🌙 Dark"):
            st.session_state.theme = 'dark'
    with col2:
        if st.button("☀️ Light"):
            st.session_state.theme = 'light'

# Initialize agents
@st.cache_resource
def initialize_agents():
    """Initialize all AI agents"""
    try:
        agents = {
            'code_analysis': CodeAnalysisAgent(),
            'bug_detection': BugDetectionAgent(),
            'documentation': DocumentationAgent(),
            'architecture': ArchitectureAgent(),
            'vector_index': CodeVectorIndex()
        }
        return agents
    except Exception as e:
        st.error(f"Failed to initialize agents: {e}")
        return None

# Function to clear cache and reinitialize
def clear_cache_and_reinitialize():
    """Clear Streamlit cache and reinitialize agents"""
    st.cache_resource.clear()
    return initialize_agents()

# Authentication simulation
def authenticate_user():
    """Simulate user authentication"""
    if st.session_state.user_role is None:
        st.markdown("<div class='main-header'>🧠 NeuroCode Assistant</div>", unsafe_allow_html=True)
        st.markdown("<div class='sub-header'>AI-Powered Code Analysis Platform</div>", unsafe_allow_html=True)
        
        st.markdown("### 🔐 Login")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username", placeholder="Enter your username")
            role = st.selectbox("Role", ["admin", "developer", "viewer"])
            
            if st.button("Login", type="primary"):
                if username:
                    st.session_state.user_role = role
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("Please enter a username")
        
        # Demo credentials
        st.markdown("### Demo Credentials")
        st.info("""
        **Admin**: Full access to all features
        **Developer**: Code analysis and bug detection
        **Viewer**: Read-only access
        """)
        
        return False
    
    return True

# Role-based access control
def check_permission(required_role):
    """Check if user has required permissions"""
    role_hierarchy = {"admin": 3, "developer": 2, "viewer": 1}
    user_level = role_hierarchy.get(st.session_state.user_role, 0)
    required_level = role_hierarchy.get(required_role, 0)
    return user_level >= required_level

# Header with user info
def show_header():
    """Display header with user information"""
    col1, col2, col3 = st.columns([2, 6, 2])
    
    with col1:
        st.markdown("<div class='main-header'>🧠 NeuroCode</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='sub-header'>AI-Powered Code Analysis</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"**User**: {st.session_state.username}")
        st.markdown(f"**Role**: {st.session_state.user_role.title()}")
        if st.button("Logout"):
            st.session_state.user_role = None
            st.session_state.username = None
            st.rerun()

# Code Analysis Tab
def code_analysis_tab():
    """Code analysis interface"""
    st.header("📊 Code Analysis")
    
    if not check_permission("developer"):
        st.error("Access denied. Developer role required.")
        return
    
    # Input section
    col1, col2 = st.columns([3, 1])
    
    with col1:
        code_input = st.text_area(
            "Enter your code:",
            height=300,
            placeholder="def your_function():\n    # Your code here\n    pass"
        )
    
    with col2:
        st.markdown("### Options")
        include_attention = st.checkbox("Include attention analysis", value=True)
        analysis_type = st.selectbox("Analysis Type", ["Full", "Quick", "Embedding Only"])
        
        if st.button("🔍 Analyze Code", type="primary"):
            if code_input.strip():
                with st.spinner("Analyzing code..."):
                    analyze_code(code_input, include_attention, analysis_type)
            else:
                st.warning("Please enter some code to analyze.")
        
        # Cache clearing button
        if st.button("🔄 Clear Cache"):
            clear_cache_and_reinitialize()
            st.success("Cache cleared! Please try again.")

def analyze_code(code_input, include_attention, analysis_type):
    """Perform code analysis"""
    agents = initialize_agents()
    if not agents:
        return
    
    try:
        # Basic analysis
        start_time = time.time()
        
        if analysis_type == "Embedding Only":
            embedding = agents['code_analysis'].analyze(code_input)
            analysis_time = time.time() - start_time
            
            st.success("✅ Analysis Complete!")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Embedding Dimensions", embedding.shape[0])
            with col2:
                st.metric("Analysis Time", f"{analysis_time:.2f}s")
            
            # Show embedding visualization
            fig = px.line(embedding[:50], title="Code Embedding (First 50 dimensions)")
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            # Full analysis with attention
            if include_attention:
                result = agents['code_analysis'].analyze_with_attention(code_input)
                analysis_time = time.time() - start_time
                
                st.success("✅ Analysis Complete!")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Embedding Dims", result['embedding'].shape[0])
                with col2:
                    st.metric("Tokens", len(result['tokens']))
                with col3:
                    st.metric("Attention Shape", f"{result['attention'].shape[0]}x{result['attention'].shape[1]}")
                with col4:
                    st.metric("Analysis Time", f"{analysis_time:.2f}s")
                
                # Attention visualization
                if st.checkbox("Show Attention Heatmap"):
                    attention_fig = go.Figure(data=go.Heatmap(
                        z=result['attention'],
                        x=result['tokens'],
                        y=result['tokens'],
                        colorscale='Viridis'
                    ))
                    attention_fig.update_layout(
                        title="Attention Heatmap",
                        xaxis_title="Tokens",
                        yaxis_title="Tokens"
                    )
                    st.plotly_chart(attention_fig, use_container_width=True)
                
                # Token importance
                if st.checkbox("Show Token Importance"):
                    # Calculate token importance
                    token_importance = result['attention'].sum(axis=0)
                    importance_df = pd.DataFrame({
                        'Token': result['tokens'],
                        'Importance': token_importance
                    }).sort_values('Importance', ascending=False)
                    
                    fig = px.bar(importance_df.head(10), 
                               x='Importance', 
                               y='Token',
                               orientation='h',
                               title="Top 10 Most Important Tokens")
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                embedding = agents['code_analysis'].analyze(code_input)
                analysis_time = time.time() - start_time
                
                st.success("✅ Analysis Complete!")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Embedding Dimensions", embedding.shape[0])
                with col2:
                    st.metric("Analysis Time", f"{analysis_time:.2f}s")
        
        # Add to history
        st.session_state.analysis_history.append({
            'timestamp': time.time(),
            'code': code_input[:100] + "..." if len(code_input) > 100 else code_input,
            'type': analysis_type,
            'duration': analysis_time
        })
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# Bug Detection Tab
def bug_detection_tab():
    """Bug detection interface"""
    st.header("🐛 Bug Detection")
    
    if not check_permission("developer"):
        st.error("Access denied. Developer role required.")
        return
    
    code_input = st.text_area(
        "Enter code to check for bugs:",
        height=300,
        placeholder="def potentially_buggy_function():\n    # Code that might have issues\n    pass"
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        severity_filter = st.selectbox("Min Severity", ["Low", "Medium", "High", "Critical"])
    
    with col2:
        bug_types = st.multiselect("Bug Types", 
                                  ["Security", "Logic", "Performance", "Style"],
                                  default=["Security", "Logic"])
    
    if st.button("🔍 Detect Bugs", type="primary"):
        if code_input.strip():
            with st.spinner("Analyzing for bugs..."):
                detect_bugs(code_input, severity_filter, bug_types)
        else:
            st.warning("Please enter some code to analyze.")

def detect_bugs(code_input, severity_filter, bug_types):
    """Perform bug detection"""
    agents = initialize_agents()
    if not agents:
        return
    
    try:
        start_time = time.time()
        result = agents['bug_detection'].predict(code_input)
        analysis_time = time.time() - start_time
        
        st.success("✅ Bug Detection Complete!")
        
        # Show results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analysis Time", f"{analysis_time:.2f}s")
        with col2:
            st.metric("Bugs Found", len(result.get('bugs', [])))
        
        # Display results based on actual structure
        if isinstance(result, dict):
            if 'prediction' in result:
                prediction = result['prediction']
                confidence = result.get('confidence', 0)
                
                if prediction == 'bug':
                    st.error(f"🚨 Potential bug detected! (Confidence: {confidence:.2f})")
                else:
                    st.success(f"✅ No bugs detected (Confidence: {confidence:.2f})")
            
            if 'bugs' in result:
                for bug in result['bugs']:
                    st.warning(f"**{bug.get('type', 'Unknown')}**: {bug.get('description', 'No description')}")
        
        # Show recommendations
        st.markdown("### 💡 Recommendations")
        recommendations = [
            "Consider adding input validation",
            "Review error handling",
            "Check for potential security vulnerabilities",
            "Optimize for performance"
        ]
        
        for rec in recommendations:
            st.info(f"• {rec}")
        
    except Exception as e:
        st.error(f"Bug detection failed: {e}")

# Documentation Tab
def documentation_tab():
    """Documentation generation interface"""
    st.header("📖 Documentation Generator")
    
    if not check_permission("developer"):
        st.error("Access denied. Developer role required.")
        return
    
    code_input = st.text_area(
        "Enter code to generate documentation:",
        height=300,
        placeholder="def example_function(param1, param2):\n    # Function implementation\n    return result"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        doc_style = st.selectbox("Documentation Style", 
                                ["Google", "NumPy", "Sphinx", "Plain"])
    
    with col2:
        include_examples = st.checkbox("Include Examples", value=True)
    
    if st.button("📝 Generate Documentation", type="primary"):
        if code_input.strip():
            with st.spinner("Generating documentation..."):
                generate_documentation(code_input, doc_style, include_examples)
        else:
            st.warning("Please enter some code to document.")

def generate_documentation(code_input, doc_style, include_examples):
    """Generate documentation"""
    agents = initialize_agents()
    if not agents:
        return
    
    try:
        start_time = time.time()
        result = agents['documentation'].generate(code_input)
        analysis_time = time.time() - start_time
        
        st.success("✅ Documentation Generated!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Generation Time", f"{analysis_time:.2f}s")
        with col2:
            st.metric("Doc Length", len(result) if isinstance(result, str) else "N/A")
        
        # Display documentation
        st.markdown("### 📄 Generated Documentation")
        if isinstance(result, str):
            st.markdown(f"```python\n{result}\n```")
        else:
            st.write(result)
        
        # Download option
        if isinstance(result, str):
            st.download_button(
                label="📥 Download Documentation",
                data=result,
                file_name="generated_docs.md",
                mime="text/markdown"
            )
    
    except Exception as e:
        st.error(f"Documentation generation failed: {e}")

# Architecture Visualization Tab
def architecture_tab():
    """Architecture visualization interface"""
    st.header("🏗️ Architecture Visualization")
    
    if not check_permission("admin"):
        st.error("Access denied. Admin role required.")
        return
    
    st.markdown("### Upload Project Structure")
    
    # File structure input
    structure_input = st.text_area(
        "Enter project structure (JSON format):",
        height=200,
        placeholder='{"src": {"main.py": "file", "utils": {"helper.py": "file"}}}',
        value='{"src": {"main.py": "file", "utils": {"helper.py": "file", "constants.py": "file"}, "tests": {"test_main.py": "file"}}}'
    )
    
    if st.button("🔍 Analyze Architecture", type="primary"):
        if structure_input.strip():
            with st.spinner("Analyzing architecture..."):
                analyze_architecture(structure_input)
        else:
            st.warning("Please enter project structure.")

def analyze_architecture(structure_input):
    """Analyze project architecture"""
    agents = initialize_agents()
    if not agents:
        return
    
    try:
        # Parse JSON structure
        file_tree = json.loads(structure_input)
        
        start_time = time.time()
        result = agents['architecture'].analyze(file_tree)
        analysis_time = time.time() - start_time
        
        st.success("✅ Architecture Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Analysis Time", f"{analysis_time:.2f}s")
        with col2:
            st.metric("Components", len(file_tree))
        
        # Display results
        st.markdown("### 🔍 Analysis Results")
        if isinstance(result, dict):
            st.json(result)
        else:
            st.write(result)
        
        # Create a simple network visualization
        st.markdown("### 🕸️ Architecture Graph")
        create_architecture_graph(file_tree)
        
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check your input.")
    except Exception as e:
        st.error(f"Architecture analysis failed: {e}")

def create_architecture_graph(file_tree):
    """Create a simple architecture visualization"""
    import networkx as nx
    
    try:
        # Create graph
        G = nx.Graph()
        
        def add_nodes(tree, parent=None):
            for name, content in tree.items():
                node_id = f"{parent}/{name}" if parent else name
                G.add_node(node_id, type="file" if content == "file" else "directory")
                
                if parent:
                    G.add_edge(parent, node_id)
                
                if isinstance(content, dict):
                    add_nodes(content, node_id)
        
        add_nodes(file_tree)
        
        # Create visualization
        pos = nx.spring_layout(G)
        
        # Extract node positions
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        # Create edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create plot
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='gray'),
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=list(G.nodes()),
            textposition="middle center",
            marker=dict(size=20, color='lightblue'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="Project Architecture",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Failed to create architecture graph: {e}")

# Search Similar Code Tab
def search_tab():
    """Code similarity search interface"""
    st.header("🔍 Code Search")
    
    if not check_permission("viewer"):
        st.error("Access denied. Viewer role required.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        query_code = st.text_area(
            "Enter code to search for similar examples:",
            height=200,
            placeholder="def example_function():\n    # Your search query\n    pass"
        )
    
    with col2:
        st.markdown("### Search Options")
        num_results = st.slider("Number of results", 1, 10, 5)
        similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.1)
        
        if st.button("🔄 Clear Cache", help="Clear Streamlit cache if having issues"):
            clear_cache_and_reinitialize()
            st.success("Cache cleared! Please try search again.")
    
    if st.button("🔍 Search Similar Code", type="primary"):
        if query_code.strip():
            with st.spinner("Searching for similar code..."):
                search_similar_code(query_code, num_results, similarity_threshold)
        else:
            st.warning("Please enter code to search for.")

def search_similar_code(query_code, num_results, similarity_threshold):
    """Search for similar code"""
    agents = initialize_agents()
    if not agents:
        return
    
    try:
        start_time = time.time()
        
        # Generate embedding for query
        embedding = agents['code_analysis'].analyze(query_code)
        
        # Get vector database stats
        try:
            stats = agents['vector_index'].get_stats()
        except AttributeError:
            # Fallback for older cached version
            stats = {
                'total_vectors': getattr(agents['vector_index'], 'index', {}).get('ntotal', 0) if hasattr(agents['vector_index'], 'index') else 0,
                'dimension': 768,
                'snippets_count': len(getattr(agents['vector_index'], 'snippets', []))
            }
        
        # Check if database is empty
        if stats['total_vectors'] == 0:
            st.warning("⚠️ Vector database is empty. Populating with sample data...")
            
            # Try to populate the database
            try:
                import subprocess
                result = subprocess.run(['python', 'populate_vector_db.py'], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode == 0:
                    st.success("✅ Database populated successfully!")
                    # Clear cache and reinitialize
                    agents = clear_cache_and_reinitialize()
                    if agents:
                        try:
                            stats = agents['vector_index'].get_stats()
                        except AttributeError:
                            stats = {
                                'total_vectors': getattr(agents['vector_index'], 'index', {}).get('ntotal', 0) if hasattr(agents['vector_index'], 'index') else 0,
                                'dimension': 768,
                                'snippets_count': len(getattr(agents['vector_index'], 'snippets', []))
                            }
                else:
                    st.error(f"❌ Failed to populate database: {result.stderr}")
                    return
            except subprocess.TimeoutExpired:
                st.error("❌ Database population timed out. Please try again.")
                return
            except Exception as e:
                st.error(f"❌ Error populating database: {e}")
                return
        
        # Search for similar code
        matches = agents['vector_index'].search(embedding, k=num_results)
        
        search_time = time.time() - start_time
        
        st.success("✅ Search Complete!")
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Search Time", f"{search_time:.2f}s")
        with col2:
            st.metric("Results Found", len(matches))
        with col3:
            st.metric("Total Vectors", stats['total_vectors'])
        
        # Display results
        if matches:
            st.markdown("### 🎯 Similar Code Found")
            
            # Filter by similarity threshold
            filtered_matches = [m for m in matches if m.get('similarity', 0) >= similarity_threshold]
            
            if filtered_matches:
                for i, match in enumerate(filtered_matches):
                    similarity = match.get('similarity', 0)
                    metadata = match.get('metadata', {})
                    
                    # Color code based on similarity
                    if similarity >= 0.8:
                        similarity_color = "🟢"
                    elif similarity >= 0.6:
                        similarity_color = "🟡"
                    else:
                        similarity_color = "🔴"
                    
                    with st.expander(f"{similarity_color} Result {i+1} - Similarity: {similarity:.3f} - {metadata.get('category', 'unknown')}"):
                        st.code(match.get('code', 'No code available'), language='python')
                        
                        # Show metadata if available
                        if metadata:
                            st.markdown("**Metadata:**")
                            st.json(metadata)
            else:
                st.info(f"No results found above similarity threshold {similarity_threshold:.2f}. Try lowering the threshold.")
        else:
            st.info("No similar code found. The vector database might be empty or the search failed.")
    
    except Exception as e:
        st.error(f"Search failed: {e}")
        st.markdown("### 🔧 Troubleshooting:")
        st.markdown("1. Make sure the vector database is populated")
        st.markdown("2. Try running: `python populate_vector_db.py`")
        st.markdown("3. Check that all agents are initialized correctly")
        st.markdown("4. Restart the application if needed")

# Analytics Dashboard
def analytics_tab():
    """Analytics and history dashboard"""
    st.header("📊 Analytics Dashboard")
    
    if not check_permission("admin"):
        st.error("Access denied. Admin role required.")
        return
    
    # Analysis history
    if st.session_state.analysis_history:
        st.markdown("### 📈 Analysis History")
        
        history_df = pd.DataFrame(st.session_state.analysis_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], unit='s')
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Analyses", len(history_df))
        
        with col2:
            st.metric("Avg Duration", f"{history_df['duration'].mean():.2f}s")
        
        with col3:
            st.metric("Fastest", f"{history_df['duration'].min():.2f}s")
        
        with col4:
            st.metric("Slowest", f"{history_df['duration'].max():.2f}s")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Analysis over time
            fig = px.line(history_df, x='timestamp', y='duration', 
                         title="Analysis Performance Over Time")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Analysis types
            type_counts = history_df['type'].value_counts()
            fig = px.pie(values=type_counts.values, names=type_counts.index,
                        title="Analysis Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed history
        st.markdown("### 📋 Detailed History")
        st.dataframe(history_df, use_container_width=True)
        
        # Export option
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="📥 Export History",
            data=csv,
            file_name="analysis_history.csv",
            mime="text/csv"
        )
    
    else:
        st.info("No analysis history available yet. Start analyzing code to see statistics here.")

# Main application
def main():
    """Main application function"""
    # Check authentication
    if not authenticate_user():
        return
    
    # Show header
    show_header()
    
    # Theme toggle
    theme_toggle()
    
    # Initialize agents
    agents = initialize_agents()
    if not agents:
        st.error("Failed to initialize AI agents. Please check your configuration.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    # Role-based tab visibility
    available_tabs = []
    
    if check_permission("viewer"):
        available_tabs.extend(["🔍 Code Search", "📊 Analytics"])
    
    if check_permission("developer"):
        available_tabs.extend(["📊 Code Analysis", "🐛 Bug Detection", "📖 Documentation"])
    
    if check_permission("admin"):
        available_tabs.extend(["🏗️ Architecture", "📊 Analytics"])
    
    # Remove duplicates while preserving order
    available_tabs = list(dict.fromkeys(available_tabs))
    
    selected_tab = st.sidebar.radio("Select Feature", available_tabs)
    
    # Display selected tab
    if selected_tab == "📊 Code Analysis":
        code_analysis_tab()
    elif selected_tab == "🐛 Bug Detection":
        bug_detection_tab()
    elif selected_tab == "📖 Documentation":
        documentation_tab()
    elif selected_tab == "🏗️ Architecture":
        architecture_tab()
    elif selected_tab == "🔍 Code Search":
        search_tab()
    elif selected_tab == "📊 Analytics":
        analytics_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔗 Links")
    st.sidebar.markdown("[MLflow UI](http://127.0.0.1:5000)")
    st.sidebar.markdown("[API Docs](http://127.0.0.1:8001/docs)")
    st.sidebar.markdown("### 📋 System Info")
    st.sidebar.info(f"Role: {st.session_state.user_role.title()}")
    st.sidebar.info(f"Agents: {'✅ Active' if agents else '❌ Inactive'}")

if __name__ == "__main__":
    main()
