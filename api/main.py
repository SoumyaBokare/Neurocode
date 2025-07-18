
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from explainability.attention_map import plot_attention, analyze_code_attention, get_attention
from agents.architecture_gnn.agent import ArchitectureAgent
from agents.code_analysis.agent import CodeAnalysisAgent
from agents.bug_detection.agent import BugDetectionAgent
from agents.documentation.agent import DocumentationAgent
from vector_db.faiss_index import CodeVectorIndex
from orchestration.agent_router import route_code_task
from orchestration.langchain_router import LangChainRouter
from auth.security import get_current_user, require_role, require_permission, log_user_activity, rate_limit_dependency
from auth.endpoints import router as auth_router

app = FastAPI(
    title="NeuroCode Assistant API",
    description="AI-Powered Code Analysis Platform with Security and RBAC",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth router
app.include_router(auth_router, prefix="/auth", tags=["Authentication"])

# Initialize agents
agent = CodeAnalysisAgent()
bug_agent = BugDetectionAgent()
doc_agent = DocumentationAgent()
vector_index = CodeVectorIndex()
langchain_router = LangChainRouter()
arch_agent = ArchitectureAgent()



class CodeInput(BaseModel):
    code: str

class SearchResult(BaseModel):
    matches: list

class BugPredictionInput(BaseModel):
    code: str

class RouteTaskInput(BaseModel):
    code: str
    task: str

class DocGenInput(BaseModel):
    code: str

class MultiAgentInput(BaseModel):
    code: str
    task: str

class ArchitectureInput(BaseModel):
    file_tree: dict

class ExplainInput(BaseModel):
    tokens: list
    attention: list


@app.get("/")
def root():
    return {"status": "NeuroCode Assistant is alive!"}

@app.post("/analyze", dependencies=[Depends(require_permission("analyze"))])
def analyze_code(input: CodeInput, current_user: dict = Depends(rate_limit_dependency)):
    log_user_activity(current_user, "code_analysis", "analyze", f"Code length: {len(input.code)}")
    embedding = agent.analyze(input.code)
    return {"vector": embedding.tolist()}

@app.post("/analyze-with-attention", dependencies=[Depends(require_permission("analyze"))])
def analyze_with_attention(input: CodeInput, current_user: dict = Depends(rate_limit_dependency)):
    """
    Analyze code and return both embedding and attention weights
    """
    log_user_activity(current_user, "code_analysis", "analyze_attention", f"Code length: {len(input.code)}")
    result = agent.analyze_with_attention(input.code)
    return {
        "vector": result["embedding"].tolist(),
        "tokens": result["tokens"],
        "attention": result["attention"].tolist(),
        "latency_ms": result["latency_ms"]
    }

@app.post("/add", dependencies=[Depends(require_permission("analyze"))])
def add_code(input: CodeInput, current_user: dict = Depends(rate_limit_dependency)):
    log_user_activity(current_user, "vector_db", "add", f"Added code: {input.code[:50]}...")
    embedding = agent.analyze(input.code)
    vector_index.add_vector(embedding, input.code)
    return {"status": "added"}

@app.post("/search", dependencies=[Depends(require_permission("search"))])
def search_code(input: CodeInput, current_user: dict = Depends(rate_limit_dependency)):
    log_user_activity(current_user, "vector_db", "search", f"Search query: {input.code[:50]}...")
    embedding = agent.analyze(input.code)
    matches = vector_index.search(embedding, k=3)
    return {"matches": matches}

@app.post("/predict-bugs", dependencies=[Depends(require_permission("bug_detection"))])
def predict_bugs(input: BugPredictionInput, current_user: dict = Depends(rate_limit_dependency)):
    log_user_activity(current_user, "bug_detection", "predict", f"Code length: {len(input.code)}")
    result = bug_agent.predict(input.code)
    return result

@app.post("/generate-docs", dependencies=[Depends(require_permission("documentation"))])
def generate_docs(input: DocGenInput, current_user: dict = Depends(rate_limit_dependency)):
    log_user_activity(current_user, "documentation", "generate", f"Code length: {len(input.code)}")
    result = doc_agent.generate(input.code)
    return result

@app.post("/analyze-architecture", dependencies=[Depends(require_role("admin"))])
def analyze_architecture(input: ArchitectureInput, current_user: dict = Depends(rate_limit_dependency)):
    log_user_activity(current_user, "architecture", "analyze", f"File tree keys: {list(input.file_tree.keys())}")
    result = arch_agent.analyze(input.file_tree)
    return result

@app.post("/analyze-attention", dependencies=[Depends(require_permission("analyze"))])
def analyze_attention(input: CodeInput, current_user: dict = Depends(rate_limit_dependency)):
    """
    Comprehensive attention analysis for code
    Returns tokens, attention weights, importance scores, and visualization
    """
    log_user_activity(current_user, "explainability", "analyze_attention", f"Code length: {len(input.code)}")
    result = analyze_code_attention(input.code)
    return result

@app.post("/get-attention", dependencies=[Depends(require_permission("analyze"))])
def get_code_attention(input: CodeInput, current_user: dict = Depends(rate_limit_dependency)):
    """
    Get raw attention weights for code
    """
    log_user_activity(current_user, "explainability", "get_attention", f"Code length: {len(input.code)}")
    tokens, attention = get_attention(input.code)@app.post("/route-task", dependencies=[Depends(require_permission("analyze"))])
def route_task(input: RouteTaskInput, current_user: dict = Depends(rate_limit_dependency)):
    log_user_activity(current_user, "orchestration", "route_task", f"Task: {input.task}")
    result = route_code_task(input.code, input.task)
    return result

@app.post("/multi-agent", dependencies=[Depends(require_permission("analyze"))])
def multi_agent(input: MultiAgentInput, current_user: dict = Depends(rate_limit_dependency)):
    log_user_activity(current_user, "orchestration", "multi_agent", f"Task: {input.task}")
    result = langchain_router.route(input.code, input.task)
    return result

@app.post("/explain", dependencies=[Depends(require_permission("analyze"))])
def explain(input: ExplainInput, current_user: dict = Depends(rate_limit_dependency)):
    log_user_activity(current_user, "explainability", "explain", f"Tokens: {len(input.tokens)}")
    tokens = input.tokens
    attention = np.array(input.attention)
    img_base64 = plot_attention(tokens, attention)
    return {"image_base64": img_base64}

# Public endpoints (no authentication required)
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "neurocode-assistant",
        "version": "2.0.0",
        "features": ["analysis", "bug_detection", "documentation", "architecture", "explainability"]
    }

@app.get("/info")
def get_info():
    return {
        "name": "NeuroCode Assistant",
        "version": "2.0.0",
        "description": "AI-Powered Code Analysis Platform",
        "endpoints": {
            "authentication": "/auth/*",
            "analysis": "/analyze",
            "bug_detection": "/predict-bugs",
            "documentation": "/generate-docs",
            "architecture": "/analyze-architecture",
            "explainability": "/analyze-attention",
            "search": "/search"
        },
        "roles": ["admin", "developer", "viewer"]
    }
