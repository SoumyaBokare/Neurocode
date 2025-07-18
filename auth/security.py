"""
Security and Authentication Module for NeuroCode Assistant
Implements JWT-based authentication and role-based access control
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

# Secret key for JWT (in production, use environment variable)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Mock user database (in production, use real database)
USERS_DB = {
    "admin": {
        "username": "admin",
        "email": "admin@neurocode.com",
        "hashed_password": bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode(),
        "role": "admin",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "permissions": ["all"]
    },
    "developer": {
        "username": "developer",
        "email": "dev@neurocode.com",
        "hashed_password": bcrypt.hashpw("dev123".encode(), bcrypt.gensalt()).decode(),
        "role": "developer",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "permissions": ["analyze", "bug_detection", "documentation"]
    },
    "viewer": {
        "username": "viewer",
        "email": "viewer@neurocode.com",
        "hashed_password": bcrypt.hashpw("view123".encode(), bcrypt.gensalt()).decode(),
        "role": "viewer",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "permissions": ["search", "view"]
    }
}

# Role hierarchy and permissions
ROLE_PERMISSIONS = {
    "admin": {
        "level": 3,
        "permissions": [
            "analyze", "bug_detection", "documentation", "search", "view", 
            "architecture", "federated_learning", "admin_panel", "user_management"
        ]
    },
    "developer": {
        "level": 2,
        "permissions": [
            "analyze", "bug_detection", "documentation", "search", "view"
        ]
    },
    "viewer": {
        "level": 1,
        "permissions": ["search", "view"]
    }
}

security = HTTPBearer()

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class AuthorizationError(Exception):
    """Custom authorization error"""
    pass

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())

def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user credentials"""
    user = USERS_DB.get(username)
    if not user:
        return None
    
    if not verify_password(password, user["hashed_password"]):
        return None
    
    if not user["is_active"]:
        return None
    
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode JWT access token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.JWTError:
        raise AuthenticationError("Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    try:
        token = credentials.credentials
        payload = decode_access_token(token)
        username = payload.get("sub")
        
        if username is None:
            raise AuthenticationError("Invalid token")
        
        user = USERS_DB.get(username)
        if user is None:
            raise AuthenticationError("User not found")
        
        if not user["is_active"]:
            raise AuthenticationError("User account is disabled")
        
        return user
    
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_role(required_role: str):
    """Dependency to require specific role"""
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role")
        
        # Check role hierarchy
        user_level = ROLE_PERMISSIONS.get(user_role, {}).get("level", 0)
        required_level = ROLE_PERMISSIONS.get(required_role, {}).get("level", 999)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_role}, Current: {user_role}"
            )
        
        return current_user
    
    return role_checker

def require_permission(permission: str):
    """Dependency to require specific permission"""
    def permission_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role")
        user_permissions = ROLE_PERMISSIONS.get(user_role, {}).get("permissions", [])
        
        if permission not in user_permissions and "all" not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied. Required: {permission}"
            )
        
        return current_user
    
    return permission_checker

def get_user_permissions(user: Dict[str, Any]) -> list:
    """Get user permissions based on role"""
    role = user.get("role")
    return ROLE_PERMISSIONS.get(role, {}).get("permissions", [])

def can_access_resource(user: Dict[str, Any], resource: str) -> bool:
    """Check if user can access a resource"""
    permissions = get_user_permissions(user)
    return resource in permissions or "all" in permissions

# Session management
ACTIVE_SESSIONS = {}

def create_user_session(user: Dict[str, Any]) -> str:
    """Create a user session"""
    session_id = jwt.encode(
        {"user": user["username"], "created": datetime.utcnow().isoformat()},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    ACTIVE_SESSIONS[session_id] = {
        "user": user,
        "created_at": datetime.utcnow(),
        "last_activity": datetime.utcnow()
    }
    
    return session_id

def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session information"""
    return ACTIVE_SESSIONS.get(session_id)

def invalidate_session(session_id: str) -> bool:
    """Invalidate a session"""
    if session_id in ACTIVE_SESSIONS:
        del ACTIVE_SESSIONS[session_id]
        return True
    return False

def cleanup_expired_sessions():
    """Remove expired sessions"""
    current_time = datetime.utcnow()
    expired_sessions = []
    
    for session_id, session_data in ACTIVE_SESSIONS.items():
        last_activity = session_data["last_activity"]
        if (current_time - last_activity).seconds > 3600:  # 1 hour timeout
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del ACTIVE_SESSIONS[session_id]

# Audit logging
def log_user_activity(user: Dict[str, Any], action: str, resource: str, details: Optional[str] = None):
    """Log user activity for audit purposes"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "username": user["username"],
        "role": user["role"],
        "action": action,
        "resource": resource,
        "details": details,
        "ip_address": "127.0.0.1"  # In production, get from request
    }
    
    # In production, save to database or log file
    print(f"AUDIT: {log_entry}")

# Rate limiting (basic implementation)
REQUEST_COUNTS = {}

def check_rate_limit(user: Dict[str, Any], limit: int = 100, window: int = 3600) -> bool:
    """Check if user has exceeded rate limit"""
    username = user["username"]
    current_time = datetime.utcnow()
    
    if username not in REQUEST_COUNTS:
        REQUEST_COUNTS[username] = []
    
    # Remove old requests outside the window
    REQUEST_COUNTS[username] = [
        req_time for req_time in REQUEST_COUNTS[username]
        if (current_time - req_time).seconds < window
    ]
    
    # Check if limit exceeded
    if len(REQUEST_COUNTS[username]) >= limit:
        return False
    
    # Add current request
    REQUEST_COUNTS[username].append(current_time)
    return True

def rate_limit_dependency(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Dependency to enforce rate limiting"""
    if not check_rate_limit(current_user):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    return current_user
