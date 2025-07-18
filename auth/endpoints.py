"""
Authentication endpoints for NeuroCode Assistant
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth.security import (
    authenticate_user, create_access_token, get_current_user,
    require_role, require_permission, log_user_activity,
    create_user_session, invalidate_session, get_user_permissions,
    USERS_DB, ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: Dict[str, Any]
    permissions: list

class UserProfile(BaseModel):
    username: str
    email: str
    role: str
    permissions: list
    is_active: bool
    created_at: str

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str

class UserUpdateRequest(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return access token
    """
    user = authenticate_user(request.username, request.password)
    
    if not user:
        log_user_activity(
            {"username": request.username, "role": "unknown"},
            "login_failed",
            "auth",
            "Invalid credentials"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    
    # Create session
    session_id = create_user_session(user)
    
    # Log successful login
    log_user_activity(user, "login_success", "auth", f"Session: {session_id}")
    
    # Remove sensitive data from response
    user_response = {
        "username": user["username"],
        "email": user["email"],
        "role": user["role"],
        "is_active": user["is_active"]
    }
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user=user_response,
        permissions=get_user_permissions(user)
    )

@router.post("/logout")
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Logout user and invalidate session
    """
    log_user_activity(current_user, "logout", "auth", "User logged out")
    
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user profile
    """
    log_user_activity(current_user, "profile_view", "auth", "User viewed profile")
    
    return UserProfile(
        username=current_user["username"],
        email=current_user["email"],
        role=current_user["role"],
        permissions=get_user_permissions(current_user),
        is_active=current_user["is_active"],
        created_at=current_user["created_at"].isoformat()
    )

@router.put("/me")
async def update_user_profile(
    update_request: UserUpdateRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update user profile (limited fields)
    """
    username = current_user["username"]
    
    # Only allow email updates for non-admin users
    if current_user["role"] != "admin":
        if update_request.role is not None or update_request.is_active is not None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to update role or status"
            )
    
    # Update user data
    if update_request.email is not None:
        USERS_DB[username]["email"] = update_request.email
    
    if update_request.role is not None and current_user["role"] == "admin":
        USERS_DB[username]["role"] = update_request.role
    
    if update_request.is_active is not None and current_user["role"] == "admin":
        USERS_DB[username]["is_active"] = update_request.is_active
    
    log_user_activity(current_user, "profile_update", "auth", f"Updated profile")
    
    return {"message": "Profile updated successfully"}

@router.post("/change-password")
async def change_password(
    password_request: ChangePasswordRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Change user password
    """
    from auth.security import verify_password, get_password_hash
    
    username = current_user["username"]
    
    # Verify current password
    if not verify_password(password_request.current_password, current_user["hashed_password"]):
        log_user_activity(current_user, "password_change_failed", "auth", "Invalid current password")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    # Update password
    new_password_hash = get_password_hash(password_request.new_password)
    USERS_DB[username]["hashed_password"] = new_password_hash
    
    log_user_activity(current_user, "password_change_success", "auth", "Password changed")
    
    return {"message": "Password changed successfully"}

@router.get("/users", dependencies=[Depends(require_role("admin"))])
async def get_all_users(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get all users (admin only)
    """
    log_user_activity(current_user, "users_list", "admin", "Admin viewed user list")
    
    users = []
    for username, user_data in USERS_DB.items():
        users.append({
            "username": username,
            "email": user_data["email"],
            "role": user_data["role"],
            "is_active": user_data["is_active"],
            "created_at": user_data["created_at"].isoformat(),
            "permissions": get_user_permissions(user_data)
        })
    
    return {"users": users}

@router.get("/roles")
async def get_available_roles(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get available roles and their permissions
    """
    from auth.security import ROLE_PERMISSIONS
    
    log_user_activity(current_user, "roles_view", "auth", "User viewed roles")
    
    return {"roles": ROLE_PERMISSIONS}

@router.get("/permissions")
async def get_user_permissions_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Get current user's permissions
    """
    permissions = get_user_permissions(current_user)
    
    return {
        "username": current_user["username"],
        "role": current_user["role"],
        "permissions": permissions
    }

@router.post("/validate-token")
async def validate_token(current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Validate if token is still valid
    """
    return {
        "valid": True,
        "user": {
            "username": current_user["username"],
            "role": current_user["role"],
            "permissions": get_user_permissions(current_user)
        }
    }

@router.get("/demo-users")
async def get_demo_users():
    """
    Get demo user credentials for testing
    """
    return {
        "demo_users": [
            {
                "username": "admin",
                "password": "admin123",
                "role": "admin",
                "description": "Full access to all features"
            },
            {
                "username": "developer",
                "password": "dev123",
                "role": "developer",
                "description": "Code analysis and bug detection"
            },
            {
                "username": "viewer",
                "password": "view123",
                "role": "viewer",
                "description": "Read-only access"
            }
        ]
    }

# Health check endpoint
@router.get("/health")
async def auth_health_check():
    """
    Check authentication service health
    """
    from auth.security import ROLE_PERMISSIONS
    return {
        "status": "healthy",
        "service": "authentication",
        "active_users": len(USERS_DB),
        "supported_roles": list(ROLE_PERMISSIONS.keys())
    }
