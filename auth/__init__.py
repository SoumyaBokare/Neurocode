"""
Authentication and Security Module for NeuroCode Assistant
"""

from .security import (
    get_current_user,
    require_role,
    require_permission,
    authenticate_user,
    create_access_token,
    log_user_activity,
    rate_limit_dependency
)

from .endpoints import router as auth_router

__all__ = [
    "get_current_user",
    "require_role", 
    "require_permission",
    "authenticate_user",
    "create_access_token",
    "log_user_activity",
    "rate_limit_dependency",
    "auth_router"
]
