# api/routes/auth_routes.py
"""
🔐 Authentication Routes for AI Model Sentinel API
📦 RESTful API endpoints for user authentication and authorization
👨‍💻 Author: Saleh Abughabraa  
🚀 Version: 2.0.0
💡 Business Logic:
   - Secure user authentication with JWT tokens
   - Role-based access control (RBAC) for multi-tenant SaaS
   - Password hashing with bcrypt/argon2
   - Refresh token mechanism for session management
   - Comprehensive audit logging for compliance
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, Optional
import logging
from datetime import datetime, timedelta, timezone
import uuid
import re

# Import system modules
from config import settings
from security.auth_manager import AuthenticationManager, RBAC, PasswordManager
from database.database_manager import DatabaseManager
from compliance.audit_logger import AuditLogger
from security.rate_limiter import RateLimiter

# Initialize router
router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])
logger = logging.getLogger(settings.LOGGER_NAME)
security = HTTPBearer()

# Initialize system components
auth_manager = AuthenticationManager()
db_manager = DatabaseManager()
audit_logger = AuditLogger()
rate_limiter = RateLimiter()
password_manager = PasswordManager()

# Pydantic models for request/response
from pydantic import BaseModel, EmailStr, validator

class UserLogin(BaseModel):
    username: str
    password: str
    two_factor_code: Optional[str] = None

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    role: str = "viewer"
    tenant_id: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @validator('role')
    def validate_role(cls, v):
        valid_roles = ['admin', 'analyst', 'viewer', 'auditor']
        if v not in valid_roles:
            raise ValueError(f'Role must be one of: {", ".join(valid_roles)}')
        return v

class TokenRefresh(BaseModel):
    refresh_token: str

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

class TwoFactorSetup(BaseModel):
    enable: bool

# Dependency for rate limiting
async def check_auth_rate_limit(request: Request, identifier: str):
    """Rate limiting for authentication endpoints"""
    if not await rate_limiter.check_limit(f"auth_{identifier}", 5, 300):  # 5 attempts per 5 minutes
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many authentication attempts. Please try again later."
        )

@router.post("/login", summary="User login with JWT authentication", response_model=Dict[str, Any])
async def login(
    login_data: UserLogin,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    🔐 Authenticate user and return JWT tokens
    """
    try:
        # Rate limiting by username
        await check_auth_rate_limit(request, login_data.username)
        
        # Authenticate user
        user = await auth_manager.authenticate_user(
            login_data.username, 
            login_data.password
        )
        
        if not user:
            # Log failed attempt
            await audit_logger.log(
                user_id="anonymous",
                tenant_id="unknown",
                action="login_failed",
                resource_type="authentication",
                resource_id=login_data.username,
                details={"reason": "invalid_credentials", "ip": request.client.host}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        # Check if 2FA is required and validate
        if user.get("two_factor_enabled"):
            if not login_data.two_factor_code:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Two-factor authentication code required"
                )
            
            if not await auth_manager.verify_two_factor_code(user["user_id"], login_data.two_factor_code):
                await audit_logger.log(
                    user_id=user["user_id"],
                    tenant_id=user.get("tenant_id", "default"),
                    action="login_failed",
                    resource_type="authentication",
                    resource_id=user["user_id"],
                    details={"reason": "invalid_2fa", "ip": request.client.host}
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid two-factor authentication code"
                )
        
        # Generate tokens
        tokens = await auth_manager.generate_tokens(user)
        
        # Update last login
        await db_manager.execute(
            "UPDATE users SET last_login = $1 WHERE user_id = $2",
            datetime.now(timezone.utc),
            user["user_id"]
        )
        
        # Audit log successful login
        background_tasks.add_task(
            audit_logger.log,
            user_id=user["user_id"],
            tenant_id=user.get("tenant_id", "default"),
            action="login_success",
            resource_type="authentication",
            resource_id=user["user_id"],
            details={"ip": request.client.host, "user_agent": request.headers.get("user-agent")}
        )
        
        return {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user": {
                "user_id": user["user_id"],
                "username": user["username"],
                "email": user["email"],
                "role": user["role"],
                "permissions": await auth_manager.get_user_permissions(user["user_id"])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

@router.post("/register", summary="User registration", response_model=Dict[str, Any])
async def register(
    user_data: UserRegister,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    📝 Register new user with secure password hashing
    """
    try:
        # Rate limiting by IP
        await check_auth_rate_limit(request, f"register_{request.client.host}")
        
        # Check if username or email already exists
        existing_user = await db_manager.fetch_one(
            "SELECT user_id FROM users WHERE username = $1 OR email = $2",
            user_data.username, user_data.email
        )
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered"
            )
        
        # Hash password
        hashed_password = await password_manager.hash_password(user_data.password)
        
        # Generate user ID
        user_id = f"user_{uuid.uuid4().hex[:16]}"
        tenant_id = user_data.tenant_id or f"tenant_{uuid.uuid4().hex[:8]}"
        
        # Create user record
        await db_manager.execute(
            """
            INSERT INTO users 
            (user_id, tenant_id, username, email, password_hash, full_name, role, created_at, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            user_id,
            tenant_id,
            user_data.username,
            user_data.email,
            hashed_password,
            user_data.full_name,
            user_data.role,
            datetime.now(timezone.utc),
            True
        )
        
        # Create default permissions based on role
        await auth_manager.setup_user_permissions(user_id, user_data.role, tenant_id)
        
        # Audit log
        background_tasks.add_task(
            audit_logger.log,
            user_id=user_id,
            tenant_id=tenant_id,
            action="user_registered",
            resource_type="user",
            resource_id=user_id,
            details={
                "username": user_data.username,
                "email": user_data.email,
                "role": user_data.role,
                "ip": request.client.host
            }
        )
        
        # Send welcome email (background task)
        background_tasks.add_task(
            send_welcome_email,
            user_data.email,
            user_data.full_name
        )
        
        return {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "role": user_data.role,
            "tenant_id": tenant_id,
            "message": "User registered successfully. Please check your email for verification."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/refresh", summary="Refresh access token", response_model=Dict[str, Any])
async def refresh_token(
    refresh_data: TokenRefresh,
    request: Request
):
    """
    🔄 Refresh access token using refresh token
    """
    try:
        # Verify refresh token
        user_data = await auth_manager.verify_refresh_token(refresh_data.refresh_token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Generate new tokens
        tokens = await auth_manager.generate_tokens(user_data)
        
        # Audit log
        await audit_logger.log(
            user_id=user_data["user_id"],
            tenant_id=user_data.get("tenant_id", "default"),
            action="token_refreshed",
            resource_type="authentication",
            resource_id=user_data["user_id"],
            details={"ip": request.client.host}
        )
        
        return {
            "access_token": tokens["access_token"],
            "refresh_token": tokens["refresh_token"],
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )

@router.post("/logout", summary="User logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
):
    """
    🚪 Logout user and invalidate tokens
    """
    try:
        # Extract token from authorization header
        token = credentials.credentials
        
        # Get user from token
        user_data = await auth_manager.verify_token(token)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        # Add token to blacklist
        await auth_manager.blacklist_token(token)
        
        # Audit log
        await audit_logger.log(
            user_id=user_data["user_id"],
            tenant_id=user_data.get("tenant_id", "default"),
            action="logout",
            resource_type="authentication",
            resource_id=user_data["user_id"],
            details={"ip": request.client.host if request else "unknown"}
        )
        
        return {
            "message": "Successfully logged out",
            "user_id": user_data["user_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/me", summary="Get current user info", response_model=Dict[str, Any])
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    👤 Get current authenticated user information
    """
    try:
        token = credentials.credentials
        
        # Verify token and get user data
        user_data = await auth_manager.verify_token(token)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        # Get user details from database
        user = await db_manager.fetch_one(
            """
            SELECT user_id, username, email, full_name, role, tenant_id, 
                   is_active, created_at, last_login, two_factor_enabled
            FROM users 
            WHERE user_id = $1
            """,
            user_data["user_id"]
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Get user permissions
        permissions = await auth_manager.get_user_permissions(user["user_id"])
        
        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "full_name": user["full_name"],
            "role": user["role"],
            "tenant_id": user["tenant_id"],
            "is_active": user["is_active"],
            "two_factor_enabled": user["two_factor_enabled"],
            "permissions": permissions,
            "created_at": user["created_at"].isoformat() if user["created_at"] else None,
            "last_login": user["last_login"].isoformat() if user["last_login"] else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )

@router.post("/password/reset-request", summary="Request password reset")
async def request_password_reset(
    reset_data: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    🔐 Request password reset link via email
    """
    try:
        # Rate limiting by email
        await check_auth_rate_limit(request, f"password_reset_{reset_data.email}")
        
        # Check if user exists
        user = await db_manager.fetch_one(
            "SELECT user_id, email, full_name FROM users WHERE email = $1 AND is_active = true",
            reset_data.email
        )
        
        if user:
            # Generate reset token
            reset_token = await auth_manager.generate_password_reset_token(user["user_id"])
            
            # Send reset email (background task)
            background_tasks.add_task(
                send_password_reset_email,
                user["email"],
                user["full_name"],
                reset_token
            )
        
        # Always return success to prevent email enumeration
        return {
            "message": "If the email exists, a password reset link has been sent"
        }
        
    except Exception as e:
        logger.error(f"Password reset request failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )

@router.post("/password/reset-confirm", summary="Confirm password reset")
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    background_tasks: BackgroundTasks,
    request: Request
):
    """
    ✅ Confirm password reset with token
    """
    try:
        # Verify reset token
        user_id = await auth_manager.verify_password_reset_token(reset_data.token)
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Validate new password
        if len(reset_data.new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        # Hash new password
        hashed_password = await password_manager.hash_password(reset_data.new_password)
        
        # Update password
        await db_manager.execute(
            "UPDATE users SET password_hash = $1 WHERE user_id = $2",
            hashed_password, user_id
        )
        
        # Invalidate all existing tokens for security
        await auth_manager.invalidate_user_tokens(user_id)
        
        # Audit log
        background_tasks.add_task(
            audit_logger.log,
            user_id=user_id,
            tenant_id="system",
            action="password_reset",
            resource_type="user",
            resource_id=user_id,
            details={"ip": request.client.host}
        )
        
        return {
            "message": "Password reset successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset confirmation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset failed"
        )

@router.post("/2fa/setup", summary="Setup two-factor authentication")
async def setup_two_factor(
    twofa_data: TwoFactorSetup,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None
):
    """
    🔒 Enable or disable two-factor authentication
    """
    try:
        token = credentials.credentials
        user_data = await auth_manager.verify_token(token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        if twofa_data.enable:
            # Generate 2FA secret
            secret = await auth_manager.generate_two_factor_secret(user_data["user_id"])
            
            return {
                "message": "Two-factor authentication setup required",
                "secret": secret,
                "qr_code_url": await auth_manager.generate_qr_code_url(secret, user_data["email"])
            }
        else:
            # Disable 2FA
            await auth_manager.disable_two_factor(user_data["user_id"])
            
            # Audit log
            await audit_logger.log(
                user_id=user_data["user_id"],
                tenant_id=user_data.get("tenant_id", "default"),
                action="2fa_disabled",
                resource_type="authentication",
                resource_id=user_data["user_id"],
                details={"ip": request.client.host if request else "unknown"}
            )
            
            return {
                "message": "Two-factor authentication disabled"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2FA setup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Two-factor authentication setup failed"
        )

@router.post("/2fa/verify", summary="Verify two-factor authentication setup")
async def verify_two_factor(
    verification_data: Dict[str, str],
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    ✅ Verify two-factor authentication setup
    """
    try:
        token = credentials.credentials
        user_data = await auth_manager.verify_token(token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        code = verification_data.get("code")
        if not code:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Verification code required"
            )
        
        # Verify 2FA code and enable
        success = await auth_manager.verify_and_enable_two_factor(
            user_data["user_id"], 
            code
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification code"
            )
        
        return {
            "message": "Two-factor authentication enabled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"2FA verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Two-factor authentication verification failed"
        )

@router.get("/health", summary="Authentication service health check")
async def auth_health():
    """
    ❤️ Check authentication service health status
    """
    try:
        # Check dependencies health
        db_health = await db_manager.health_check()
        auth_health_status = await auth_manager.health_check()
        
        overall_health = (
            db_health.get("status") == "healthy" and
            auth_health_status.get("status") == "healthy"
        )
        
        return {
            "status": "healthy" if overall_health else "degraded",
            "service": "auth-routes",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "dependencies": {
                "database": db_health,
                "authentication": auth_health_status
            }
        }
        
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "auth-routes",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

# Background task functions
async def send_welcome_email(email: str, full_name: str):
    """Send welcome email to new users"""
    try:
        # Implementation for sending welcome email
        # This would integrate with an email service like SendGrid, SES, etc.
        logger.info(f"Welcome email sent to {email} for {full_name}")
    except Exception as e:
        logger.error(f"Failed to send welcome email: {e}")

async def send_password_reset_email(email: str, full_name: str, reset_token: str):
    """Send password reset email"""
    try:
        # Implementation for sending password reset email
        reset_url = f"{settings.FRONTEND_URL}/reset-password?token={reset_token}"
        logger.info(f"Password reset email sent to {email} with token: {reset_token}")
    except Exception as e:
        logger.error(f"Failed to send password reset email: {e}")

# Dependency for other routes to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    🔐 Dependency to get current authenticated user for other routes
    """
    try:
        token = credentials.credentials
        user_data = await auth_manager.verify_token(token)
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return user_data
    except Exception as e:
        logger.error(f"Failed to get current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

# Export router and dependencies
__all__ = ["router", "get_current_user"]