"""
AI Model Sentinel v2.0.0 - Advanced Security System
Production-Grade Authentication, Authorization, and Security
"""

import os
import time
import secrets
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import wraps

import jwt
from jose import JWTError
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
from pydantic import BaseModel

from app.database import DatabaseManager, get_db
from app import config

# Security configurations
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(64))
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(64))
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS512")
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(
    schemes=["argon2", "bcrypt"],
    default="argon2",
    argon2__time_cost=3,
    argon2__memory_cost=65536,
    argon2__parallelism=4,
    argon2__hash_len=32
)

# Rate limiting storage
rate_limit_store = {}

class TokenData(BaseModel):
    username: str
    user_id: int
    scopes: List[str] = []
    exp: datetime

class UserSecurity:
    """Advanced user security management"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate secure password hash"""
        return pwd_context.hash(password)
    
    @staticmethod
    def generate_mfa_secret() -> str:
        """Generate MFA secret"""
        return secrets.token_hex(16)
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key"""
        return f"sk_{secrets.token_urlsafe(32)}"
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()

class JWTHandler:
    """JWT token management"""
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def create_refresh_token(data: dict) -> str:
        """Create JWT refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Optional[TokenData]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            username: str = payload.get("sub")
            user_id: int = payload.get("user_id")
            scopes: List[str] = payload.get("scopes", [])
            exp: int = payload.get("exp")
            
            if username is None or user_id is None:
                return None
            
            return TokenData(
                username=username,
                user_id=user_id,
                scopes=scopes,
                exp=datetime.fromtimestamp(exp)
            )
        except JWTError:
            return None

class RateLimiter:
    """Advanced rate limiting system"""
    
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URL"))
    
    async def is_rate_limited(self, key: str, limit: int, window: int) -> bool:
        """Check if request is rate limited"""
        current = int(time.time())
        window_start = current - window
        
        try:
            # Remove old entries
            await self.redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            request_count = await self.redis_client.zcard(key)
            
            if request_count >= limit:
                return True
            
            # Add current request
            await self.redis_client.zadd(key, {str(current): current})
            await self.redis_client.expire(key, window)
            
            return False
            
        except Exception:
            # Fallback to in-memory rate limiting
            return self._fallback_rate_limiting(key, limit, window)
    
    def _fallback_rate_limiting(self, key: str, limit: int, window: int) -> bool:
        """In-memory rate limiting fallback"""
        current_time = time.time()
        window_start = current_time - window
        
        if key not in rate_limit_store:
            rate_limit_store[key] = []
        
        # Remove old entries
        rate_limit_store[key] = [t for t in rate_limit_store[key] if t > window_start]
        
        if len(rate_limit_store[key]) >= limit:
            return True
        
        rate_limit_store[key].append(current_time)
        return False

class SecurityMiddleware:
    """Advanced security middleware"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.http_bearer = HTTPBearer(auto_error=False)
    
    async def __call__(self, request: Request):
        """Process request security"""
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Rate limiting by IP
        rate_limit_key = f"rate_limit:{client_ip}"
        if await self.rate_limiter.is_rate_limited(rate_limit_key, 100, 300):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Check for suspicious user agents
        if self._is_suspicious_user_agent(user_agent):
            await self._log_security_event(
                request, "suspicious_user_agent", 
                {"user_agent": user_agent, "ip": client_ip}
            )
        
        return True
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Detect suspicious user agents"""
        suspicious_patterns = [
            "bot", "crawler", "spider", "scanner", 
            "sqlmap", "nikto", "metasploit"
        ]
        
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)
    
    async def _log_security_event(self, request: Request, event_type: str, details: dict):
        """Log security event to database"""
        try:
            async with get_db() as db:
                await db.execute(
                    "INSERT INTO security_events (event_type, ip_address, user_agent, details) VALUES ($1, $2, $3, $4)",
                    event_type, request.client.host, request.headers.get("user-agent"), details
                )
        except Exception:
            pass  # Don't fail the request if logging fails

# Security dependencies
security_middleware = SecurityMiddleware()

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_middleware.http_bearer),
    request: Request = None
):
    """Get current authenticated user"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token_data = JWTHandler.verify_token(credentials.credentials)
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user exists and is active
    async with get_db() as db:
        user = await db.fetchrow(
            "SELECT * FROM users WHERE id = $1 AND is_active = TRUE",
            token_data.user_id
        )
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    
    # Log successful authentication
    if request:
        await security_middleware._log_security_event(
            request, "authentication_success",
            {"user_id": user["id"], "username": user["username"]}
        )
    
    return dict(user)

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """Get current active user"""
    if not current_user.get("is_active"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def rate_limiter(request: Request):
    """Rate limiting dependency"""
    await security_middleware(request)
    return True

# Permission-based access control
def require_permission(permission: str):
    """Decorator for permission-based access control"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = kwargs.get("current_user")
            if not current_user or permission not in current_user.get("permissions", []):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Scope-based access control
def require_scope(scope: str):
    """Decorator for scope-based access control"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            credentials = kwargs.get("credentials")
            if credentials:
                token_data = JWTHandler.verify_token(credentials.credentials)
                if token_data and scope in token_data.scopes:
                    return await func(*args, **kwargs)
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient scopes"
            )
        return wrapper
    return decorator