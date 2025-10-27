"""
AI Model Sentinel v2.0.0 - API Endpoints
Production-Grade REST API with Comprehensive Security
"""

import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, status, File, UploadFile, Form, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

from app.security import (
    get_current_user, get_current_active_user, rate_limiter, 
    require_permission, require_scope, JWTHandler, UserSecurity
)
from app.database import DatabaseManager, get_db, cache_manager
from app.engines import (
    FileMetadata, get_engine, get_all_engines, initialize_all_engines,
    EngineType, ThreatLevel, fusion_engine
)
from app.monitoring.metrics import metrics_collector

# Create API router
router = APIRouter(prefix="", tags=["api"])

# Security
security = HTTPBearer()

# Pydantic Models
class ScanRequest(BaseModel):
    file_path: str = Field(..., description="Path to file for analysis")
    engines: List[str] = Field(
        default=["quantum", "ml", "behavioral"],
        description="Detection engines to use"
    )
    priority: str = Field(default="normal", description="Scan priority")
    
    @validator('engines')
    def validate_engines(cls, v):
        valid_engines = ["quantum", "ml", "behavioral", "signature", "all"]
        for engine in v:
            if engine not in valid_engines and engine != "all":
                raise ValueError(f"Invalid engine: {engine}")
        return v

class ScanResponse(BaseModel):
    scan_id: str
    threat_level: str
    threat_score: float
    confidence: float
    processing_time: float
    engines_used: List[str]
    file_path: str
    details: List[str]
    metadata: Dict[str, Any]

class UserProfile(BaseModel):
    id: int
    username: str
    email: str
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool
    created_at: datetime

class SystemStats(BaseModel):
    uptime_seconds: float
    total_scans: int
    total_users: int
    total_requests: int
    average_scan_time: float
    threat_distribution: Dict[str, int]
    system_health: str

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: Dict[str, str]
    metrics: Dict[str, Any]

# Authentication endpoints
@router.post("/auth/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    full_name: Optional[str] = Form(None)
):
    """Register new user"""
    try:
        # Check if user already exists
        async with DatabaseManager.get_connection() as conn:
            existing_user = await conn.fetchrow(
                "SELECT id FROM users WHERE username = $1 OR email = $2",
                username, email
            )
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username or email already registered"
                )
            
            # Create user
            hashed_password = UserSecurity.get_password_hash(password)
            
            user = await conn.fetchrow("""
                INSERT INTO users (username, email, hashed_password, full_name)
                VALUES ($1, $2, $3, $4)
                RETURNING id, username, email, full_name, is_active, is_superuser, created_at
            """, username, email, hashed_password, full_name)
            
            # Log security event
            await conn.execute("""
                INSERT INTO security_events (event_type, user_id, details)
                VALUES ($1, $2, $3)
            """, "user_registered", user["id"], {"username": username})
            
            metrics_collector.record_security_event("user_registered", "info")
            
            return dict(user)
            
    except HTTPException:
        raise
    except Exception as e:
        metrics_collector.record_error("user_registration_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/auth/token")
async def login(
    username: str = Form(...),
    password: str = Form(...)
):
    """Login and get access token"""
    try:
        metrics_collector.record_auth_attempt()
        
        async with DatabaseManager.get_connection() as conn:
            # Get user
            user = await conn.fetchrow(
                "SELECT * FROM users WHERE username = $1 AND is_active = TRUE",
                username
            )
            
            if not user or not UserSecurity.verify_password(password, user["hashed_password"]):
                metrics_collector.record_auth_attempt("failed")
                
                # Log failed attempt
                await conn.execute("""
                    INSERT INTO security_events (event_type, details)
                    VALUES ($1, $2)
                """, "login_failed", {"username": username, "reason": "invalid_credentials"})
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password"
                )
            
            # Check if account is locked
            if user["locked_until"] and user["locked_until"] > datetime.now():
                raise HTTPException(
                    status_code=status.HTTP_423_LOCKED,
                    detail="Account temporarily locked due to failed login attempts"
                )
            
            # Reset failed attempts on successful login
            await conn.execute(
                "UPDATE users SET failed_login_attempts = 0, last_login = $1 WHERE id = $2",
                datetime.now(), user["id"]
            )
            
            # Create token
            access_token = JWTHandler.create_access_token({
                "sub": user["username"],
                "user_id": user["id"],
                "scopes": ["scan", "read_profile"]
            })
            
            # Log successful login
            await conn.execute("""
                INSERT INTO security_events (event_type, user_id, details)
                VALUES ($1, $2, $3)
            """, "login_success", user["id"], {"username": username})
            
            metrics_collector.record_auth_attempt("success")
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user": {
                    "id": user["id"],
                    "username": user["username"],
                    "email": user["email"]
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        metrics_collector.record_error("login_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

# Scan endpoints
@router.post("/scan", response_model=ScanResponse)
@require_scope("scan")
async def scan_file(
    request: ScanRequest,
    current_user: dict = Depends(get_current_active_user),
    rate_limit: bool = Depends(rate_limiter)
):
    """Scan file for threats"""
    start_time = time.time()
    
    try:
        metrics_collector.record_scan_start()
        
        # Validate file exists
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File not found"
            )
        
        # Check file size limit
        file_size = os.path.getsize(request.file_path)
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File too large (max 100MB)"
            )
        
        # Calculate file hashes
        file_hash_sha256 = await _calculate_file_hash(request.file_path, "sha256")
        file_hash_md5 = await _calculate_file_hash(request.file_path, "md5")
        
        # Check cache first
        cached_result = await cache_manager.get(f"scan_result:{file_hash_sha256}")
        if cached_result:
            import json
            result_data = json.loads(cached_result)
            result_data["cached"] = True
            return ScanResponse(**result_data)
        
        # Create file metadata
        file_metadata = FileMetadata(
            file_path=request.file_path,
            file_size=file_size,
            file_hash_sha256=file_hash_sha256,
            file_hash_md5=file_hash_md5,
            file_type=os.path.splitext(request.file_path)[1][1:].lower(),
            mime_type="application/octet-stream",  # Would use python-magic in production
            entropy=await _calculate_file_entropy(request.file_path),
            created_time=datetime.fromtimestamp(os.path.getctime(request.file_path)),
            modified_time=datetime.fromtimestamp(os.path.getmtime(request.file_path))
        )
        
        # Use fusion engine for analysis
        fusion_result = await fusion_engine.analyze(file_metadata)
        
        processing_time = time.time() - start_time
        
        # Prepare response
        scan_response = ScanResponse(
            scan_id=fusion_result.metadata.get("engine_results", {}).get("fusion", {}).get("scan_id", "unknown"),
            threat_level=fusion_result.threat_level.value,
            threat_score=fusion_result.threat_score,
            confidence=fusion_result.confidence,
            processing_time=processing_time,
            engines_used=list(fusion_result.metadata.get("engine_results", {}).keys()),
            file_path=request.file_path,
            details=fusion_result.details,
            metadata=fusion_result.metadata
        )
        
        # Cache result
        await cache_manager.set(
            f"scan_result:{file_hash_sha256}",
            scan_response.json(),
            expire=300  # 5 minutes
        )
        
        # Log scan to database
        async with DatabaseManager.get_connection() as conn:
            await conn.execute("""
                INSERT INTO scan_history 
                (scan_id, user_id, file_path, file_size, file_hash_sha256, file_hash_md5,
                 threat_level, threat_score, confidence, processing_time, engines_used, details)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, (
                scan_response.scan_id,
                current_user["id"],
                request.file_path,
                file_size,
                file_hash_sha256,
                file_hash_md5,
                scan_response.threat_level,
                scan_response.threat_score,
                scan_response.confidence,
                processing_time,
                scan_response.engines_used,
                scan_response.details
            ))
        
        # Record metrics
        metrics_collector.record_scan_completion(
            scan_response.threat_level,
            "fusion",
            processing_time
        )
        
        return scan_response
        
    except HTTPException:
        metrics_collector.record_scan_completion("error", "fusion", time.time() - start_time)
        raise
    except Exception as e:
        metrics_collector.record_scan_completion("error", "fusion", time.time() - start_time)
        metrics_collector.record_error("scan_failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scan failed: {str(e)}"
        )

@router.get("/scans/history")
@require_scope("scan")
async def get_scan_history(
    current_user: dict = Depends(get_current_active_user),
    limit: int = Query(10, le=100),
    offset: int = Query(0, ge=0)
):
    """Get user's scan history"""
    try:
        async with DatabaseManager.get_connection() as conn:
            scans = await conn.fetch("""
                SELECT scan_id, file_path, file_size, threat_level, threat_score,
                       confidence, processing_time, created_at
                FROM scan_history 
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
            """, current_user["id"], limit, offset)
            
            return [dict(scan) for scan in scans]
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch scan history: {str(e)}"
        )

# System endpoints
@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check"""
    components = {}
    
    # Check database
    try:
        async with DatabaseManager.get_connection() as conn:
            await conn.execute("SELECT 1")
            components["database"] = "healthy"
    except Exception as e:
        components["database"] = f"unhealthy: {str(e)}"
    
    # Check Redis
    try:
        redis_client = DatabaseManager.get_redis()
        await redis_client.ping()
        components["redis"] = "healthy"
    except Exception as e:
        components["redis"] = f"unhealthy: {str(e)}"
    
    # Check engines
    try:
        engines = get_all_engines()
        for engine_name, engine in engines.items():
            health = await engine.health_check()
            components[f"engine_{engine_name.value}"] = health["status"]
    except Exception as e:
        components["engines"] = f"unhealthy: {str(e)}"
    
    # Determine overall status
    if all("healthy" in status for status in components.values()):
        overall_status = "healthy"
    elif any("unhealthy" in status for status in components.values()):
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        version="2.0.0",
        timestamp=datetime.now().isoformat(),
        components=components,
        metrics=metrics_collector.get_performance_summary()
    )

@router.get("/stats", response_model=SystemStats)
@require_permission("view_stats")
async def get_system_stats(current_user: dict = Depends(get_current_active_user)):
    """Get system statistics"""
    try:
        async with DatabaseManager.get_connection() as conn:
            # Total scans
            total_scans = await conn.fetchval("SELECT COUNT(*) FROM scan_history")
            
            # Total users
            total_users = await conn.fetchval("SELECT COUNT(*) FROM users")
            
            # Threat distribution
            threat_distribution = await conn.fetch("""
                SELECT threat_level, COUNT(*) as count
                FROM scan_history 
                GROUP BY threat_level
            """)
            
            # Average scan time
            avg_scan_time = await conn.fetchval("""
                SELECT AVG(processing_time) FROM scan_history
            """) or 0.0
            
            threat_dist = {row["threat_level"]: row["count"] for row in threat_distribution}
            
            return SystemStats(
                uptime_seconds=metrics_collector.get_uptime(),
                total_scans=total_scans,
                total_users=total_users,
                total_requests=metrics_collector.get_total_requests(),
                average_scan_time=avg_scan_time,
                threat_distribution=threat_dist,
                system_health="healthy"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch system stats: {str(e)}"
        )

@router.get("/engines/status")
@require_permission("view_engines")
async def get_engines_status(current_user: dict = Depends(get_current_active_user)):
    """Get status of all detection engines"""
    try:
        engines = get_all_engines()
        status_report = {}
        
        for engine_name, engine in engines.items():
            health = await engine.health_check()
            status_report[engine_name.value] = health
        
        return status_report
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch engine status: {str(e)}"
        )

# Utility functions
async def _calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Calculate file hash"""
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

async def _calculate_file_entropy(file_path: str) -> float:
    """Calculate file entropy"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        
        if len(data) == 0:
            return 0.0
        
        entropy = 0.0
        for x in range(256):
            p_x = float(data.count(x)) / len(data)
            if p_x > 0:
                entropy += -p_x * math.log2(p_x)
        
        return entropy
        
    except Exception:
        return 0.0

import math  # Add this at the top if not already present

# Initialize engines on startup
@router.on_event("startup")
async def startup_event():
    """Initialize engines on startup"""
    await initialize_all_engines()