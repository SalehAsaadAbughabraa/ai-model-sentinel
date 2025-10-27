# api/routes/scan_routes.py
"""
🔍 Scan Routes for AI Model Sentinel API
📦 RESTful API endpoints for security scan operations
👨‍💻 Author: Saleh Abughabraa  
🚀 Version: 2.0.0
💡 Business Logic:
   - Manages security scan creation, execution, and monitoring
   - Provides real-time scan status and results
   - Supports multi-tenant scan operations
   - Integrates with threat intelligence and analytics
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional, Dict, Any
import logging
import asyncio
from datetime import datetime, timezone
from uuid import uuid4
import json
from enum import Enum

# Import system modules
from config import settings
from security.auth_manager import AuthenticationManager, RBAC
from database.database_manager import DatabaseManager
from analytics.threat_engine import ThreatIntelligenceEngine
from visualization.chart_generator import create_chart_from_query, ExportFormat
from compliance.audit_logger import AuditLogger
from analytics.data_pipeline import DataPipelineManager

# Initialize router
router = APIRouter(prefix="/api/v1/scans", tags=["security-scans"])
logger = logging.getLogger(settings.LOGGER_NAME)
security = HTTPBearer()

# Initialize system components
auth_manager = AuthenticationManager()
db_manager = DatabaseManager()
threat_engine = ThreatIntelligenceEngine()
audit_logger = AuditLogger()
pipeline_manager = DataPipelineManager()

# Enums for scan types and status
class ScanType(str, Enum):
    WEB = "web"
    NETWORK = "network"
    CONTAINER = "container"
    AI_MODEL = "ai_model"
    CODE = "code"
    INFRASTRUCTURE = "infrastructure"

class ScanStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ThreatLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

# WebSocket connections for real-time updates
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, scan_id: str):
        await websocket.accept()
        if scan_id not in self.active_connections:
            self.active_connections[scan_id] = []
        self.active_connections[scan_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, scan_id: str):
        if scan_id in self.active_connections:
            self.active_connections[scan_id].remove(websocket)
            if not self.active_connections[scan_id]:
                del self.active_connections[scan_id]
    
    async def send_update(self, scan_id: str, message: Dict[str, Any]):
        if scan_id in self.active_connections:
            disconnected = []
            for connection in self.active_connections[scan_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    disconnected.append(connection)
            
            for connection in disconnected:
                self.disconnect(connection, scan_id)

connection_manager = ConnectionManager()

# Dependency injections
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user with RBAC"""
    try:
        user = await auth_manager.verify_token(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        return user
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def require_permission(permission: str):
    """RBAC permission requirement"""
    async def permission_dependency(current_user: Dict[str, Any] = Depends(get_current_user)):
        has_permission = await auth_manager.check_permission(
            current_user["user_id"], permission, current_user.get("tenant_id")
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return permission_dependency

# Rate limiting storage
scan_attempts = {}

def check_rate_limit(user_id: str, max_requests: int = 10, window_seconds: int = 60):
    """Rate limiting for scan operations"""
    current_time = datetime.now(timezone.utc).timestamp()
    user_key = f"scan_ratelimit_{user_id}"
    
    if user_key not in scan_attempts:
        scan_attempts[user_key] = []
    
    # Clean old attempts
    scan_attempts[user_key] = [
        attempt_time for attempt_time in scan_attempts[user_key]
        if current_time - attempt_time < window_seconds
    ]
    
    if len(scan_attempts[user_key]) >= max_requests:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    scan_attempts[user_key].append(current_time)

@router.get("/health", summary="Scan service health check")
async def scan_health():
    """
    🔍 Check scan service health status with dependency checks
    """
    try:
        # Check all dependencies health
        dependencies_health = {}
        
        # Database health
        try:
            db_health = await db_manager.health_check()
            dependencies_health["database"] = db_health
        except Exception as e:
            dependencies_health["database"] = {"status": "unhealthy", "error": str(e)}
        
        # Authentication health
        try:
            auth_health = await auth_manager.health_check()
            dependencies_health["authentication"] = auth_health
        except Exception as e:
            dependencies_health["authentication"] = {"status": "unhealthy", "error": str(e)}
        
        # Threat engine health
        try:
            threat_health = await threat_engine.health_check()
            dependencies_health["threat_engine"] = threat_health
        except Exception as e:
            dependencies_health["threat_engine"] = {"status": "unhealthy", "error": str(e)}
        
        # Determine overall status
        all_healthy = all(
            dep.get("status") == "healthy" 
            for dep in dependencies_health.values()
        )
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "service": "scan-routes",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "2.0.0",
            "dependencies": dependencies_health
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "scan-routes",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

@router.post("/", summary="Create new security scan", status_code=status.HTTP_201_CREATED)
async def create_scan(
    scan_request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_permission("scan:create"))
):
    """
    🚀 Create and initiate a new security scan with multi-tenant support
    """
    try:
        # Rate limiting
        check_rate_limit(current_user["user_id"])
        
        # Validate scan type
        scan_type = scan_request.get("scan_type", ScanType.WEB)
        if scan_type not in [st.value for st in ScanType]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid scan type. Must be one of: {[st.value for st in ScanType]}"
            )
        
        # Generate scan ID
        scan_id = f"scan_{uuid4().hex[:16]}"
        tenant_id = current_user.get("tenant_id", "default")
        
        # Handle scheduled scans
        schedule_time = scan_request.get("schedule_time")
        status = ScanStatus.SCHEDULED if schedule_time else ScanStatus.PENDING
        
        # Create scan record in database
        scan_data = {
            "scan_id": scan_id,
            "tenant_id": tenant_id,
            "user_id": current_user["user_id"],
            "scan_type": scan_type,
            "target": scan_request.get("target", ""),
            "config": scan_request.get("config", {}),
            "status": status,
            "schedule_time": datetime.fromisoformat(schedule_time.replace('Z', '+00:00')) if schedule_time else None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        await db_manager.execute(
            """
            INSERT INTO security_scans 
            (scan_id, tenant_id, user_id, scan_type, target, config, status, schedule_time, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            scan_data["scan_id"],
            scan_data["tenant_id"],
            scan_data["user_id"],
            scan_data["scan_type"],
            scan_data["target"],
            json.dumps(scan_data["config"]),
            scan_data["status"],
            scan_data["schedule_time"],
            scan_data["created_at"],
            scan_data["updated_at"]
        )
        
        # Start scan execution (immediate or scheduled)
        if status == ScanStatus.PENDING:
            background_tasks.add_task(
                execute_scan_pipeline,
                scan_id,
                tenant_id,
                current_user["user_id"],
                scan_request
            )
        
        # Audit log
        await audit_logger.log(
            user_id=current_user["user_id"],
            tenant_id=tenant_id,
            action="scan_created",
            resource_type="security_scan",
            resource_id=scan_id,
            details={
                "scan_type": scan_type,
                "target": scan_request.get("target"),
                "scheduled": bool(schedule_time)
            }
        )
        
        return {
            "scan_id": scan_id,
            "status": status,
            "message": "Scan created successfully" + (" and scheduled" if schedule_time else " and queued for execution"),
            "created_at": scan_data["created_at"].isoformat(),
            "schedule_time": schedule_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scan creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create scan"
        )

async def execute_scan_pipeline(scan_id: str, tenant_id: str, user_id: str, scan_request: Dict[str, Any]):
    """
    🎯 Execute complete scan pipeline with ETL and threat analysis
    """
    try:
        # Update scan status to running
        await update_scan_status(scan_id, ScanStatus.RUNNING)
        
        # Notify via WebSocket
        await connection_manager.send_update(scan_id, {
            "scan_id": scan_id,
            "status": "running",
            "message": "Scan execution started",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress": 0
        })
        
        # Step 1: Data extraction
        await connection_manager.send_update(scan_id, {
            "scan_id": scan_id,
            "status": "running",
            "step": "data_extraction",
            "message": "Extracting target data",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress": 25
        })
        
        extraction_result = await pipeline_manager.extract_data(
            source_type=scan_request.get("scan_type", "web"),
            source_config=scan_request.get("config", {})
        )
        
        # Step 2: Data transformation and loading
        await connection_manager.send_update(scan_id, {
            "scan_id": scan_id,
            "status": "running",
            "step": "data_processing",
            "message": "Processing and analyzing data",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress": 50
        })
        
        processed_data = await pipeline_manager.transform_data(
            data=extraction_result,
            transformations=scan_request.get("transformations", [])
        )
        
        # Step 3: Threat analysis
        await connection_manager.send_update(scan_id, {
            "scan_id": scan_id,
            "status": "running",
            "step": "threat_analysis",
            "message": "Analyzing threats and vulnerabilities",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress": 75
        })
        
        threat_results = await threat_engine.analyze_threats(
            scan_data=processed_data,
            scan_context=scan_request,
            tenant_id=tenant_id
        )
        
        # Step 4: Store results
        await connection_manager.send_update(scan_id, {
            "scan_id": scan_id,
            "status": "running",
            "step": "storing_results",
            "message": "Storing scan results and generating reports",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress": 90
        })
        
        # Store comprehensive results
        await db_manager.execute(
            """
            INSERT INTO scan_results 
            (scan_id, tenant_id, threat_results, risk_score, created_at)
            VALUES ($1, $2, $3, $4, $5)
            """,
            scan_id,
            tenant_id,
            json.dumps(threat_results),
            threat_results.get("risk_score", 0),
            datetime.now(timezone.utc)
        )
        
        # Update scan status to completed
        await update_scan_status(scan_id, ScanStatus.COMPLETED, threat_results)
        
        # Final notification
        await connection_manager.send_update(scan_id, {
            "scan_id": scan_id,
            "status": "completed",
            "message": "Scan completed successfully",
            "results_summary": threat_results.get("summary", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "progress": 100
        })
        
        # Audit log
        await audit_logger.log(
            user_id=user_id,
            tenant_id=tenant_id,
            action="scan_completed",
            resource_type="security_scan",
            resource_id=scan_id,
            details={"threat_summary": threat_results.get("summary", {})}
        )
        
    except Exception as e:
        logger.error(f"Scan execution failed for {scan_id}: {e}")
        
        # Update scan status to failed
        await update_scan_status(scan_id, ScanStatus.FAILED, {"error": str(e)})
        
        # Error notification
        await connection_manager.send_update(scan_id, {
            "scan_id": scan_id,
            "status": "failed",
            "message": f"Scan failed: {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Audit log
        await audit_logger.log(
            user_id=user_id,
            tenant_id=tenant_id,
            action="scan_failed",
            resource_type="security_scan",
            resource_id=scan_id,
            details={"error": str(e)}
        )

async def update_scan_status(scan_id: str, status: ScanStatus, results: Optional[Dict[str, Any]] = None):
    """Update scan status in database"""
    try:
        update_data = {
            "status": status,
            "updated_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc) if status == ScanStatus.COMPLETED else None
        }
        
        query = """
            UPDATE security_scans 
            SET status = $1, updated_at = $2, completed_at = $3
            WHERE scan_id = $4
        """
        params = [update_data["status"], update_data["updated_at"], update_data["completed_at"], scan_id]
        
        if results:
            # Also update results if provided
            query = """
                UPDATE security_scans 
                SET status = $1, updated_at = $2, completed_at = $3, results = $4
                WHERE scan_id = $5
            """
            params = [update_data["status"], update_data["updated_at"], update_data["completed_at"], json.dumps(results), scan_id]
        
        await db_manager.execute(query, *params)
    except Exception as e:
        logger.error(f"Failed to update scan status for {scan_id}: {e}")

@router.get("/", summary="List security scans")
async def list_scans(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Items per page"),
    status: Optional[ScanStatus] = Query(None, description="Filter by status"),
    scan_type: Optional[ScanType] = Query(None, description="Filter by scan type"),
    date_from: Optional[str] = Query(None, description="Start date (ISO format)"),
    date_to: Optional[str] = Query(None, description="End date (ISO format)"),
    search: Optional[str] = Query(None, description="Search in scan ID or target"),
    current_user: Dict[str, Any] = Depends(require_permission("scan:read"))
):
    """
    📋 List all security scans with pagination, filtering, and multi-tenant support
    """
    try:
        tenant_id = current_user.get("tenant_id", "default")
        user_id = current_user["user_id"]
        
        # Build query with filters
        base_query = """
            SELECT scan_id, scan_type, target, status, created_at, updated_at, completed_at
            FROM security_scans 
            WHERE tenant_id = $1
        """
        query_params = [tenant_id]
        param_count = 1
        
        # Apply RBAC filter
        if not await auth_manager.check_permission(user_id, "scan:read_all", tenant_id):
            param_count += 1
            base_query += f" AND user_id = ${param_count}"
            query_params.append(user_id)
        
        # Apply status filter
        if status:
            param_count += 1
            base_query += f" AND status = ${param_count}"
            query_params.append(status.value)
        
        # Apply scan type filter
        if scan_type:
            param_count += 1
            base_query += f" AND scan_type = ${param_count}"
            query_params.append(scan_type.value)
        
        # Apply date filters
        if date_from:
            param_count += 1
            base_query += f" AND created_at >= ${param_count}"
            query_params.append(datetime.fromisoformat(date_from.replace('Z', '+00:00')))
        
        if date_to:
            param_count += 1
            base_query += f" AND created_at <= ${param_count}"
            query_params.append(datetime.fromisoformat(date_to.replace('Z', '+00:00')))
        
        # Apply search filter
        if search:
            param_count += 1
            base_query += f" AND (scan_id LIKE ${param_count} OR target LIKE ${param_count})"
            query_params.append(f"%{search}%")
        
        # Count total records
        count_query = f"SELECT COUNT(*) as total FROM ({base_query}) as filtered"
        total_result = await db_manager.fetch_one(count_query, *query_params)
        total = total_result["total"] if total_result else 0
        
        # Apply pagination
        offset = (page - 1) * page_size
        paginated_query = f"""
            {base_query}
            ORDER BY created_at DESC
            LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """
        query_params.extend([page_size, offset])
        
        # Execute query
        scans = await db_manager.fetch_all(paginated_query, *query_params)
        
        # Apply data masking for non-admin users
        masked_scans = []
        for scan in scans:
            masked_scan = dict(scan)
            # Convert datetime objects to ISO format
            for date_field in ['created_at', 'updated_at', 'completed_at']:
                if masked_scan.get(date_field):
                    masked_scan[date_field] = masked_scan[date_field].isoformat()
            
            masked_scans.append(masked_scan)
        
        return {
            "scans": masked_scans,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "filters": {
                "status": status.value if status else None,
                "scan_type": scan_type.value if scan_type else None,
                "date_from": date_from,
                "date_to": date_to,
                "search": search
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to list scans: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve scans"
        )

@router.get("/{scan_id}", summary="Get scan details")
async def get_scan(
    scan_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("scan:read"))
):
    """
    🔍 Get detailed information about a specific scan with multi-tenant access control
    """
    try:
        tenant_id = current_user.get("tenant_id", "default")
        user_id = current_user["user_id"]
        
        # Build query with access control
        query = """
            SELECT scan_id, tenant_id, user_id, scan_type, target, config, status, 
                   created_at, updated_at, completed_at, schedule_time, results
            FROM security_scans 
            WHERE scan_id = $1 AND tenant_id = $2
        """
        scan = await db_manager.fetch_one(query, scan_id, tenant_id)
        
        if not scan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scan not found"
            )
        
        # Check if user has access to this specific scan
        if scan["user_id"] != user_id and not await auth_manager.check_permission(user_id, "scan:read_all", tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this scan"
            )
        
        # Apply data masking
        scan_data = dict(scan)
        if not await auth_manager.check_permission(user_id, "scan:view_sensitive", tenant_id):
            # Mask sensitive configuration details
            if "config" in scan_data and scan_data["config"]:
                config = scan_data["config"]
                if isinstance(config, dict):
                    # Mask API keys, tokens, passwords, etc.
                    sensitive_keys = ["api_key", "token", "password", "secret", "private_key", "credential"]
                    for key in sensitive_keys:
                        if key in config:
                            config[key] = "***MASKED***"
                scan_data["config"] = config
        
        # Convert datetime objects to ISO format
        for date_field in ['created_at', 'updated_at', 'completed_at', 'schedule_time']:
            if scan_data.get(date_field):
                scan_data[date_field] = scan_data[date_field].isoformat()
        
        # Add report links
        if scan_data["status"] == ScanStatus.COMPLETED:
            scan_data["report_links"] = {
                "json": f"/api/v1/scans/{scan_id}/results?format=json",
                "html": f"/api/v1/scans/{scan_id}/results?format=html",
                "pdf": f"/api/v1/scans/{scan_id}/results?format=pdf"
            }
        
        return scan_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scan {scan_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve scan details"
        )

@router.get("/{scan_id}/results", summary="Get scan results")
async def get_scan_results(
    scan_id: str,
    format: str = Query("json", regex="^(json|html|pdf)$"),
    include_visualization: bool = Query(True),
    current_user: Dict[str, Any] = Depends(require_permission("scan:read"))
):
    """
    📊 Get comprehensive results for a completed scan with multiple export formats
    """
    try:
        tenant_id = current_user.get("tenant_id", "default")
        user_id = current_user["user_id"]
        
        # Get scan results with access control
        query = """
            SELECT s.scan_id, s.user_id, s.scan_type, s.target, s.status, s.created_at, 
                   r.threat_results, r.risk_score
            FROM security_scans s
            LEFT JOIN scan_results r ON s.scan_id = r.scan_id
            WHERE s.scan_id = $1 AND s.tenant_id = $2
        """
        result = await db_manager.fetch_one(query, scan_id, tenant_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scan results not found"
            )
        
        # Check access
        if result["user_id"] != user_id and not await auth_manager.check_permission(user_id, "scan:read_all", tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to these scan results"
            )
        
        if result["status"] != ScanStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Scan is not completed. Current status: {result['status']}"
            )
        
        threat_results = result["threat_results"] or {}
        
        # Base response data
        response_data = {
            "scan_id": scan_id,
            "scan_type": result["scan_type"],
            "target": result["target"],
            "status": result["status"],
            "scan_date": result["created_at"].isoformat() if result["created_at"] else None,
            "risk_score": result["risk_score"],
            "threat_summary": threat_results.get("summary", {}),
            "detailed_findings": self._categorize_findings(threat_results.get("findings", [])),
            "recommendations": threat_results.get("recommendations", [])
        }
        
        # Add visualization if requested
        if include_visualization and threat_results.get("findings"):
            visualization_data = self._prepare_visualization_data(threat_results["findings"])
            visualization_html = await create_chart_from_query(
                query_result=visualization_data,
                chart_type="bar",
                theme="security"
            )
            response_data["visualization"] = visualization_html
        
        # Handle different formats
        if format == "html":
            return self._generate_html_report(response_data)
        elif format == "pdf":
            return await self._generate_pdf_report(scan_id, response_data)
        else:  # json
            return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get scan results for {scan_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve scan results"
        )

def _categorize_findings(self, findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize findings by threat level"""
    categorized = {level.value: [] for level in ThreatLevel}
    
    for finding in findings:
        level = finding.get("level", ThreatLevel.INFO.value)
        if level in categorized:
            categorized[level].append(finding)
    
    return categorized

def _prepare_visualization_data(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare data for visualization charts"""
    threat_counts = {}
    for finding in findings:
        level = finding.get("level", ThreatLevel.INFO.value)
        threat_counts[level] = threat_counts.get(level, 0) + 1
    
    return [{"threat_level": level, "count": count} for level, count in threat_counts.items()]

@router.post("/{scan_id}/retry", summary="Retry failed scan")
async def retry_scan(
    scan_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(require_permission("scan:create"))
):
    """
    🔄 Retry a failed security scan
    """
    try:
        tenant_id = current_user.get("tenant_id", "default")
        
        # Get original scan
        query = """
            SELECT scan_id, user_id, config, status, scan_type, target
            FROM security_scans 
            WHERE scan_id = $1 AND tenant_id = $2
        """
        scan = await db_manager.fetch_one(query, scan_id, tenant_id)
        
        if not scan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scan not found"
            )
        
        if scan["status"] != ScanStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only retry failed scans"
            )
        
        # Check access
        if scan["user_id"] != current_user["user_id"] and not await auth_manager.check_permission(current_user["user_id"], "scan:retry_all", tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to retry this scan"
            )
        
        # Restart scan in background
        background_tasks.add_task(
            execute_scan_pipeline,
            scan_id,
            tenant_id,
            current_user["user_id"],
            {
                "scan_type": scan["scan_type"],
                "target": scan["target"],
                "config": scan["config"]
            }
        )
        
        # Audit log
        await audit_logger.log(
            user_id=current_user["user_id"],
            tenant_id=tenant_id,
            action="scan_retried",
            resource_type="security_scan",
            resource_id=scan_id
        )
        
        return {
            "scan_id": scan_id,
            "status": "retrying",
            "message": "Scan retry initiated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scan retry failed for {scan_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retry scan"
        )

@router.post("/{scan_id}/cancel", summary="Cancel ongoing scan")
async def cancel_scan(
    scan_id: str,
    current_user: Dict[str, Any] = Depends(require_permission("scan:delete"))
):
    """
    🚫 Cancel an ongoing security scan
    """
    try:
        tenant_id = current_user.get("tenant_id", "default")
        
        # Get scan status
        query = """
            SELECT scan_id, user_id, status
            FROM security_scans 
            WHERE scan_id = $1 AND tenant_id = $2
        """
        scan = await db_manager.fetch_one(query, scan_id, tenant_id)
        
        if not scan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Scan not found"
            )
        
        if scan["status"] not in [ScanStatus.PENDING, ScanStatus.RUNNING, ScanStatus.SCHEDULED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only cancel pending, scheduled, or running scans"
            )
        
        # Check access
        if scan["user_id"] != current_user["user_id"] and not await auth_manager.check_permission(current_user["user_id"], "scan:cancel_all", tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to cancel this scan"
            )
        
        # Update status to cancelled
        await update_scan_status(scan_id, ScanStatus.CANCELLED)
        
        # Notify via WebSocket
        await connection_manager.send_update(scan_id, {
            "scan_id": scan_id,
            "status": "cancelled",
            "message": "Scan cancelled by user",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Audit log
        await audit_logger.log(
            user_id=current_user["user_id"],
            tenant_id=tenant_id,
            action="scan_cancelled",
            resource_type="security_scan",
            resource_id=scan_id
        )
        
        return {
            "scan_id": scan_id,
            "status": "cancelled",
            "message": "Scan cancelled successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scan cancellation failed for {scan_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel scan"
        )

@router.websocket("/{scan_id}/ws")
async def websocket_endpoint(
    websocket: WebSocket, 
    scan_id: str,
    token: str = Query(...)
):
    """
    📡 WebSocket for real-time scan updates
    """
    try:
        # Authenticate WebSocket connection
        user = await auth_manager.verify_token(token)
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Check scan access
        tenant_id = user.get("tenant_id", "default")
        query = """
            SELECT scan_id, user_id 
            FROM security_scans 
            WHERE scan_id = $1 AND tenant_id = $2
        """
        scan = await db_manager.fetch_one(query, scan_id, tenant_id)
        
        if not scan:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        # Check if user has access to this scan
        if scan["user_id"] != user["user_id"] and not await auth_manager.check_permission(user["user_id"], "scan:read_all", tenant_id):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        await connection_manager.connect(websocket, scan_id)
        
        try:
            while True:
                # Keep connection alive and handle messages
                data = await websocket.receive_text()
                # Could handle client messages here (e.g., pause/resume requests)
        except WebSocketDisconnect:
            connection_manager.disconnect(websocket, scan_id)
            
    except Exception as e:
        logger.error(f"WebSocket error for scan {scan_id}: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass

def _generate_html_report(self, scan_data: Dict[str, Any]) -> str:
    """Generate HTML report for scan results"""
    # Simplified HTML report generation
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security Scan Report - {scan_data['scan_id']}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
            .finding {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
            .critical {{ border-left-color: #dc3545; background: #f8d7da; }}
            .high {{ border-left-color: #fd7e14; background: #fff3cd; }}
            .medium {{ border-left-color: #ffc107; background: #fff3cd; }}
            .low {{ border-left-color: #28a745; background: #d4edda; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Security Scan Report</h1>
            <p><strong>Scan ID:</strong> {scan_data['scan_id']}</p>
            <p><strong>Target:</strong> {scan_data['target']}</p>
            <p><strong>Risk Score:</strong> {scan_data['risk_score']}</p>
        </div>
        <div class="visualization">
            {scan_data.get('visualization', '')}
        </div>
        <div class="findings">
            <h2>Detailed Findings</h2>
            <!-- Findings would be rendered here -->
        </div>
    </body>
    </html>
    """
    return html_content

async def _generate_pdf_report(self, scan_id: str, scan_data: Dict[str, Any]) -> str:
    """Generate PDF report for scan results"""
    # This would integrate with a PDF generation service
    # For now, return a placeholder
    pdf_url = f"/api/v1/reports/{scan_id}.pdf"
    
    # Audit log for PDF generation
    await audit_logger.log(
        user_id=scan_data.get("user_id", "system"),
        tenant_id=scan_data.get("tenant_id", "default"),
        action="pdf_report_generated",
        resource_type="security_scan",
        resource_id=scan_id
    )
    
    return {"pdf_url": pdf_url, "message": "PDF report generation initiated"}

# Export router
__all__ = ["router"]