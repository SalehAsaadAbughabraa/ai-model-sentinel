# api/routes/health_routes.py
"""
❤️ Health Routes for AI Model Sentinel API
📦 RESTful API endpoints for system health monitoring
👨‍💻 Author: Saleh Abughabraa  
🚀 Version: 2.0.0
💡 Business Logic:
   - Comprehensive system health monitoring and metrics collection
   - Kubernetes-ready health probes (liveness, readiness, startup)
   - Integration with all system components for dependency checking
   - Performance metrics and resource utilization tracking
   - Prometheus metrics export for global monitoring
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any, List
import logging
import time
import psutil
import asyncio
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import os

# Import system modules
from config import settings
from security.auth_manager import AuthenticationManager
from database.database_manager import DatabaseManager
from database.redis_manager import RedisManager
from security.encryption_manager import EncryptionManager
from visualization.chart_generator import chart_generator
from analytics.data_pipeline import DataPipelineManager
from analytics.threat_engine import ThreatIntelligenceEngine
from compliance.audit_logger import AuditLogger

# Initialize router
router = APIRouter(prefix="/api/v1/health", tags=["health"])
logger = logging.getLogger(settings.LOGGER_NAME)
security = HTTPBearer()

# Initialize system components
auth_manager = AuthenticationManager()
db_manager = DatabaseManager()
redis_manager = RedisManager()
encryption_manager = EncryptionManager()
pipeline_manager = DataPipelineManager()
threat_engine = ThreatIntelligenceEngine()
audit_logger = AuditLogger()

# Global variables for health tracking
start_time = datetime.now(timezone.utc)
health_metrics = {
    "total_requests": 0,
    "error_count": 0,
    "last_error": None,
    "component_status": {}
}

# Dependency for admin access
async def require_admin_access(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Require admin privileges for detailed health information"""
    try:
        user = await auth_manager.verify_token(credentials.credentials)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        # Check if user has admin role
        if user.get("role") != "admin" and not await auth_manager.check_permission(user["user_id"], "system:monitor"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required for detailed health information"
            )
        
        return user
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

async def check_component_health(component_name: str, health_check_func, timeout: int = 10) -> Dict[str, Any]:
    """Check health of a specific component with timeout"""
    try:
        # Execute health check with timeout
        health_result = await asyncio.wait_for(health_check_func(), timeout=timeout)
        return {
            "status": health_result.get("status", "healthy"),
            "response_time": health_result.get("response_time", 0),
            "details": health_result.get("details", {}),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": None
        }
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "response_time": timeout * 1000,
            "details": {"message": f"Health check timed out after {timeout} seconds"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": f"Timeout after {timeout} seconds"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "response_time": 0,
            "details": {"message": str(e)},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

async def get_system_metrics() -> Dict[str, Any]:
    """Get system-level performance metrics"""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        
        # Get disk usage
        disk = psutil.disk_usage('/')
        
        # Get process information
        process = psutil.Process()
        
        # Get network statistics
        net_io = psutil.net_io_counters()
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "cores": psutil.cpu_count(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "memory": {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "used_percent": memory.percent,
                "used_gb": round(memory.used / (1024 ** 3), 2)
            },
            "disk": {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "used_percent": disk.percent
            },
            "process": {
                "memory_mb": round(process.memory_info().rss / (1024 ** 2), 2),
                "cpu_percent": process.cpu_percent(),
                "threads": process.num_threads(),
                "open_files": len(process.open_files()) if process.open_files() else 0
            },
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv
            } if net_io else {}
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {"error": str(e)}

@router.get("/", summary="Basic system health status")
async def health_check():
    """
    ❤️ Basic health check - Public endpoint for load balancers and monitoring
    """
    global health_metrics
    health_metrics["total_requests"] += 1
    
    try:
        # Check critical components quickly
        start_time = time.time()
        
        # Check database connectivity (basic)
        db_health = await check_component_health("database", db_manager.health_check, timeout=5)
        
        # Check Redis connectivity (basic)
        redis_health = await check_component_health("redis", redis_manager.health_check, timeout=5)
        
        response_time = round((time.time() - start_time) * 1000, 2)
        
        # Determine overall status
        critical_services = [db_health, redis_health]
        unhealthy_services = [s for s in critical_services if s["status"] not in ["healthy", "degraded"]]
        
        overall_status = "healthy" if not unhealthy_services else "unhealthy"
        
        # Log health check (without sensitive details)
        await audit_logger.log(
            user_id="system",
            tenant_id="system",
            action="health_check_basic",
            resource_type="system",
            resource_id="health",
            details={"status": overall_status, "response_time_ms": response_time}
        )
        
        return {
            "status": overall_status,
            "version": "2.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_time_ms": response_time,
            "services": {
                "api": "healthy",
                "database": db_health["status"],
                "redis": redis_health["status"]
            },
            "uptime_seconds": int((datetime.now(timezone.utc) - start_time).total_seconds())
        }
        
    except Exception as e:
        health_metrics["error_count"] += 1
        health_metrics["last_error"] = str(e)
        logger.error(f"Health check failed: {e}")
        
        return {
            "status": "unhealthy",
            "version": "2.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": "Health check execution failed",
            "services": {
                "api": "unhealthy",
                "database": "unknown",
                "redis": "unknown"
            }
        }

@router.get("/detailed", summary="Detailed health status with metrics")
async def detailed_health(current_user: Dict[str, Any] = Depends(require_admin_access)):
    """
    📊 Detailed health status with comprehensive system metrics - Admin only
    """
    global health_metrics
    health_metrics["total_requests"] += 1
    
    try:
        start_time = time.time()
        
        # Check all system components in parallel
        components_to_check = {
            "api_server": lambda: asyncio.sleep(0),  # Self-check
            "database": db_manager.health_check,
            "redis_cache": redis_manager.health_check,
            "encryption": encryption_manager.health_check,
            "authentication": auth_manager.health_check,
            "chart_generator": chart_generator.health_check,
            "data_pipeline": pipeline_manager.health_check,
            "threat_engine": threat_engine.health_check,
            "audit_logger": audit_logger.health_check
        }
        
        # Execute health checks concurrently
        health_tasks = []
        for name, check_func in components_to_check.items():
            task = check_component_health(name, check_func, timeout=10)
            health_tasks.append(task)
        
        health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        # Process results
        components_health = {}
        for i, (name, result) in enumerate(zip(components_to_check.keys(), health_results)):
            if isinstance(result, Exception):
                components_health[name] = {
                    "status": "error",
                    "response_time": 0,
                    "details": {"message": str(result)},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(result)
                }
            else:
                components_health[name] = result
        
        # Get system metrics
        system_metrics = await get_system_metrics()
        
        # Calculate overall status
        component_statuses = [comp["status"] for comp in components_health.values()]
        critical_components = ["database", "redis_cache", "authentication"]
        
        critical_status = all(
            components_health[comp]["status"] in ["healthy", "degraded"] 
            for comp in critical_components if comp in components_health
        )
        
        overall_status = "healthy" if critical_status else "unhealthy"
        
        response_time = round((time.time() - start_time) * 1000, 2)
        
        # Update global metrics
        health_metrics["component_status"] = components_health
        
        # Log detailed health check
        await audit_logger.log(
            user_id=current_user["user_id"],
            tenant_id=current_user.get("tenant_id", "default"),
            action="health_check_detailed",
            resource_type="system",
            resource_id="health",
            details={
                "status": overall_status,
                "response_time_ms": response_time,
                "components_checked": len(components_health)
            }
        )
        
        return {
            "status": overall_status,
            "version": "2.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response_time_ms": response_time,
            "uptime_seconds": int((datetime.now(timezone.utc) - start_time).total_seconds()),
            "system_metrics": system_metrics,
            "components": components_health,
            "health_metrics": {
                "total_requests": health_metrics["total_requests"],
                "error_count": health_metrics["error_count"],
                "last_error": health_metrics["last_error"]
            }
        }
        
    except Exception as e:
        health_metrics["error_count"] += 1
        health_metrics["last_error"] = str(e)
        logger.error(f"Detailed health check failed: {e}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Detailed health check failed"
        )

@router.get("/liveness", summary="Kubernetes liveness probe")
async def liveness_probe():
    """
    🔄 Kubernetes liveness probe - Check if service is running
    """
    try:
        # Simple check - if we can respond, we're alive
        basic_health = await health_check()
        
        if basic_health["status"] == "healthy":
            return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}
        else:
            return {"status": "unhealthy", "timestamp": datetime.now(timezone.utc).isoformat()}
            
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        return {"status": "unhealthy", "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/readiness", summary="Kubernetes readiness probe")
async def readiness_probe():
    """
    📦 Kubernetes readiness probe - Check if service is ready to receive traffic
    """
    try:
        # Check critical dependencies
        db_ready = await check_component_health("database", db_manager.health_check, timeout=5)
        redis_ready = await check_component_health("redis", redis_manager.health_check, timeout=5)
        
        # Service is ready if critical dependencies are healthy
        is_ready = (
            db_ready["status"] in ["healthy", "degraded"] and
            redis_ready["status"] in ["healthy", "degraded"]
        )
        
        status = "ready" if is_ready else "not_ready"
        
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dependencies": {
                "database": db_ready["status"],
                "redis": redis_ready["status"]
            }
        }
        
    except Exception as e:
        logger.error(f"Readiness probe failed: {e}")
        return {"status": "not_ready", "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/startup", summary="Kubernetes startup probe")
async def startup_probe():
    """
    🚀 Kubernetes startup probe - Check if service has started completely
    """
    try:
        # Check all critical components for startup
        components_to_check = {
            "database": db_manager.health_check,
            "redis": redis_manager.health_check,
            "authentication": auth_manager.health_check,
            "encryption": encryption_manager.health_check
        }
        
        # Execute startup checks
        startup_tasks = []
        for name, check_func in components_to_check.items():
            task = check_component_health(name, check_func, timeout=15)
            startup_tasks.append(task)
        
        startup_results = await asyncio.gather(*startup_tasks)
        
        # All critical components must be healthy for startup
        all_healthy = all(
            result["status"] in ["healthy", "degraded"] 
            for result in startup_results
        )
        
        status = "started" if all_healthy else "starting"
        
        return {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                name: result["status"] 
                for name, result in zip(components_to_check.keys(), startup_results)
            }
        }
        
    except Exception as e:
        logger.error(f"Startup probe failed: {e}")
        return {"status": "starting", "timestamp": datetime.now(timezone.utc).isoformat()}

@router.get("/metrics", summary="Prometheus metrics endpoint")
async def metrics_endpoint():
    """
    📈 Prometheus metrics export for system monitoring
    """
    try:
        # Get basic health information
        basic_health = await health_check()
        
        # Get system metrics
        system_metrics = await get_system_metrics()
        
        # Format metrics in Prometheus text format
        metrics_lines = []
        
        # Basic health metrics
        metrics_lines.append(f'sentinel_health_status{{service="api"}} {1 if basic_health["status"] == "healthy" else 0}')
        metrics_lines.append(f'sentinel_uptime_seconds {basic_health.get("uptime_seconds", 0)}')
        metrics_lines.append(f'sentinel_health_requests_total {health_metrics["total_requests"]}')
        metrics_lines.append(f'sentinel_health_errors_total {health_metrics["error_count"]}')
        
        # System metrics
        if "cpu" in system_metrics:
            metrics_lines.append(f'sentinel_cpu_usage_percent {system_metrics["cpu"]["percent"]}')
            metrics_lines.append(f'sentinel_cpu_cores {system_metrics["cpu"]["cores"]}')
        
        if "memory" in system_metrics:
            metrics_lines.append(f'sentinel_memory_usage_percent {system_metrics["memory"]["used_percent"]}')
            metrics_lines.append(f'sentinel_memory_usage_bytes {int(system_metrics["memory"]["used_gb"] * 1024 * 1024 * 1024)}')
            metrics_lines.append(f'sentinel_memory_total_bytes {int(system_metrics["memory"]["total_gb"] * 1024 * 1024 * 1024)}')
        
        if "disk" in system_metrics:
            metrics_lines.append(f'sentinel_disk_usage_percent {system_metrics["disk"]["used_percent"]}')
            metrics_lines.append(f'sentinel_disk_usage_bytes {int(system_metrics["disk"]["used_gb"] * 1024 * 1024 * 1024)}')
        
        if "process" in system_metrics:
            metrics_lines.append(f'sentinel_process_memory_bytes {int(system_metrics["process"]["memory_mb"] * 1024 * 1024)}')
            metrics_lines.append(f'sentinel_process_threads {system_metrics["process"]["threads"]}')
        
        # Component status metrics
        for component, status in health_metrics["component_status"].items():
            status_value = 1 if status.get("status") in ["healthy", "degraded"] else 0
            metrics_lines.append(f'sentinel_component_healthy{{component="{component}"}} {status_value}')
            metrics_lines.append(f'sentinel_component_response_ms{{component="{component}"}} {status.get("response_time", 0)}')
        
        metrics_text = "\n".join(metrics_lines)
        
        return metrics_text
        
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        return f"sentinel_metrics_error 1\n"

@router.get("/components", summary="Individual component health status")
async def component_health(
    component: str,
    current_user: Dict[str, Any] = Depends(require_admin_access)
):
    """
    🔧 Get health status for a specific system component
    """
    try:
        component_checks = {
            "database": db_manager.health_check,
            "redis": redis_manager.health_check,
            "encryption": encryption_manager.health_check,
            "authentication": auth_manager.health_check,
            "charts": chart_generator.health_check,
            "pipeline": pipeline_manager.health_check,
            "threat": threat_engine.health_check,
            "audit": audit_logger.health_check
        }
        
        if component not in component_checks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown component: {component}. Available: {list(component_checks.keys())}"
            )
        
        health_result = await check_component_health(component, component_checks[component], timeout=10)
        
        return {
            "component": component,
            "status": health_result["status"],
            "response_time_ms": health_result["response_time"],
            "timestamp": health_result["timestamp"],
            "details": health_result["details"],
            "error": health_result["error"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Component health check failed for {component}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed for component: {component}"
        )

@router.get("/history", summary="Health check history and trends")
async def health_history(
    hours: int = 24,
    current_user: Dict[str, Any] = Depends(require_admin_access)
):
    """
    📊 Get health check history and trend analysis
    """
    try:
        # This would typically query a time-series database
        # For now, return mock data with current metrics
        current_health = await detailed_health(current_user)
        
        return {
            "period_hours": hours,
            "current_status": current_health["status"],
            "availability_percent": 99.8,  # Mock data
            "average_response_time_ms": 45.2,  # Mock data
            "incident_count": 2,  # Mock data
            "last_incident": "2024-01-09T15:30:00Z",  # Mock data
            "trend": "stable",  # improving, stable, degrading
            "components_history": {
                "database": {"uptime_percent": 99.9, "avg_response_ms": 12.5},
                "redis": {"uptime_percent": 99.8, "avg_response_ms": 2.1},
                "api": {"uptime_percent": 99.95, "avg_response_ms": 8.7}
            }
        }
        
    except Exception as e:
        logger.error(f"Health history retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve health history"
        )

# Background task to periodically update health metrics
async def update_health_metrics():
    """Periodically update health metrics in background"""
    while True:
        try:
            # Update component status
            basic_health = await health_check()
            health_metrics["component_status"] = basic_health.get("services", {})
            
            # Sleep for 30 seconds
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Health metrics update failed: {e}")
            await asyncio.sleep(60)  # Wait longer on error

# Export router
__all__ = ["router"]