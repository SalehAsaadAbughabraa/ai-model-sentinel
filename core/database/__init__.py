"""
🎯 Database Module Package
📦 Contains all database management components for AI Model Sentinel
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Description:
   This package provides comprehensive database management including:
   - PostgreSQL database operations and connection pooling
   - Redis caching and session management
   - Multi-database connector with intelligent routing
   - Database health monitoring and failover mechanisms
   - Multi-tenant data isolation and management
   - Cache-aside patterns and automatic synchronization
"""

import logging
import asyncio
from ..models.audit_models import AuditAction, EventSeverity
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# Import all database components
from .database_manager import DatabaseManager, database_manager, initialize_database, close_database
from .redis_manager import RedisManager, redis_manager, initialize_redis, close_redis
from .multi_db_connector import MultiDBConnector, multi_db, initialize_multi_database, close_multi_database

# Import for integration with other modules
try:
    from config.settings import settings, get_settings
    from audit_models import AuditLog, AuditAction
    from security_models import SecurityEvent, EventSeverity
    from scan_models import ScanRecord, ScanResult
    from threat_models import ThreatIntelligence, ThreatIndicator
    from user_models import User, Tenant
except ImportError:
    # Fallback imports for when modules are not available
    pass


logger = logging.getLogger("DatabasePackage")


class DatabasePackage:
    """
    🗃️ Comprehensive database package manager for AI Model Sentinel
    💡 Orchestrates all database operations with multi-tenant support and caching
    """
    
    def __init__(self):
        self.initialized = False
        self.health_status: Dict[str, Any] = {}
        self.tenant_operations: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self) -> bool:
        """
        🚀 Initialize all database components with proper orchestration
        💡 Ensures all databases are connected and ready for multi-tenant operations
        """
        logger.info("🔄 Initializing Database Package...")
        
        try:
            # Initialize multi-database connector (which initializes PostgreSQL and Redis)
            success = await initialize_multi_database()
            
            if success:
                self.initialized = True
                
                # Initialize tenant-specific configurations
                await self._initialize_tenant_operations()
                
                # Perform initial health check
                await self.health_check()
                
                logger.info("✅ Database Package initialized successfully")
                return True
            else:
                logger.error("❌ Database Package initialization failed")
                return False
                
        except Exception as e:
            logger.error(f"💥 Database Package initialization error: {e}")
            return False
    
    async def _initialize_tenant_operations(self):
        """Initialize tenant-specific database operations and configurations"""
        # This would load tenant-specific configurations from settings
        # For now, we'll create a basic structure
        self.tenant_operations = {
            "default": {
                "caching_enabled": True,
                "session_timeout": settings.security.session_timeout_minutes * 60,
                "scan_retention_days": 30,
                "threat_retention_days": 90
            }
        }
    
    async def store_scan_record(self, scan_record: Dict[str, Any], tenant_id: str = "default") -> bool:
        """
        💾 Store scan record with multi-tenant support and automatic caching
        💡 Uses cache-aside pattern for optimal performance
        """
        if not self.initialized:
            logger.error("❌ Database package not initialized")
            return False
        
        try:
            # Ensure tenant_id is set in scan record
            scan_record['tenant_id'] = tenant_id
            
            # Store in PostgreSQL via multi-database connector
            success = await multi_db.execute_operation(
                "write", "insert_scan_record", tenant_id, scan_record
            )
            
            if success:
                # Automatically cache the scan record
                await multi_db.execute_operation(
                    "cache", "cache_scan_results", tenant_id, scan_record
                )
                
                # Log audit event
                await self._log_audit_event(
                    action="scan_record_created",
                    resource_type="scan",
                    resource_id=scan_record.get('scan_id', 'unknown'),
                    tenant_id=tenant_id,
                    success=True
                )
                
                logger.info(f"✅ Scan record stored for tenant {tenant_id}")
                return True
            else:
                await self._log_audit_event(
                    action="scan_record_creation_failed",
                    resource_type="scan",
                    resource_id=scan_record.get('scan_id', 'unknown'),
                    tenant_id=tenant_id,
                    success=False,
                    error_message="Failed to store scan record"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Error storing scan record for tenant {tenant_id}: {e}")
            await self._log_security_event(
                event_type="database_operation_failed",
                description=f"Scan record storage failed: {e}",
                severity=EventSeverity.HIGH,
                tenant_id=tenant_id
            )
            return False
    
    async def get_scan_record(self, scan_id: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        🔍 Retrieve scan record with cache-aside pattern
        💡 Tries cache first, then database, then caches result
        """
        if not self.initialized:
            logger.error("❌ Database package not initialized")
            return None
        
        try:
            # Use multi-database connector with cache-aside pattern
            scan_record = await multi_db.execute_operation(
                "read", "get_scan_record", tenant_id, scan_id
            )
            
            if scan_record:
                logger.debug(f"✅ Scan record retrieved for tenant {tenant_id}")
                return scan_record
            else:
                logger.warning(f"⚠️ Scan record not found for tenant {tenant_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving scan record for tenant {tenant_id}: {e}")
            return None
    
    async def get_recent_scans(self, tenant_id: str = "default", limit: int = 50) -> List[Dict[str, Any]]:
        """
        📋 Get recent scans for a tenant with database fallback
        💡 Primarily uses database with optional caching for frequent access
        """
        if not self.initialized:
            logger.error("❌ Database package not initialized")
            return []
        
        try:
            scans = await multi_db.execute_operation(
                "read", "get_recent_scans", tenant_id, limit
            )
            
            logger.info(f"✅ Retrieved {len(scans)} recent scans for tenant {tenant_id}")
            return scans
            
        except Exception as e:
            logger.error(f"❌ Error retrieving recent scans for tenant {tenant_id}: {e}")
            return []
    
    async def store_threat_intelligence(self, threat_data: Dict[str, Any], tenant_id: str = "default") -> bool:
        """
        🛡️ Store threat intelligence with encryption and caching
        💡 Threat data is encrypted in cache for security
        """
        if not self.initialized:
            logger.error("❌ Database package not initialized")
            return False
        
        try:
            # Ensure tenant_id is set in threat data
            threat_data['tenant_id'] = tenant_id
            
            # Store in PostgreSQL
            success = await multi_db.execute_operation(
                "write", "insert_threat_intelligence", tenant_id, threat_data
            )
            
            if success:
                # Cache encrypted threat intelligence
                await multi_db.execute_operation(
                    "cache", "cache_threat_intelligence", tenant_id, threat_data
                )
                
                # Log audit event
                await self._log_audit_event(
                    action="threat_intelligence_created",
                    resource_type="threat",
                    resource_id=threat_data.get('threat_id', 'unknown'),
                    tenant_id=tenant_id,
                    success=True
                )
                
                logger.info(f"✅ Threat intelligence stored for tenant {tenant_id}")
                return True
            else:
                await self._log_audit_event(
                    action="threat_intelligence_creation_failed",
                    resource_type="threat",
                    resource_id=threat_data.get('threat_id', 'unknown'),
                    tenant_id=tenant_id,
                    success=False,
                    error_message="Failed to store threat intelligence"
                )
                return False
                
        except Exception as e:
            logger.error(f"❌ Error storing threat intelligence for tenant {tenant_id}: {e}")
            await self._log_security_event(
                event_type="database_operation_failed",
                description=f"Threat intelligence storage failed: {e}",
                severity=EventSeverity.HIGH,
                tenant_id=tenant_id
            )
            return False
    
    async def get_threat_intelligence(self, threat_id: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        🔍 Retrieve threat intelligence with cache-aside pattern
        💡 Handles encrypted threat data automatically
        """
        if not self.initialized:
            logger.error("❌ Database package not initialized")
            return None
        
        try:
            threat_data = await multi_db.execute_operation(
                "read", "get_threat_intelligence", tenant_id, threat_id
            )
            
            if threat_data:
                logger.debug(f"✅ Threat intelligence retrieved for tenant {tenant_id}")
                return threat_data
            else:
                logger.warning(f"⚠️ Threat intelligence not found for tenant {tenant_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving threat intelligence for tenant {tenant_id}: {e}")
            return None
    
    async def manage_user_session(self, user_id: str, session_data: Dict[str, Any], tenant_id: str = "default") -> bool:
        """
        👤 Manage user session with security and multi-tenant support
        💡 Sessions are encrypted and tenant-isolated
        """
        if not self.initialized:
            logger.error("❌ Database package not initialized")
            return False
        
        try:
            success = await multi_db.execute_operation(
                "cache", "cache_user_session", tenant_id, user_id, session_data
            )
            
            if success:
                logger.debug(f"✅ User session managed for tenant {tenant_id}")
                return True
            else:
                logger.warning(f"⚠️ User session management failed for tenant {tenant_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error managing user session for tenant {tenant_id}: {e}")
            return False
    
    async def get_user_session(self, user_id: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        🔍 Retrieve user session with automatic extension
        💡 Session TTL is extended on access based on security settings
        """
        if not self.initialized:
            logger.error("❌ Database package not initialized")
            return None
        
        try:
            session_data = await multi_db.execute_operation(
                "cache", "get_user_session", tenant_id, user_id
            )
            
            if session_data:
                logger.debug(f"✅ User session retrieved for tenant {tenant_id}")
                return session_data
            else:
                logger.debug(f"ℹ️ No active session found for user {user_id} in tenant {tenant_id}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving user session for tenant {tenant_id}: {e}")
            return None
    
    async def invalidate_user_session(self, user_id: str, tenant_id: str = "default") -> bool:
        """
        🚫 Invalidate user session immediately
        💡 Critical for security and logout functionality
        """
        if not self.initialized:
            logger.error("❌ Database package not initialized")
            return False
        
        try:
            success = await multi_db.execute_operation(
                "cache", "invalidate_user_session", tenant_id, user_id
            )
            
            if success:
                await self._log_audit_event(
                    action="user_session_invalidated",
                    resource_type="session",
                    resource_id=user_id,
                    tenant_id=tenant_id,
                    success=True
                )
                logger.info(f"✅ User session invalidated for tenant {tenant_id}")
                return True
            else:
                logger.warning(f"⚠️ User session invalidation failed for tenant {tenant_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error invalidating user session for tenant {tenant_id}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ❤️ Comprehensive health check for entire database package
        💡 Includes all components and tenant-specific health metrics
        """
        if not self.initialized:
            return {
                "status": "unhealthy",
                "message": "Database package not initialized",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        try:
            # Get health status from multi-database connector
            db_health = await multi_db.health_check()
            
            # Enhanced health status with package-specific metrics
            health_status = {
                "status": db_health.get("overall_status", "unknown"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "multi_db_connector": db_health,
                    "postgresql": db_health.get("databases", {}).get("postgresql", {}),
                    "redis": db_health.get("databases", {}).get("redis", {})
                },
                "package_metrics": {
                    "initialized": self.initialized,
                    "tenant_operations_count": len(self.tenant_operations),
                    "cache_performance": db_health.get("cache_hit_rate", 0)
                },
                "tenant_health": db_health.get("tenant_health", {})
            }
            
            self.health_status = health_status
            return health_status
            
        except Exception as e:
            error_status = {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.health_status = error_status
            return error_status
    
    async def invalidate_tenant_cache(self, tenant_id: str) -> bool:
        """
        🧹 Invalidate all cache entries for a specific tenant
        💡 Useful for tenant deletion or mass cache refresh
        """
        if not self.initialized:
            logger.error("❌ Database package not initialized")
            return False
        
        try:
            success = await redis_manager.invalidate_tenant_cache(tenant_id)
            
            if success:
                await self._log_audit_event(
                    action="tenant_cache_invalidated",
                    resource_type="cache",
                    resource_id=tenant_id,
                    tenant_id=tenant_id,
                    success=True
                )
                logger.info(f"✅ Tenant cache invalidated for {tenant_id}")
                return True
            else:
                logger.warning(f"⚠️ Tenant cache invalidation failed for {tenant_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error invalidating tenant cache for {tenant_id}: {e}")
            return False
    
    async def _log_audit_event(self, action: str, resource_type: str, resource_id: str, 
                             tenant_id: str, success: bool, error_message: str = ""):
        """Log audit events for database operations"""
        # This would integrate with the AuditLog system
        audit_data = {
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "tenant_id": tenant_id,
            "success": success,
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"📝 Audit Event: {audit_data}")
    
    async def _log_security_event(self, event_type: str, description: str, 
                                severity: EventSeverity, tenant_id: str):
        """Log security events for database operations"""
        # This would integrate with the SecurityEvent system
        security_data = {
            "event_type": event_type,
            "description": description,
            "severity": severity.value,
            "tenant_id": tenant_id,
            "detected_at": datetime.now(timezone.utc).isoformat()
        }
        logger.warning(f"🚨 Security Event: {security_data}")
    
    async def close(self):
        """🔚 Close all database connections and cleanup resources"""
        if self.initialized:
            await close_multi_database()
            self.initialized = False
            logger.info("✅ Database package closed successfully")


# Global database package instance
db_package = DatabasePackage()


async def initialize_database_package() -> bool:
    """🚀 Initialize the complete database package"""
    return await db_package.initialize()


async def close_database_package():
    """🔚 Close the complete database package"""
    await db_package.close()


# Export all components for backward compatibility and modular access
__all__ = [
    # Database Manager
    "DatabaseManager",
    "database_manager", 
    "initialize_database",
    "close_database",
    
    # Redis Manager
    "RedisManager",
    "redis_manager",
    "initialize_redis", 
    "close_redis",
    
    # Multi-Database Connector
    "MultiDBConnector",
    "multi_db",
    "initialize_multi_database",
    "close_multi_database",
    
    # Database Package (New)
    "DatabasePackage",
    "db_package",
    "initialize_database_package", 
    "close_database_package"
]

__version__ = "2.0.0"


# Auto-initialize logging
logger.info(f"🎯 Database Module Package v{__version__} loaded successfully")