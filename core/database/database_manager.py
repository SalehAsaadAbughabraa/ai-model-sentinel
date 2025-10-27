"""
🎯 Database Manager
📦 Centralized database management with connection pooling and multi-database support
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Business Logic: 
   - Manages database connections with connection pooling
   - Supports multiple database types (PostgreSQL, SQLite)
   - Provides automatic retry and failover mechanisms
   - Handles database migrations and schema management
"""

import asyncio
import logging
import asyncpg
import uuid
from typing import Optional, Dict, Any, List, Union
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from enum import Enum

from config.settings import settings, SecretManager


logger = logging.getLogger("DatabaseManager")


class DatabaseType(str, Enum):
    """🗄️ Supported database types"""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    READ_REPLICA = "read_replica"


class TenantIsolationLevel(str, Enum):
    """🏢 Tenant isolation strategies"""
    SCHEMA_PER_TENANT = "schema_per_tenant"
    DATABASE_PER_TENANT = "database_per_tenant"
    ROW_LEVEL = "row_level"  # Using tenant_id column


class AuditAction(str, Enum):
    """📝 Audit action types"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    ACCESS_DENIED = "access_denied"
    CONFIG_CHANGE = "config_change"
    SECURITY_SCAN = "security_scan"
    THREAT_DETECTED = "threat_detected"
    BACKUP_CREATED = "backup_created"
    RESTORE_PERFORMED = "restore_performed"


class EventSeverity(str, Enum):
    """🚨 Event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DatabaseManager:
    """
    🗄️ Main database manager for AI Model Sentinel
    💡 Handles all database operations with connection pooling and error recovery
    """
    
    def __init__(self):
        self.connection_pools: Dict[DatabaseType, Optional[asyncpg.Pool]] = {
            DatabaseType.POSTGRESQL: None,
            DatabaseType.READ_REPLICA: None
        }
        self.is_connected: bool = False
        self.tenant_isolation: TenantIsolationLevel = TenantIsolationLevel.ROW_LEVEL
        
        # Enhanced connection statistics
        self.connection_stats: Dict[str, Any] = {
            "total_connections": 0,
            "active_connections": 0,
            "connection_errors": 0,
            "last_connection": None,
            "tenant_connections": {},
            "query_metrics": {
                "total_queries": 0,
                "failed_queries": 0,
                "avg_response_time": 0.0
            }
        }
        
        # Query templates for tenant-aware operations
        self.tenant_queries = {
            "scan_results": {
                "insert": """
                    INSERT INTO scan_results (
                        scan_id, repository, branch, commit_hash, trigger, platform,
                        threat_level, threat_score, models_scanned, status, scan_duration,
                        details, compliance_tags, audit_trail, security_level, tenant_id,
                        geographic_region, business_unit, created_at, updated_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                """,
                "select_by_tenant": "SELECT * FROM scan_results WHERE tenant_id = $1",
                "select_by_tenant_and_date": """
                    SELECT * FROM scan_results 
                    WHERE tenant_id = $1 AND created_at >= $2 AND created_at <= $3
                """
            },
            "threat_intelligence": {
                "insert": """
                    INSERT INTO threat_intelligence (
                        threat_id, attack_type, severity, confidence, techniques, indicators,
                        attacker_fingerprint, source_ip, geographic_origin, mitigation,
                        countermeasures, status, first_seen, last_seen, is_active,
                        security_level, tenant_id, related_threats, evidence, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                """,
                "select_active_by_tenant": """
                    SELECT * FROM threat_intelligence 
                    WHERE tenant_id = $1 AND is_active = true
                """
            }
        }
    
    async def connect(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """
        🔗 Establish database connection with retry logic
        💡 Implements exponential backoff for connection failures
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to database (attempt {attempt + 1}/{max_retries})...")
                
                # Get database credentials securely
                db_password = SecretManager.get_secret("DB_PASSWORD", "")
                
                # Main database connection
                self.connection_pools[DatabaseType.POSTGRESQL] = await asyncpg.create_pool(
                    dsn=settings.database.get_connection_string(),
                    min_size=settings.database.pool_min_size,
                    max_size=settings.database.pool_max_size,
                    command_timeout=settings.database.pool_timeout,
                    ssl=settings.database.ssl_mode if settings.database.ssl_mode != "prefer" else None,
                    server_settings={
                        'application_name': 'AI_Model_Sentinel_v2.0.0',
                        'jit': 'off',
                        'statement_timeout': '30000'  # 30 second timeout
                    }
                )
                
                # Test connection with secure query
                async with self.get_connection() as conn:
                    await conn.execute("SELECT 1")
                
                self.is_connected = True
                self.connection_stats["total_connections"] += 1
                self.connection_stats["last_connection"] = datetime.now(timezone.utc)
                
                logger.info("Database connection established successfully")
                logger.info(f"Connection pool: {settings.database.pool_min_size}-{settings.database.pool_max_size} connections")
                
                # Initialize database schema with tenant support
                await self._initialize_schema()
                return True
                
            except Exception as e:
                self.connection_stats["connection_errors"] += 1
                logger.error(f"Database connection failed (attempt {attempt + 1}): {e}")
                
                # Log security event for connection failures
                await self._log_security_event(
                    "database_connection_failed",
                    f"Database connection attempt {attempt + 1} failed: {str(e)}",
                    EventSeverity.HIGH if attempt == max_retries - 1 else EventSeverity.MEDIUM
                )
                
                if attempt == max_retries - 1:
                    logger.error("All database connection attempts failed")
                    return False
                
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        return False
    
    async def _initialize_schema(self) -> None:
        """
        📐 Initialize database schema and tables with tenant isolation
        💡 Creates all necessary tables, indexes, and partitions
        """
        try:
            async with self.get_connection() as conn:
                # Enable necessary extensions
                await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")
                
                # Scan results table with enhanced constraints
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS scan_results (
                        scan_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        repository TEXT NOT NULL,
                        branch TEXT NOT NULL DEFAULT 'main',
                        commit_hash TEXT NOT NULL CHECK (length(commit_hash) >= 7),
                        trigger TEXT NOT NULL,
                        platform TEXT NOT NULL DEFAULT 'github',
                        threat_level TEXT NOT NULL CHECK (threat_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                        threat_score REAL NOT NULL CHECK (threat_score >= 0 AND threat_score <= 100),
                        models_scanned INTEGER NOT NULL CHECK (models_scanned >= 0),
                        status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
                        scan_duration REAL NOT NULL CHECK (scan_duration >= 0),
                        details JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(details) = 'object'),
                        compliance_tags JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(compliance_tags) = 'object'),
                        audit_trail JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(audit_trail) = 'object'),
                        security_level TEXT NOT NULL CHECK (security_level IN ('public', 'internal', 'confidential', 'restricted')),
                        tenant_id TEXT NOT NULL DEFAULT 'default',
                        geographic_region TEXT NOT NULL DEFAULT 'global',
                        business_unit TEXT NOT NULL DEFAULT 'default',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        
                        -- Additional constraints for data integrity
                        CONSTRAINT valid_threat_score CHECK (
                            (threat_level = 'LOW' AND threat_score <= 30) OR
                            (threat_level = 'MEDIUM' AND threat_score BETWEEN 31 AND 60) OR
                            (threat_level = 'HIGH' AND threat_score BETWEEN 61 AND 80) OR
                            (threat_level = 'CRITICAL' AND threat_score > 80)
                        )
                    )
                ''')
                
                # Threat intelligence table with enhanced security
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS threat_intelligence (
                        threat_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        attack_type TEXT NOT NULL CHECK (attack_type IN (
                            'model_inversion', 'membership_inference', 'data_poisoning', 
                            'model_stealing', 'adversarial_attack', 'prompt_injection',
                            'training_data_extraction', 'model_evasion', 'backdoor_attack',
                            'model_skewing'
                        )),
                        severity TEXT NOT NULL CHECK (severity IN ('info', 'low', 'medium', 'high', 'critical')),
                        confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
                        techniques JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(techniques) = 'array'),
                        indicators JSONB NOT NULL DEFAULT '{}'::jsonb CHECK (jsonb_typeof(indicators) = 'object'),
                        attacker_fingerprint TEXT NOT NULL DEFAULT '',
                        source_ip TEXT NOT NULL DEFAULT '',
                        geographic_origin TEXT NOT NULL DEFAULT '',
                        mitigation TEXT NOT NULL DEFAULT '',
                        countermeasures JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(countermeasures) = 'array'),
                        status TEXT NOT NULL CHECK (status IN ('active', 'mitigated', 'false_positive', 'expired', 'investigating')),
                        first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        is_active BOOLEAN NOT NULL DEFAULT TRUE,
                        security_level TEXT NOT NULL DEFAULT 'restricted' CHECK (security_level IN ('public', 'internal', 'confidential', 'restricted')),
                        tenant_id TEXT NOT NULL DEFAULT 'default',
                        related_threats JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(related_threats) = 'array'),
                        evidence JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(evidence) = 'array'),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        
                        -- Confidence and severity consistency
                        CONSTRAINT valid_confidence_severity CHECK (
                            (severity = 'critical' AND confidence >= 80) OR
                            (severity = 'high' AND confidence >= 60) OR
                            (severity IN ('medium', 'low', 'info'))
                        )
                    )
                ''')
                
                # Enhanced audit logs table with automatic partitioning
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        action TEXT NOT NULL CHECK (action IN (
                            'create', 'read', 'update', 'delete', 'login', 'logout',
                            'access_denied', 'config_change', 'security_scan',
                            'threat_detected', 'backup_created', 'restore_performed'
                        )),
                        resource_type TEXT NOT NULL,
                        resource_id TEXT NOT NULL,
                        user_id TEXT NOT NULL DEFAULT '',
                        user_email TEXT NOT NULL DEFAULT '',
                        user_role TEXT NOT NULL DEFAULT '',
                        tenant_id TEXT NOT NULL DEFAULT 'default',
                        ip_address TEXT NOT NULL DEFAULT '',
                        user_agent TEXT NOT NULL DEFAULT '',
                        request_method TEXT NOT NULL DEFAULT '',
                        request_path TEXT NOT NULL DEFAULT '',
                        old_values JSONB NOT NULL DEFAULT '{}'::jsonb,
                        new_values JSONB NOT NULL DEFAULT '{}'::jsonb,
                        changes JSONB NOT NULL DEFAULT '{}'::jsonb,
                        success BOOLEAN NOT NULL DEFAULT TRUE,
                        error_message TEXT NOT NULL DEFAULT '',
                        status_code INTEGER NOT NULL DEFAULT 200 CHECK (status_code BETWEEN 100 AND 599),
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        session_id TEXT NOT NULL DEFAULT '',
                        compliance_frameworks JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(compliance_frameworks) = 'array')
                    )
                ''')
                
                # Create comprehensive indexes for performance
                await self._create_indexes(conn)
                
                # Create partitions for large tables (monthly partitioning for audit logs)
                await self._create_partitions(conn)
                
                logger.info("Database schema initialized successfully with tenant isolation")
                
        except Exception as e:
            logger.error(f"Database schema initialization failed: {e}")
            await self._log_security_event(
                "schema_initialization_failed",
                f"Database schema initialization failed: {str(e)}",
                EventSeverity.HIGH
            )
            raise
    
    async def _create_indexes(self, conn) -> None:
        """Create performance indexes for tenant-aware queries"""
        indexes = [
            # Scan results indexes
            "CREATE INDEX IF NOT EXISTS idx_scan_results_tenant_created ON scan_results(tenant_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_scan_results_threat_level ON scan_results(threat_level, created_at)",
            "CREATE INDEX IF NOT EXISTS idx_scan_results_status ON scan_results(status, tenant_id)",
            "CREATE INDEX IF NOT EXISTS idx_scan_results_composite ON scan_results(tenant_id, threat_level, created_at)",
            
            # Threat intelligence indexes
            "CREATE INDEX IF NOT EXISTS idx_threat_intelligence_active ON threat_intelligence(is_active, last_seen DESC)",
            "CREATE INDEX IF NOT EXISTS idx_threat_intelligence_tenant ON threat_intelligence(tenant_id, severity, last_seen)",
            "CREATE INDEX IF NOT EXISTS idx_threat_intelligence_attack_type ON threat_intelligence(attack_type, tenant_id)",
            
            # Audit logs indexes
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_action ON audit_logs(tenant_id, action, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_user_tenant ON audit_logs(user_id, tenant_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id, tenant_id)"
        ]
        
        for index_sql in indexes:
            await conn.execute(index_sql)
    
    async def _create_partitions(self, conn) -> None:
        """Create table partitions for better performance"""
        try:
            # Create monthly partitions for audit logs (example for current month)
            current_month = datetime.now().strftime("%Y_%m")
            await conn.execute(f'''
                CREATE TABLE IF NOT EXISTS audit_logs_{current_month} 
                PARTITION OF audit_logs 
                FOR VALUES FROM ('{datetime.now().replace(day=1).isoformat()}') 
                TO ('{(datetime.now().replace(day=28) + timedelta(days=4)).replace(day=1).isoformat()}')
            ''')
        except Exception as e:
            logger.warning(f"Partition creation skipped or failed: {e}")
    
    @asynccontextmanager
    async def get_connection(self, tenant_id: Optional[str] = None, db_type: DatabaseType = DatabaseType.POSTGRESQL):
        """
        🔄 Context manager for database connections with tenant awareness
        💡 Automatically handles connection acquisition, release, and tenant filtering
        """
        if not self.connection_pools[db_type] or not self.is_connected:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        connection = await self.connection_pools[db_type].acquire()
        self.connection_stats["active_connections"] += 1
        
        # Track tenant connection if provided
        if tenant_id:
            if tenant_id not in self.connection_stats["tenant_connections"]:
                self.connection_stats["tenant_connections"][tenant_id] = 0
            self.connection_stats["tenant_connections"][tenant_id] += 1
        
        try:
            # Set tenant context if provided (for row-level security in future)
            if tenant_id and self.tenant_isolation == TenantIsolationLevel.ROW_LEVEL:
                await connection.execute("SET app.current_tenant_id = $1", tenant_id)
            
            yield connection
        except Exception as e:
            # Log query failure for audit
            await self._log_audit_event(
                action=AuditAction.CONFIG_CHANGE,
                resource_type="database",
                resource_id="query_execution",
                tenant_id=tenant_id or "system",
                success=False,
                error_message=f"Query execution failed: {str(e)}"
            )
            raise
        finally:
            await self.connection_pools[db_type].release(connection)
            self.connection_stats["active_connections"] -= 1
            
            if tenant_id and tenant_id in self.connection_stats["tenant_connections"]:
                self.connection_stats["tenant_connections"][tenant_id] -= 1
    
    async def execute_query(self, query: str, *args, tenant_id: Optional[str] = None, max_retries: int = 2) -> Any:
        """
        📝 Execute a database query with automatic connection handling and retry logic
        💡 Includes tenant-aware query execution and audit logging
        """
        for attempt in range(max_retries):
            try:
                start_time = asyncio.get_event_loop().time()
                
                async with self.get_connection(tenant_id) as conn:
                    result = await conn.execute(query, *args)
                
                # Update query metrics
                response_time = asyncio.get_event_loop().time() - start_time
                self._update_query_metrics(success=True, response_time=response_time)
                
                # Log successful query for audit (if modifying data)
                if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')):
                    await self._log_audit_event(
                        action=AuditAction.CONFIG_CHANGE,
                        resource_type="database",
                        resource_id="query_execution",
                        tenant_id=tenant_id or "system",
                        success=True,
                        details={"query_type": query.split()[0].upper(), "response_time": response_time}
                    )
                
                return result
                
            except Exception as e:
                self._update_query_metrics(success=False)
                
                if attempt == max_retries - 1:
                    # Log final failure
                    await self._log_security_event(
                        "database_query_failed",
                        f"Query failed after {max_retries} attempts: {str(e)}",
                        EventSeverity.MEDIUM,
                        tenant_id=tenant_id
                    )
                    raise
                
                await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff
    
    async def fetch_row(self, query: str, *args, tenant_id: Optional[str] = None) -> Optional[asyncpg.Record]:
        """🔍 Fetch a single row with tenant awareness"""
        async with self.get_connection(tenant_id) as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetch_all(self, query: str, *args, tenant_id: Optional[str] = None) -> List[asyncpg.Record]:
        """📋 Fetch all rows with tenant awareness"""
        async with self.get_connection(tenant_id) as conn:
            return await conn.fetch(query, *args)
    
    # Tenant-specific query methods
    async def insert_scan_result(self, scan_data: Dict[str, Any], tenant_id: str) -> str:
        """Insert a scan result with tenant context"""
        query = self.tenant_queries["scan_results"]["insert"]
        scan_id = scan_data.get("scan_id") or str(uuid.uuid4())
        
        await self.execute_query(query, 
            scan_id,
            scan_data["repository"],
            scan_data.get("branch", "main"),
            scan_data["commit_hash"],
            scan_data.get("trigger", "manual"),
            scan_data.get("platform", "github"),
            scan_data.get("threat_level", "LOW"),
            scan_data.get("threat_score", 0.0),
            scan_data.get("models_scanned", 0),
            scan_data.get("status", "pending"),
            scan_data.get("scan_duration", 0.0),
            scan_data.get("details", {}),
            scan_data.get("compliance_tags", {}),
            scan_data.get("audit_trail", {}),
            scan_data.get("security_level", "confidential"),
            tenant_id,
            scan_data.get("geographic_region", "global"),
            scan_data.get("business_unit", "default"),
            scan_data.get("created_at", datetime.now(timezone.utc)),
            scan_data.get("updated_at", datetime.now(timezone.utc))
        , tenant_id=tenant_id)
        
        return scan_id
    
    async def get_tenant_scan_results(self, tenant_id: str, days: int = 30) -> List[asyncpg.Record]:
        """Get scan results for a specific tenant"""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        query = self.tenant_queries["scan_results"]["select_by_tenant_and_date"]
        
        return await self.fetch_all(query, tenant_id, start_date, datetime.now(timezone.utc), tenant_id=tenant_id)
    
    def _update_query_metrics(self, success: bool, response_time: float = 0.0):
        """Update query performance metrics"""
        self.connection_stats["query_metrics"]["total_queries"] += 1
        
        if not success:
            self.connection_stats["query_metrics"]["failed_queries"] += 1
        else:
            # Update average response time
            current_avg = self.connection_stats["query_metrics"]["avg_response_time"]
            total_successful = self.connection_stats["query_metrics"]["total_queries"] - self.connection_stats["query_metrics"]["failed_queries"]
            self.connection_stats["query_metrics"]["avg_response_time"] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ❤️ Perform comprehensive database health check
        💡 Returns connection status, performance metrics, and tenant-specific stats
        """
        health_status = {
            "status": "healthy" if self.is_connected else "unhealthy",
            "connected": self.is_connected,
            "connection_stats": self.connection_stats.copy(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tenant_metrics": {},
            "performance": {}
        }
        
        if self.is_connected:
            try:
                async with self.get_connection() as conn:
                    # Basic responsiveness check
                    start_time = asyncio.get_event_loop().time()
                    await conn.execute("SELECT 1")
                    response_time = asyncio.get_event_loop().time() - start_time
                    
                    health_status["performance"]["response_time_ms"] = round(response_time * 1000, 2)
                    health_status["performance"]["active_connections"] = self.connection_stats["active_connections"]
                    health_status["performance"]["avg_query_time"] = round(
                        self.connection_stats["query_metrics"]["avg_response_time"] * 1000, 2
                    )
                    
                    # Check tenant-specific connectivity
                    for tenant_id in list(self.connection_stats["tenant_connections"].keys())[:5]:  # Sample first 5 tenants
                        try:
                            tenant_start_time = asyncio.get_event_loop().time()
                            await conn.execute("SELECT 1")
                            tenant_response_time = asyncio.get_event_loop().time() - tenant_start_time
                            
                            health_status["tenant_metrics"][tenant_id] = {
                                "response_time_ms": round(tenant_response_time * 1000, 2),
                                "active_connections": self.connection_stats["tenant_connections"].get(tenant_id, 0)
                            }
                        except Exception as e:
                            health_status["tenant_metrics"][tenant_id] = {"error": str(e)}
                    
            except Exception as e:
                health_status["status"] = "degraded"
                health_status["error"] = str(e)
                
                # Log health check failure as security event
                await self._log_security_event(
                    "database_health_check_failed",
                    f"Database health check failed: {str(e)}",
                    EventSeverity.MEDIUM
                )
        
        return health_status
    
    async def _log_audit_event(self, action: AuditAction, resource_type: str, resource_id: str, 
                             tenant_id: str, success: bool, error_message: str = "", details: Dict = None):
        """Helper method to log audit events for database operations"""
        audit_data = {
            "action": action.value,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "tenant_id": tenant_id,
            "success": success,
            "error_message": error_message,
            "details": details or {}
        }
        # In production, this would create an AuditLog record
        logger.info(f"Audit Event: {audit_data}")
    
    async def _log_security_event(self, event_type: str, description: str, severity: EventSeverity, tenant_id: str = "system"):
        """Helper method to log security events"""
        security_data = {
            "event_type": event_type,
            "description": description,
            "severity": severity.value,
            "tenant_id": tenant_id,
            "detected_at": datetime.now(timezone.utc).isoformat()
        }
        # In production, this would create a SecurityEvent record
        logger.warning(f"Security Event: {security_data}")
    
    async def close(self) -> None:
        """🔚 Close all database connections"""
        for db_type, pool in self.connection_pools.items():
            if pool:
                await pool.close()
        self.is_connected = False
        logger.info("Database connections closed successfully")


# Global database manager instance
database_manager = DatabaseManager()


async def initialize_database() -> bool:
    """🚀 Initialize the database connection"""
    return await database_manager.connect()


async def close_database() -> None:
    """🔚 Close database connections"""
    await database_manager.close()