"""
🎯 Multi-Database Connector
📦 Unified interface for multiple database systems with failover support
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Business Logic: 
   - Provides unified API for PostgreSQL, Redis, and future databases
   - Implements automatic failover and retry mechanisms
   - Supports read/write separation and connection pooling
   - Enables seamless database switching and migration
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from enum import Enum

from .database_manager import database_manager, DatabaseManager
from .redis_manager import redis_manager, RedisManager
from config.settings import settings, get_settings


logger = logging.getLogger("MultiDBConnector")


class DatabaseType(str, Enum):
    """🗄️ Supported database types in the multi-database system"""
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"


class OperationType(str, Enum):
    """⚡ Database operation types for routing and optimization"""
    READ = "read"
    WRITE = "write"
    CACHE = "cache"
    ANALYTICS = "analytics"


class MultiDBConnector:
    """
    🔄 Unified database connector for AI Model Sentinel
    💡 Provides seamless access to multiple database systems with intelligent routing
    """
    
    def __init__(self):
        self.databases: Dict[DatabaseType, Any] = {}
        self.operation_stats: Dict[str, Any] = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "database_failovers": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_operation": None
        }
        
        # Operation routing configuration with tenant awareness
        self.operation_routing = {
            OperationType.READ: [DatabaseType.REDIS, DatabaseType.POSTGRESQL],  # Try cache first
            OperationType.WRITE: [DatabaseType.POSTGRESQL, DatabaseType.REDIS],  # Write to both
            OperationType.CACHE: [DatabaseType.REDIS],
            OperationType.ANALYTICS: [DatabaseType.SNOWFLAKE, DatabaseType.BIGQUERY]
        }
        
        # Tenant-aware operation configuration
        self.tenant_operations: Dict[str, Dict[OperationType, List[DatabaseType]]] = {}
    
    async def initialize(self) -> bool:
        """
        🚀 Initialize all database connections with system settings integration
        💡 Establishes connections to all configured databases with tenant awareness
        """
        logger.info("🔄 Initializing multi-database connector...")
        
        initialization_results = {}
        
        # Initialize PostgreSQL with settings integration
        try:
            postgres_initialized = await database_manager.connect()
            if postgres_initialized:
                self.databases[DatabaseType.POSTGRESQL] = database_manager
                initialization_results[DatabaseType.POSTGRESQL] = "✅ Connected"
                
                # Configure PostgreSQL with system settings
                await self._configure_postgresql_with_settings()
            else:
                initialization_results[DatabaseType.POSTGRESQL] = "❌ Failed"
        except Exception as e:
            initialization_results[DatabaseType.POSTGRESQL] = f"❌ Error: {e}"
        
        # Initialize Redis with settings integration
        try:
            redis_initialized = await redis_manager.connect()
            if redis_initialized:
                self.databases[DatabaseType.REDIS] = redis_manager
                initialization_results[DatabaseType.REDIS] = "✅ Connected"
                
                # Configure Redis with system settings
                await self._configure_redis_with_settings()
            else:
                initialization_results[DatabaseType.REDIS] = "⚠️ Disabled"
        except Exception as e:
            initialization_results[DatabaseType.REDIS] = f"❌ Error: {e}"
        
        # Log initialization results
        for db_type, status in initialization_results.items():
            logger.info(f"   {db_type.value}: {status}")
        
        # Check if we have at least one operational database
        operational_dbs = [db for db, status in initialization_results.items() if "✅" in status]
        
        if operational_dbs:
            logger.info(f"✅ Multi-database connector initialized with {len(operational_dbs)} databases")
            
            # Initialize tenant-specific routing
            await self._initialize_tenant_routing()
            return True
        else:
            logger.error("💥 No databases could be initialized")
            return False
    
    async def _configure_postgresql_with_settings(self):
        """Configure PostgreSQL with system settings"""
        # Database configuration is already handled in DatabaseManager
        # This method can be extended for additional PostgreSQL-specific settings
        logger.info("⚙️ PostgreSQL configured with system settings")
    
    async def _configure_redis_with_settings(self):
        """Configure Redis with system settings"""
        # Redis configuration is already handled in RedisManager
        # This method can be extended for additional Redis-specific settings
        logger.info("⚙️ Redis configured with system settings")
    
    async def _initialize_tenant_routing(self):
        """Initialize tenant-specific operation routing"""
        # Load tenant-specific configurations from settings
        for tenant_id, tenant_config in settings.tenant_configs.items():
            self.tenant_operations[tenant_id] = {
                OperationType.READ: [DatabaseType.REDIS, DatabaseType.POSTGRESQL],
                OperationType.WRITE: [DatabaseType.POSTGRESQL, DatabaseType.REDIS],
                OperationType.CACHE: [DatabaseType.REDIS],
                OperationType.ANALYTICS: self._get_tenant_analytics_databases(tenant_id)
            }
    
    def _get_tenant_analytics_databases(self, tenant_id: str) -> List[DatabaseType]:
        """Get analytics databases for specific tenant"""
        tenant_config = settings.get_tenant_config(tenant_id)
        
        if settings.is_feature_enabled("big_data_analytics", tenant_id):
            if settings.analytics.provider == "snowflake":
                return [DatabaseType.SNOWFLAKE]
            elif settings.analytics.provider == "bigquery":
                return [DatabaseType.BIGQUERY]
        
        return [DatabaseType.POSTGRESQL]  # Fallback to PostgreSQL
    
    async def execute_operation(
        self, 
        operation_type: OperationType,
        operation: str,
        tenant_id: str = "default",
        *args,
        **kwargs
    ) -> Any:
        """
        ⚡ Execute database operation with intelligent routing and tenant awareness
        💡 Automatically routes operations to appropriate databases with cache-aside pattern
        """
        self.operation_stats["total_operations"] += 1
        self.operation_stats["last_operation"] = datetime.now(timezone.utc).isoformat()
        
        # Get tenant-specific database routing
        preferred_databases = self._get_tenant_databases(tenant_id, operation_type)
        
        # Execute operation with failover support
        result = await self._execute_with_failover(
            operation_type, operation, tenant_id, preferred_databases, *args, **kwargs
        )
        
        if result is not None:
            self.operation_stats["successful_operations"] += 1
        else:
            self.operation_stats["failed_operations"] += 1
        
        return result
    
    def _get_tenant_databases(self, tenant_id: str, operation_type: OperationType) -> List[DatabaseType]:
        """Get database routing for specific tenant and operation"""
        if tenant_id in self.tenant_operations:
            return self.tenant_operations[tenant_id].get(operation_type, [])
        
        # Fallback to global routing
        return self.operation_routing.get(operation_type, [DatabaseType.POSTGRESQL])
    
    async def _execute_with_failover(
        self,
        operation_type: OperationType,
        operation: str,
        tenant_id: str,
        preferred_databases: List[DatabaseType],
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with failover support and cache-aside pattern"""
        
        # Special handling for cache-aside read operations
        if operation_type == OperationType.READ and operation.startswith("get_"):
            return await self._execute_cache_aside_read(operation, tenant_id, preferred_databases, *args, **kwargs)
        
        # Execute on preferred databases
        for db_type in preferred_databases:
            if db_type not in self.databases:
                continue
            
            try:
                database = self.databases[db_type]
                result = await self._execute_on_database(database, db_type, operation, tenant_id, *args, **kwargs)
                
                if result is not None:
                    # For write operations, update cache
                    if operation_type == OperationType.WRITE and db_type == DatabaseType.POSTGRESQL:
                        await self._sync_cache_after_write(operation, tenant_id, result, *args, **kwargs)
                    
                    return result
                    
            except Exception as e:
                logger.warning(f"⚠️ Operation failed on {db_type.value}: {e}")
                continue
        
        # Failover to any available database
        return await self._execute_failover(operation_type, operation, tenant_id, preferred_databases, *args, **kwargs)
    
    async def _execute_cache_aside_read(
        self,
        operation: str,
        tenant_id: str,
        preferred_databases: List[DatabaseType],
        *args,
        **kwargs
    ) -> Any:
        """Execute cache-aside pattern for read operations"""
        
        # Try cache first if available
        if DatabaseType.REDIS in self.databases and DatabaseType.REDIS in preferred_databases:
            try:
                cached_result = await self._execute_on_database(
                    self.databases[DatabaseType.REDIS], DatabaseType.REDIS, operation, tenant_id, *args, **kwargs
                )
                
                if cached_result is not None:
                    self.operation_stats["cache_hits"] += 1
                    return cached_result
                
                self.operation_stats["cache_misses"] += 1
            except Exception as e:
                logger.warning(f"⚠️ Cache read failed: {e}")
        
        # Try database for cache miss
        for db_type in preferred_databases:
            if db_type != DatabaseType.REDIS and db_type in self.databases:
                try:
                    database = self.databases[db_type]
                    result = await self._execute_on_database(database, db_type, operation, tenant_id, *args, **kwargs)
                    
                    if result is not None:
                        # Cache the result for future requests
                        await self._cache_read_result(operation, tenant_id, result, *args, **kwargs)
                        return result
                        
                except Exception as e:
                    logger.warning(f"⚠️ Database read failed on {db_type.value}: {e}")
                    continue
        
        return None
    
    async def _execute_failover(
        self,
        operation_type: OperationType,
        operation: str,
        tenant_id: str,
        preferred_databases: List[DatabaseType],
        *args,
        **kwargs
    ) -> Any:
        """Execute failover to any available database"""
        for db_type, database in self.databases.items():
            if db_type not in preferred_databases:
                try:
                    result = await self._execute_on_database(database, db_type, operation, tenant_id, *args, **kwargs)
                    
                    if result is not None:
                        self.operation_stats["database_failovers"] += 1
                        logger.info(f"🔄 Failover to {db_type.value} succeeded for tenant {tenant_id}")
                        return result
                        
                except Exception as e:
                    logger.warning(f"⚠️ Failover to {db_type.value} failed: {e}")
                    continue
        
        # All databases failed
        logger.error(f"💥 Operation '{operation}' failed on all databases for tenant {tenant_id}")
        return None
    
    async def _execute_on_database(
        self, 
        database: Any,
        db_type: DatabaseType,
        operation: str,
        tenant_id: str,
        *args,
        **kwargs
    ) -> Any:
        """
        🔧 Execute specific operation on a database with tenant context
        💡 Handles database-specific operation mapping with tenant awareness
        """
        try:
            if db_type == DatabaseType.POSTGRESQL:
                return await self._execute_postgresql_operation(database, operation, tenant_id, *args, **kwargs)
            elif db_type == DatabaseType.REDIS:
                return await self._execute_redis_operation(database, operation, tenant_id, *args, **kwargs)
            else:
                logger.warning(f"⚠️ Unsupported database type: {db_type}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Database operation failed on {db_type.value} for tenant {tenant_id}: {e}")
            raise
    
    async def _execute_postgresql_operation(
        self, 
        database: DatabaseManager,
        operation: str,
        tenant_id: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute PostgreSQL-specific operations with tenant context"""
        if operation == "insert_scan_record":
            return await self._insert_scan_record(database, tenant_id, *args, **kwargs)
        elif operation == "get_scan_record":
            return await self._get_scan_record(database, tenant_id, *args, **kwargs)
        elif operation == "get_recent_scans":
            return await self._get_recent_scans(database, tenant_id, *args, **kwargs)
        elif operation == "insert_threat_intelligence":
            return await self._insert_threat_intelligence(database, tenant_id, *args, **kwargs)
        elif operation == "get_threat_intelligence":
            return await self._get_threat_intelligence(database, tenant_id, *args, **kwargs)
        else:
            raise ValueError(f"Unknown PostgreSQL operation: {operation}")
    
    async def _execute_redis_operation(
        self, 
        database: RedisManager,
        operation: str,
        tenant_id: str,
        *args,
        **kwargs
    ) -> Any:
        """Execute Redis-specific operations with tenant context"""
        if operation == "cache_scan_results":
            return await database.cache_scan_results(*args, **kwargs)
        elif operation == "get_cached_scan":
            return await database.get_cached_scan(tenant_id, *args, **kwargs)
        elif operation == "cache_threat_intelligence":
            return await database.cache_threat_intelligence(*args, **kwargs)
        elif operation == "get_cached_threat":
            return await database.get_cached_threat(tenant_id, *args, **kwargs)
        elif operation == "cache_user_session":
            return await database.cache_user_session(*args, **kwargs)
        elif operation == "get_user_session":
            return await database.get_user_session(*args, **kwargs)
        else:
            raise ValueError(f"Unknown Redis operation: {operation}")
    
    # Cache synchronization methods
    async def _sync_cache_after_write(self, operation: str, tenant_id: str, result: Any, *args, **kwargs):
        """Sync cache after write operations"""
        try:
            if operation == "insert_scan_record" and result and DatabaseType.REDIS in self.databases:
                scan_data = args[0] if args else kwargs.get('scan_data')
                if scan_data:
                    await self.databases[DatabaseType.REDIS].cache_scan_results(scan_data)
            
            elif operation == "insert_threat_intelligence" and result and DatabaseType.REDIS in self.databases:
                threat_data = args[0] if args else kwargs.get('threat_data')
                if threat_data:
                    await self.databases[DatabaseType.REDIS].cache_threat_intelligence(threat_data)
                    
        except Exception as e:
            logger.warning(f"⚠️ Cache sync failed after {operation}: {e}")
    
    async def _cache_read_result(self, operation: str, tenant_id: str, result: Any, *args, **kwargs):
        """Cache read results for future requests"""
        try:
            if operation == "get_scan_record" and result and DatabaseType.REDIS in self.databases:
                await self.databases[DatabaseType.REDIS].cache_scan_results(result)
            
            elif operation == "get_threat_intelligence" and result and DatabaseType.REDIS in self.databases:
                await self.databases[DatabaseType.REDIS].cache_threat_intelligence(result)
                
        except Exception as e:
            logger.warning(f"⚠️ Read result caching failed for {operation}: {e}")
    
    # PostgreSQL Operation Implementations with Tenant Context
    async def _insert_scan_record(self, database: DatabaseManager, tenant_id: str, scan_data: Dict[str, Any]) -> bool:
        """Insert scan record into PostgreSQL with tenant context"""
        # Ensure tenant_id is set in scan_data
        scan_data['tenant_id'] = tenant_id
        
        query = """
            INSERT INTO scan_results (
                scan_id, repository, branch, commit_hash, trigger, platform,
                threat_level, threat_score, models_scanned, status, scan_duration,
                details, compliance_tags, audit_trail, security_level, tenant_id,
                geographic_region, business_unit
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
        """
        
        await database.execute_query(
            query,
            scan_data.get('scan_id'),
            scan_data.get('repository'),
            scan_data.get('branch'),
            scan_data.get('commit_hash'),
            scan_data.get('trigger'),
            scan_data.get('platform'),
            scan_data.get('threat_level'),
            scan_data.get('threat_score'),
            scan_data.get('models_scanned'),
            scan_data.get('status'),
            scan_data.get('scan_duration'),
            scan_data.get('details', {}),
            scan_data.get('compliance_tags', {}),
            scan_data.get('audit_trail', {}),
            scan_data.get('security_level'),
            tenant_id,  # Explicit tenant_id
            scan_data.get('geographic_region', 'global'),
            scan_data.get('business_unit', 'default'),
            tenant_id=tenant_id  # Pass tenant context to database manager
        )
        
        return True
    
    async def _get_scan_record(self, database: DatabaseManager, tenant_id: str, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get scan record from PostgreSQL with tenant context"""
        query = "SELECT * FROM scan_results WHERE scan_id = $1 AND tenant_id = $2"
        row = await database.fetch_row(query, scan_id, tenant_id, tenant_id=tenant_id)
        
        if row:
            return dict(row)
        return None
    
    async def _get_recent_scans(
        self, 
        database: DatabaseManager, 
        tenant_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get recent scans for a tenant from PostgreSQL"""
        query = """
            SELECT * FROM scan_results 
            WHERE tenant_id = $1 
            ORDER BY created_at DESC 
            LIMIT $2
        """
        
        rows = await database.fetch_all(query, tenant_id, limit, tenant_id=tenant_id)
        return [dict(row) for row in rows]
    
    async def _insert_threat_intelligence(
        self, 
        database: DatabaseManager, 
        tenant_id: str,
        threat_data: Dict[str, Any]
    ) -> bool:
        """Insert threat intelligence into PostgreSQL with tenant context"""
        # Ensure tenant_id is set in threat_data
        threat_data['tenant_id'] = tenant_id
        
        query = """
            INSERT INTO threat_intelligence (
                threat_id, attack_type, severity, confidence, techniques, indicators,
                attacker_fingerprint, source_ip, geographic_origin, mitigation,
                countermeasures, status, first_seen, last_seen, is_active,
                security_level, tenant_id, related_threats, evidence
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
        """
        
        await database.execute_query(
            query,
            threat_data.get('threat_id'),
            threat_data.get('attack_type'),
            threat_data.get('severity'),
            threat_data.get('confidence'),
            threat_data.get('techniques', []),
            threat_data.get('indicators', {}),
            threat_data.get('attacker_fingerprint'),
            threat_data.get('source_ip', ''),
            threat_data.get('geographic_origin', ''),
            threat_data.get('mitigation'),
            threat_data.get('countermeasures', []),
            threat_data.get('status'),
            threat_data.get('first_seen'),
            threat_data.get('last_seen'),
            threat_data.get('is_active', True),
            threat_data.get('security_level', 'restricted'),
            tenant_id,  # Explicit tenant_id
            threat_data.get('related_threats', []),
            threat_data.get('evidence', []),
            tenant_id=tenant_id  # Pass tenant context to database manager
        )
        
        return True
    
    async def _get_threat_intelligence(
        self, 
        database: DatabaseManager, 
        tenant_id: str,
        threat_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get threat intelligence from PostgreSQL with tenant context"""
        query = "SELECT * FROM threat_intelligence WHERE threat_id = $1 AND tenant_id = $2"
        row = await database.fetch_row(query, threat_id, tenant_id, tenant_id=tenant_id)
        
        if row:
            return dict(row)
        return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ❤️ Perform comprehensive health check on all databases with tenant awareness
        💡 Returns comprehensive status of all connected databases with performance metrics
        """
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation_stats": self.operation_stats.copy(),
            "databases": {},
            "tenant_health": {}
        }
        
        operational_dbs = 0
        
        # Check each database health
        for db_type, database in self.databases.items():
            try:
                db_health = await database.health_check()
                health_status["databases"][db_type.value] = db_health
                
                if db_health.get("status") == "healthy":
                    operational_dbs += 1
                else:
                    health_status["overall_status"] = "degraded"
                    
            except Exception as e:
                health_status["databases"][db_type.value] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["overall_status"] = "degraded"
        
        # Check tenant-specific health
        for tenant_id in list(self.tenant_operations.keys())[:10]:  # Sample first 10 tenants
            try:
                tenant_health = await self._check_tenant_health(tenant_id)
                health_status["tenant_health"][tenant_id] = tenant_health
            except Exception as e:
                health_status["tenant_health"][tenant_id] = {"status": "error", "error": str(e)}
        
        if operational_dbs == 0:
            health_status["overall_status"] = "unhealthy"
        
        health_status["operational_databases"] = operational_dbs
        health_status["total_databases"] = len(self.databases)
        
        # Calculate cache performance
        total_cache_ops = self.operation_stats["cache_hits"] + self.operation_stats["cache_misses"]
        if total_cache_ops > 0:
            health_status["cache_hit_rate"] = round(
                self.operation_stats["cache_hits"] / total_cache_ops, 3
            )
        
        return health_status
    
    async def _check_tenant_health(self, tenant_id: str) -> Dict[str, Any]:
        """Check health for specific tenant"""
        tenant_health = {
            "status": "healthy",
            "databases_available": [],
            "last_operation": self.operation_stats["last_operation"]
        }
        
        # Test basic operations for the tenant
        try:
            # Test read operation
            test_result = await self.execute_operation(
                OperationType.READ, "get_recent_scans", tenant_id, limit=1
            )
            
            if test_result is not None:
                tenant_health["databases_available"].append("read_operations")
            
            # Test cache operation if Redis is available
            if DatabaseType.REDIS in self.databases:
                cache_test = await self.databases[DatabaseType.REDIS].get_cache_stats()
                tenant_health["cache_status"] = "healthy"
            
        except Exception as e:
            tenant_health["status"] = "degraded"
            tenant_health["error"] = str(e)
        
        return tenant_health
    
    async def close(self) -> None:
        """
        🔚 Close all database connections
        💡 Clean shutdown of all database connections with proper resource cleanup
        """
        logger.info("🔚 Closing multi-database connections...")
        
        for db_type, database in self.databases.items():
            try:
                await database.close()
                logger.info(f"✅ Closed {db_type.value} connection")
            except Exception as e:
                logger.error(f"❌ Error closing {db_type.value}: {e}")


# Global multi-database connector instance
multi_db = MultiDBConnector()


async def initialize_multi_database() -> bool:
    """
    🚀 Initialize multi-database connector with system settings integration
    💡 Main entry point for multi-database setup
    """
    return await multi_db.initialize()


async def close_multi_database() -> None:
    """
    🔚 Close all database connections
    💡 Clean shutdown procedure
    """
    await multi_db.close()