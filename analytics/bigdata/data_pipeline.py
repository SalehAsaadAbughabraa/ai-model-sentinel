"""
🎯 Data Pipeline Manager
📦 Orchestrates data flow between transactional and analytical databases
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Business Logic: 
   - Synchronizes data between PostgreSQL and analytical databases
   - Implements ETL (Extract, Transform, Load) processes
   - Handles data validation and quality checks
   - Supports incremental updates and batch processing
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from config.settings import settings
from core.database.multi_db_connector import multi_db, OperationType, DatabaseType


logger = logging.getLogger("DataPipeline")


class PipelineStatus(str, Enum):
    """📊 Data pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineType(str, Enum):
    """🔄 Types of data pipeline operations"""
    INCREMENTAL_SYNC = "incremental_sync"
    FULL_SYNC = "full_sync"
    BACKFILL = "backfill"
    REAL_TIME = "real_time"
    LOCAL_SYNC = "local_sync"


@dataclass
class PipelineExecution:
    """📝 Data pipeline execution record"""
    execution_id: str
    pipeline_type: PipelineType
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    records_processed: int = 0
    errors: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}


class LocalAnalyticalEngine:
    """
    💻 Local Analytical Engine for high-performance analytics
    💡 Uses PostgreSQL/DuckDB for analytical processing
    """
    
    def __init__(self):
        self.is_connected = False
        self.engine_type = "postgresql"  # Default to PostgreSQL
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize local analytical engine"""
        try:
            # Use existing PostgreSQL connection from multi_db
            self.is_connected = multi_db.is_connected(DatabaseType.POSTGRESQL)
            if self.is_connected:
                logger.info("✅ Local analytical engine (PostgreSQL) initialized successfully")
                
                # Create analytical tables and indexes if they don't exist
                asyncio.create_task(self._setup_analytical_schema())
            else:
                logger.warning("⚠️ PostgreSQL not available for analytical engine")
                
        except Exception as e:
            logger.error(f"❌ Local analytical engine initialization failed: {e}")
    
    async def _setup_analytical_schema(self):
        """Setup analytical schema with optimized indexes"""
        try:
            # Create analytical tables (if using separate schema)
            analytical_tables = [
                """
                CREATE TABLE IF NOT EXISTS analytics_scan_results (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id VARCHAR NOT NULL,
                    repository VARCHAR NOT NULL,
                    threat_level VARCHAR NOT NULL,
                    threat_score FLOAT,
                    scan_duration FLOAT,
                    models_scanned INTEGER,
                    status VARCHAR,
                    geographic_region VARCHAR,
                    user_id VARCHAR,
                    created_at TIMESTAMP WITH TIME ZONE,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    sync_version INTEGER DEFAULT 1
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS analytics_threat_intelligence (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id VARCHAR NOT NULL,
                    threat_type VARCHAR NOT NULL,
                    severity VARCHAR NOT NULL,
                    confidence_score FLOAT,
                    description TEXT,
                    mitigation_action VARCHAR,
                    created_at TIMESTAMP WITH TIME ZONE,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    sync_version INTEGER DEFAULT 1
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS analytics_audit_logs (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id VARCHAR NOT NULL,
                    user_id VARCHAR NOT NULL,
                    action VARCHAR NOT NULL,
                    resource_type VARCHAR,
                    resource_id VARCHAR,
                    success BOOLEAN,
                    ip_address VARCHAR,
                    user_agent TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE,
                    sync_version INTEGER DEFAULT 1
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS pipeline_sync_metadata (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id VARCHAR NOT NULL,
                    table_name VARCHAR NOT NULL,
                    last_sync_timestamp TIMESTAMP WITH TIME ZONE,
                    records_synced INTEGER DEFAULT 0,
                    sync_status VARCHAR DEFAULT 'pending',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(tenant_id, table_name)
                )
                """
            ]
            
            # Create performance indexes
            analytical_indexes = [
                "CREATE INDEX IF NOT EXISTS idx_analytics_scans_tenant_created ON analytics_scan_results(tenant_id, created_at)",
                "CREATE INDEX IF NOT EXISTS idx_analytics_scans_threat_level ON analytics_scan_results(threat_level, threat_score)",
                "CREATE INDEX IF NOT EXISTS idx_analytics_scans_repository ON analytics_scan_results(repository)",
                "CREATE INDEX IF NOT EXISTS idx_analytics_threats_tenant_created ON analytics_threat_intelligence(tenant_id, created_at)",
                "CREATE INDEX IF NOT EXISTS idx_analytics_audit_tenant_timestamp ON analytics_audit_logs(tenant_id, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_analytics_audit_action ON analytics_audit_logs(action, success)",
                "CREATE INDEX IF NOT EXISTS idx_sync_metadata_tenant ON pipeline_sync_metadata(tenant_id, table_name)"
            ]
            
            # Execute schema setup
            for table_sql in analytical_tables:
                try:
                    await multi_db.execute_operation(
                        OperationType.WRITE,
                        "execute",
                        table_sql
                    )
                except Exception as e:
                    logger.debug(f"Table might already exist: {e}")
            
            for index_sql in analytical_indexes:
                try:
                    await multi_db.execute_operation(
                        OperationType.WRITE,
                        "execute",
                        index_sql
                    )
                except Exception as e:
                    logger.debug(f"Index might already exist: {e}")
            
            logger.info("✅ Analytical schema setup completed")
            
        except Exception as e:
            logger.error(f"❌ Analytical schema setup failed: {e}")
    
    async def load_data(self, table_name: str, records: List[Dict[str, Any]]) -> bool:
        """
        📤 Load data to local analytical tables
        💡 Uses upsert to handle incremental updates
        """
        if not self.is_connected:
            logger.error("❌ Local analytical engine not connected")
            return False
        
        try:
            if not records:
                logger.warning("⚠️ No records to load")
                return True
            
            analytical_table = f"analytics_{table_name}"
            
            if table_name == "scan_results":
                await self._upsert_scan_results(analytical_table, records)
            elif table_name == "threat_intelligence":
                await self._upsert_threat_intelligence(analytical_table, records)
            elif table_name == "audit_logs":
                await self._upsert_audit_logs(analytical_table, records)
            else:
                logger.error(f"❌ Unknown table: {table_name}")
                return False
            
            logger.info(f"✅ Loaded {len(records)} records to {analytical_table}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Data load failed for {table_name}: {e}")
            return False
    
    async def _upsert_scan_results(self, table: str, records: List[Dict[str, Any]]):
        """Upsert scan results with conflict resolution"""
        query = f"""
            INSERT INTO {table} 
            (id, tenant_id, repository, threat_level, threat_score, scan_duration, 
             models_scanned, status, geographic_region, user_id, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW())
            ON CONFLICT (id) 
            DO UPDATE SET 
                threat_level = EXCLUDED.threat_level,
                threat_score = EXCLUDED.threat_score,
                scan_duration = EXCLUDED.scan_duration,
                models_scanned = EXCLUDED.models_scanned,
                status = EXCLUDED.status,
                updated_at = NOW(),
                sync_version = {table}.sync_version + 1
        """
        
        for record in records:
            await multi_db.execute_operation(
                OperationType.WRITE,
                "execute",
                query,
                record.get('id'),
                record.get('tenant_id'),
                record.get('repository'),
                record.get('threat_level'),
                record.get('threat_score'),
                record.get('scan_duration'),
                record.get('models_scanned'),
                record.get('status'),
                record.get('geographic_region'),
                record.get('user_id'),
                record.get('created_at')
            )
    
    async def _upsert_threat_intelligence(self, table: str, records: List[Dict[str, Any]]):
        """Upsert threat intelligence data"""
        query = f"""
            INSERT INTO {table} 
            (id, tenant_id, threat_type, severity, confidence_score, description, 
             mitigation_action, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (id) 
            DO UPDATE SET 
                severity = EXCLUDED.severity,
                confidence_score = EXCLUDED.confidence_score,
                description = EXCLUDED.description,
                mitigation_action = EXCLUDED.mitigation_action,
                updated_at = NOW(),
                sync_version = {table}.sync_version + 1
        """
        
        for record in records:
            await multi_db.execute_operation(
                OperationType.WRITE,
                "execute",
                query,
                record.get('id'),
                record.get('tenant_id'),
                record.get('threat_type'),
                record.get('severity'),
                record.get('confidence_score'),
                record.get('description'),
                record.get('mitigation_action'),
                record.get('created_at')
            )
    
    async def _upsert_audit_logs(self, table: str, records: List[Dict[str, Any]]):
        """Upsert audit logs data"""
        query = f"""
            INSERT INTO {table} 
            (id, tenant_id, user_id, action, resource_type, resource_id, 
             success, ip_address, user_agent, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) 
            DO UPDATE SET 
                success = EXCLUDED.success,
                sync_version = {table}.sync_version + 1
        """
        
        for record in records:
            await multi_db.execute_operation(
                OperationType.WRITE,
                "execute",
                query,
                record.get('id'),
                record.get('tenant_id'),
                record.get('user_id'),
                record.get('action'),
                record.get('resource_type'),
                record.get('resource_id'),
                record.get('success'),
                record.get('ip_address'),
                record.get('user_agent'),
                record.get('timestamp')
            )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check local analytical engine health"""
        return {
            "status": "healthy" if self.is_connected else "disconnected",
            "engine_type": self.engine_type,
            "connected": self.is_connected,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class DataPipelineManager:
    """
    🔄 Data pipeline manager for AI Model Sentinel
    💡 Orchestrates data flow between operational and analytical systems
    """
    
    def __init__(self):
        self.active_pipelines: Dict[str, PipelineExecution] = {}
        self.pipeline_stats: Dict[str, Any] = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_records_processed": 0,
            "last_execution": None
        }
        self.local_engine = LocalAnalyticalEngine()
    
    async def run_incremental_sync(self, tenant_id: str) -> PipelineExecution:
        """
        🔄 Run incremental data synchronization
        💡 Syncs only new/changed data since last sync
        """
        execution = PipelineExecution(
            execution_id=f"inc_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pipeline_type=PipelineType.INCREMENTAL_SYNC,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(timezone.utc),
            metadata={"tenant_id": tenant_id}
        )
        
        self.active_pipelines[execution.execution_id] = execution
        self.pipeline_stats["total_executions"] += 1
        
        try:
            logger.info(f"🔄 Starting incremental sync for tenant {tenant_id}")
            
            # Get last sync timestamp
            last_sync = await self._get_last_sync_timestamp(tenant_id)
            
            # Extract new data from PostgreSQL
            new_scans = await self._extract_new_scans(tenant_id, last_sync)
            execution.records_processed += len(new_scans)
            
            new_threats = await self._extract_new_threats(tenant_id, last_sync)
            execution.records_processed += len(new_threats)
            
            new_audit_logs = await self._extract_new_audit_logs(tenant_id, last_sync)
            execution.records_processed += len(new_audit_logs)
            
            # Load to local analytical engine
            if new_scans:
                await self._load_to_local_engine("scan_results", new_scans)
            
            if new_threats:
                await self._load_to_local_engine("threat_intelligence", new_threats)
            
            if new_audit_logs:
                await self._load_to_local_engine("audit_logs", new_audit_logs)
            
            # Update sync timestamp
            await self._update_sync_timestamp(tenant_id, "incremental_sync", execution.records_processed)
            
            # Mark execution as completed
            execution.status = PipelineStatus.COMPLETED
            execution.end_time = datetime.now(timezone.utc)
            self.pipeline_stats["successful_executions"] += 1
            self.pipeline_stats["total_records_processed"] += execution.records_processed
            self.pipeline_stats["last_execution"] = execution.end_time.isoformat()
            
            logger.info(f"✅ Incremental sync completed: {execution.records_processed} records processed")
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.end_time = datetime.now(timezone.utc)
            execution.errors.append(str(e))
            self.pipeline_stats["failed_executions"] += 1
            logger.error(f"❌ Incremental sync failed: {e}")
        
        return execution
    
    async def run_full_sync(self, tenant_id: str, start_date: datetime, end_date: datetime) -> PipelineExecution:
        """
        📦 Run full data synchronization for a date range
        💡 Used for initial setup or backfilling historical data
        """
        execution = PipelineExecution(
            execution_id=f"full_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            pipeline_type=PipelineType.FULL_SYNC,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(timezone.utc),
            metadata={
                "tenant_id": tenant_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            }
        )
        
        self.active_pipelines[execution.execution_id] = execution
        self.pipeline_stats["total_executions"] += 1
        
        try:
            logger.info(f"📦 Starting full sync for tenant {tenant_id} from {start_date} to {end_date}")
            
            # Extract all data for the date range
            all_scans = await self._extract_scans_by_date_range(tenant_id, start_date, end_date)
            execution.records_processed += len(all_scans)
            
            all_threats = await self._extract_threats_by_date_range(tenant_id, start_date, end_date)
            execution.records_processed += len(all_threats)
            
            all_audit_logs = await self._extract_audit_logs_by_date_range(tenant_id, start_date, end_date)
            execution.records_processed += len(all_audit_logs)
            
            # Load to local analytical engine
            if all_scans:
                await self._load_to_local_engine("scan_results", all_scans)
            
            if all_threats:
                await self._load_to_local_engine("threat_intelligence", all_threats)
            
            if all_audit_logs:
                await self._load_to_local_engine("audit_logs", all_audit_logs)
            
            # Update sync metadata
            await self._update_sync_timestamp(tenant_id, "full_sync", execution.records_processed)
            
            # Mark execution as completed
            execution.status = PipelineStatus.COMPLETED
            execution.end_time = datetime.now(timezone.utc)
            self.pipeline_stats["successful_executions"] += 1
            self.pipeline_stats["total_records_processed"] += execution.records_processed
            self.pipeline_stats["last_execution"] = execution.end_time.isoformat()
            
            logger.info(f"✅ Full sync completed: {execution.records_processed} records processed")
            
        except Exception as e:
            execution.status = PipelineStatus.FAILED
            execution.end_time = datetime.now(timezone.utc)
            execution.errors.append(str(e))
            self.pipeline_stats["failed_executions"] += 1
            logger.error(f"❌ Full sync failed: {e}")
        
        return execution
    
    async def _extract_new_scans(self, tenant_id: str, since: Optional[datetime]) -> List[Dict[str, Any]]:
        """Extract new scan records since last sync"""
        try:
            query = """
                SELECT * FROM scan_results 
                WHERE tenant_id = $1 
                AND created_at > $2
                ORDER BY created_at
            """
            
            since_param = since or datetime(1970, 1, 1, tzinfo=timezone.utc)
            rows = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_all",
                query,
                tenant_id,
                since_param
            )
            
            return [dict(row) for row in rows] if rows else []
            
        except Exception as e:
            logger.error(f"❌ Scan extraction failed: {e}")
            return []
    
    async def _extract_new_threats(self, tenant_id: str, since: Optional[datetime]) -> List[Dict[str, Any]]:
        """Extract new threat intelligence records since last sync"""
        try:
            query = """
                SELECT * FROM threat_intelligence 
                WHERE tenant_id = $1 
                AND created_at > $2
                ORDER BY created_at
            """
            
            since_param = since or datetime(1970, 1, 1, tzinfo=timezone.utc)
            rows = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_all",
                query,
                tenant_id,
                since_param
            )
            
            return [dict(row) for row in rows] if rows else []
            
        except Exception as e:
            logger.error(f"❌ Threat extraction failed: {e}")
            return []
    
    async def _extract_new_audit_logs(self, tenant_id: str, since: Optional[datetime]) -> List[Dict[str, Any]]:
        """Extract new audit log records since last sync"""
        try:
            query = """
                SELECT * FROM audit_logs 
                WHERE tenant_id = $1 
                AND timestamp > $2
                ORDER BY timestamp
            """
            
            since_param = since or datetime(1970, 1, 1, tzinfo=timezone.utc)
            rows = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_all",
                query,
                tenant_id,
                since_param
            )
            
            return [dict(row) for row in rows] if rows else []
            
        except Exception as e:
            logger.error(f"❌ Audit log extraction failed: {e}")
            return []
    
    async def _extract_scans_by_date_range(self, tenant_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Extract scan records for a date range"""
        try:
            query = """
                SELECT * FROM scan_results 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
                ORDER BY created_at
            """
            
            rows = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_all",
                query,
                tenant_id,
                start_date,
                end_date
            )
            
            return [dict(row) for row in rows] if rows else []
            
        except Exception as e:
            logger.error(f"❌ Scan range extraction failed: {e}")
            return []
    
    async def _extract_threats_by_date_range(self, tenant_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Extract threat records for a date range"""
        try:
            query = """
                SELECT * FROM threat_intelligence 
                WHERE tenant_id = $1 
                AND created_at BETWEEN $2 AND $3
                ORDER BY created_at
            """
            
            rows = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_all",
                query,
                tenant_id,
                start_date,
                end_date
            )
            
            return [dict(row) for row in rows] if rows else []
            
        except Exception as e:
            logger.error(f"❌ Threat range extraction failed: {e}")
            return []
    
    async def _extract_audit_logs_by_date_range(self, tenant_id: str, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Extract audit log records for a date range"""
        try:
            query = """
                SELECT * FROM audit_logs 
                WHERE tenant_id = $1 
                AND timestamp BETWEEN $2 AND $3
                ORDER BY timestamp
            """
            
            rows = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_all",
                query,
                tenant_id,
                start_date,
                end_date
            )
            
            return [dict(row) for row in rows] if rows else []
            
        except Exception as e:
            logger.error(f"❌ Audit log range extraction failed: {e}")
            return []
    
    async def _load_to_local_engine(self, table_name: str, records: List[Dict[str, Any]]) -> bool:
        """Load records to local analytical engine"""
        return await self.local_engine.load_data(table_name, records)
    
    async def _get_last_sync_timestamp(self, tenant_id: str) -> Optional[datetime]:
        """Get the last successful sync timestamp for a tenant"""
        try:
            query = """
                SELECT last_sync_timestamp FROM pipeline_sync_metadata 
                WHERE tenant_id = $1 AND table_name = $2
                ORDER BY updated_at DESC LIMIT 1
            """
            
            result = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_row",
                query,
                tenant_id,
                "all_tables"
            )
            
            return result["last_sync_timestamp"] if result else None
            
        except Exception as e:
            logger.debug(f"Last sync timestamp query failed: {e}")
            return None
    
    async def _update_sync_timestamp(self, tenant_id: str, sync_type: str, records_processed: int) -> bool:
        """Update the last sync timestamp for a tenant"""
        try:
            query = """
                INSERT INTO pipeline_sync_metadata 
                (tenant_id, table_name, last_sync_timestamp, records_synced, sync_status)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (tenant_id, table_name) 
                DO UPDATE SET 
                    last_sync_timestamp = EXCLUDED.last_sync_timestamp,
                    records_synced = EXCLUDED.records_synced,
                    sync_status = EXCLUDED.sync_status,
                    updated_at = NOW()
            """
            
            await multi_db.execute_operation(
                OperationType.WRITE,
                "execute",
                query,
                tenant_id,
                "all_tables",
                datetime.now(timezone.utc),
                records_processed,
                "completed"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Sync timestamp update failed: {e}")
            return False
    
    async def validate_data_quality(self, tenant_id: str) -> Dict[str, Any]:
        """
        ✅ Perform data quality validation
        💡 Checks data consistency and integrity across systems
        """
        validation_results = {
            "status": "passed",
            "checks_performed": [],
            "issues_found": [],
            "recommendations": []
        }
        
        try:
            # Check record counts consistency between source and analytical tables
            source_count = await self._get_postgres_record_count("scan_results", tenant_id)
            analytical_count = await self._get_analytical_record_count("scan_results", tenant_id)
            
            validation_results["checks_performed"].append("record_count_consistency")
            
            if source_count != analytical_count:
                validation_results["issues_found"].append(
                    f"Record count mismatch: source={source_count}, analytical={analytical_count}"
                )
                validation_results["recommendations"].append("Run full data sync to resolve discrepancies")
            
            # Check data freshness
            latest_source_record = await self._get_latest_postgres_record("scan_results", tenant_id)
            validation_results["checks_performed"].append("data_freshness")
            
            if latest_source_record:
                data_age = datetime.now(timezone.utc) - latest_source_record
                if data_age > timedelta(hours=24):
                    validation_results["issues_found"].append("Data may be stale")
                    validation_results["recommendations"].append("Run incremental data sync")
            
            # Check for data anomalies
            anomaly_check = await self._check_data_anomalies(tenant_id)
            validation_results["checks_performed"].append("data_anomalies")
            
            if anomaly_check["has_anomalies"]:
                validation_results["status"] = "warning"
                validation_results["issues_found"].extend(anomaly_check["anomalies"])
            
            if validation_results["issues_found"]:
                validation_results["status"] = "failed" if validation_results["status"] != "warning" else "warning"
            
            return validation_results
            
        except Exception as e:
            validation_results["status"] = "error"
            validation_results["issues_found"].append(f"Validation error: {e}")
            return validation_results
    
    async def _get_postgres_record_count(self, table_name: str, tenant_id: str) -> int:
        """Get record count from PostgreSQL"""
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name} WHERE tenant_id = $1"
            result = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_row",
                query,
                tenant_id
            )
            return result["count"] if result else 0
        except Exception:
            return 0
    
    async def _get_analytical_record_count(self, table_name: str, tenant_id: str) -> int:
        """Get record count from analytical tables"""
        try:
            query = f"SELECT COUNT(*) as count FROM analytics_{table_name} WHERE tenant_id = $1"
            result = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_row",
                query,
                tenant_id
            )
            return result["count"] if result else 0
        except Exception:
            return 0
    
    async def _get_latest_postgres_record(self, table_name: str, tenant_id: str) -> Optional[datetime]:
        """Get timestamp of latest record from PostgreSQL"""
        try:
            timestamp_column = "created_at" if table_name != "audit_logs" else "timestamp"
            query = f"SELECT MAX({timestamp_column}) as latest FROM {table_name} WHERE tenant_id = $1"
            result = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_row",
                query,
                tenant_id
            )
            return result["latest"] if result and result["latest"] else None
        except Exception:
            return None
    
    async def _check_data_anomalies(self, tenant_id: str) -> Dict[str, Any]:
        """Check for data anomalies in analytical tables"""
        try:
            # Check for duplicate records
            duplicate_query = """
                SELECT id, COUNT(*) as duplicate_count 
                FROM analytics_scan_results 
                WHERE tenant_id = $1 
                GROUP BY id 
                HAVING COUNT(*) > 1
                LIMIT 10
            """
            
            duplicates = await multi_db.execute_operation(
                OperationType.READ,
                "fetch_all",
                duplicate_query,
                tenant_id
            )
            
            anomalies = []
            if duplicates:
                anomalies.append(f"Found {len(duplicates)} duplicate records in scan results")
            
            return {
                "has_anomalies": len(anomalies) > 0,
                "anomalies": anomalies
            }
            
        except Exception as e:
            return {
                "has_anomalies": False,
                "anomalies": [f"Anomaly check failed: {e}"]
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ❤️ Perform data pipeline health check
        💡 Verifies pipeline functionality and data consistency
        """
        health_status = {
            "status": "healthy",
            "pipeline_stats": self.pipeline_stats.copy(),
            "active_pipelines": len(self.active_pipelines),
            "local_analytical_engine": await self.local_engine.health_check(),
            "database_connection": multi_db.is_connected(DatabaseType.POSTGRESQL),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        try:
            # Test basic pipeline functionality with a small sync
            test_tenant = "health_check_tenant"
            test_execution = await self.run_incremental_sync(test_tenant)
            health_status["test_execution"] = test_execution.status.value
            
            if test_execution.status == PipelineStatus.FAILED:
                health_status["status"] = "degraded"
                health_status["test_errors"] = test_execution.errors
            
            # Check data quality
            quality_check = await self.validate_data_quality(test_tenant)
            health_status["data_quality"] = quality_check["status"]
            
        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)
        
        return health_status


# Global data pipeline manager instance
data_pipeline = DataPipelineManager()


async def initialize_data_pipeline() -> bool:
    """
    🚀 Initialize data pipeline system
    💡 Main entry point for data pipeline setup
    """
    try:
        # Initialize local analytical engine
        await data_pipeline.local_engine._setup_analytical_schema()
        
        health = await data_pipeline.health_check()
        if health["status"] in ["healthy", "degraded"]:
            logger.info("✅ Data pipeline system initialized successfully")
            return True
        else:
            logger.error("❌ Data pipeline system health check failed")
            return False
    except Exception as e:
        logger.error(f"❌ Data pipeline system initialization failed: {e}")
        return False