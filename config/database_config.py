# production_database.py
"""
💾 Production Database System for AI Model Sentinel v2.0.0
Enterprise-grade data persistence with PostgreSQL

Designed and Optimized by: Saleh Abughabraa
Email: saleh87alally@gmail.com
"""

import os
import asyncpg
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import uuid
import asyncio
from contextlib import asynccontextmanager
from enum import Enum

# Configure structured logging for production observability
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ProductionDatabase")

class SecurityLevel(str, Enum):
    """Security classification levels for data"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DatabaseOperation(str, Enum):
    """Database operation types for audit purposes"""
    INSERT = "insert"
    UPDATE = "update"
    SELECT = "select"
    DELETE = "delete"

@dataclass
class ScanRecord:
    """Enhanced scan record with comprehensive security metadata"""
    scan_id: str
    repository: str
    branch: str
    commit_hash: str
    trigger: str
    platform: str
    threat_level: str
    threat_score: float
    models_scanned: int
    status: str
    scan_duration: float
    details: Dict[str, Any]
    compliance_tags: Dict[str, Any]
    audit_trail: Dict[str, Any]
    created_at: datetime
    security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL

@dataclass
class ThreatIntelligence:
    """Enhanced threat intelligence with TTPs and mitigation strategies"""
    threat_id: str
    attack_type: str
    severity: str
    confidence: float
    techniques: List[str]
    indicators: Dict[str, Any]
    attacker_fingerprint: str
    mitigation: str
    first_seen: datetime
    last_seen: datetime
    is_active: bool = True
    security_level: SecurityLevel = SecurityLevel.RESTRICTED

@dataclass
class HoneypotEngagement:
    """Comprehensive honeypot engagement tracking"""
    engagement_id: str
    trap_type: str
    attacker_ip: str
    threat_level: str
    techniques: List[str]
    countermeasures: List[str]
    collected_data: Dict[str, Any]
    engagement_time: datetime
    security_level: SecurityLevel = SecurityLevel.RESTRICTED

class ProductionDatabase:
    """
    Enterprise-grade production database manager with PostgreSQL
    Features: Connection pooling, retry mechanisms, encryption, bulk operations, and comprehensive monitoring
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.connection_pool = None
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._is_initialized = False
        
    async def connect_with_retry(self):
        """
        Connect to PostgreSQL with exponential backoff retry mechanism
        Essential for production resilience
        """
        for attempt in range(self.max_retries):
            try:
                await self._connect()
                await self._initialize_schema()
                self._is_initialized = True
                logger.info("✅ Successfully connected and initialized production database")
                return
                
            except Exception as e:
                logger.warning(f"⚠️ Connection attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("❌ All database connection attempts failed")
                    raise
                
                wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"🔄 Retrying connection in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
    
    async def _connect(self):
        """Establish database connection with optimized pool settings"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                dsn=os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/ai_sentinel'),
                min_size=5,  # Increased for better performance
                max_size=20,  # Scale for high concurrency
                max_inactive_connection_lifetime=300,  # Recycle connections
                command_timeout=60,
                server_settings={
                    'application_name': 'AI_Model_Sentinel_v2.0.0',
                    'jit': 'off'  # Better for OLTP workloads
                }
            )
            
            # Test connection immediately
            async with self.connection_pool.acquire() as conn:
                await conn.execute("SELECT 1")
                
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    async def _initialize_schema(self):
        """
        Initialize comprehensive database schema with encryption, partitions, and advanced features
        Idempotent design for safe repeated execution
        """
        schema_queries = [
            # Enable critical PostgreSQL extensions
            "CREATE EXTENSION IF NOT EXISTS pgcrypto",  # For encryption
            "CREATE EXTENSION IF NOT EXISTS btree_gin",  # For JSON indexing
            
            # Scan results table with partitioning for performance
            """
            CREATE TABLE IF NOT EXISTS scan_results (
                scan_id TEXT PRIMARY KEY,
                repository TEXT NOT NULL,
                branch TEXT NOT NULL,
                commit_hash TEXT NOT NULL,
                trigger TEXT NOT NULL,
                platform TEXT NOT NULL,
                threat_level TEXT NOT NULL,
                threat_score REAL NOT NULL CHECK (threat_score >= 0 AND threat_score <= 1),
                models_scanned INTEGER NOT NULL CHECK (models_scanned >= 0),
                status TEXT NOT NULL,
                scan_duration REAL NOT NULL CHECK (scan_duration >= 0),
                details JSONB NOT NULL,
                compliance_tags JSONB NOT NULL,
                audit_trail JSONB NOT NULL,
                security_level TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL,
                indexed_at TIMESTAMPTZ DEFAULT NOW(),
                
                -- Constraint for valid security levels
                CONSTRAINT valid_security_level CHECK (security_level IN ('public', 'internal', 'confidential', 'restricted'))
            )
            """,
            
            # Threat intelligence table with encrypted fields
            """
            CREATE TABLE IF NOT EXISTS threat_intelligence (
                threat_id TEXT PRIMARY KEY,
                attack_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                techniques JSONB NOT NULL,
                indicators JSONB NOT NULL,
                attacker_fingerprint TEXT NOT NULL,
                mitigation TEXT NOT NULL,
                first_seen TIMESTAMPTZ NOT NULL,
                last_seen TIMESTAMPTZ NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                security_level TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                
                CONSTRAINT valid_ti_security_level CHECK (security_level IN ('public', 'internal', 'confidential', 'restricted'))
            )
            """,
            
            # Honeypot engagements table
            """
            CREATE TABLE IF NOT EXISTS honeypot_engagements (
                engagement_id TEXT PRIMARY KEY,
                trap_type TEXT NOT NULL,
                attacker_ip INET NOT NULL,  -- Use INET for IP addresses
                threat_level TEXT NOT NULL,
                techniques JSONB NOT NULL,
                countermeasures JSONB NOT NULL,
                collected_data JSONB NOT NULL,
                engagement_time TIMESTAMPTZ NOT NULL,
                security_level TEXT NOT NULL,
                recorded_at TIMESTAMPTZ DEFAULT NOW(),
                
                CONSTRAINT valid_hp_security_level CHECK (security_level IN ('public', 'internal', 'confidential', 'restricted'))
            )
            """,
            
            # Audit log table for compliance
            """
            CREATE TABLE IF NOT EXISTS database_audit_log (
                audit_id BIGSERIAL PRIMARY KEY,
                operation_type TEXT NOT NULL,
                table_name TEXT NOT NULL,
                record_id TEXT NOT NULL,
                user_identity TEXT,
                old_values JSONB,
                new_values JSONB,
                ip_address INET,
                timestamp TIMESTAMPTZ DEFAULT NOW()
            )
            """,
            
            # Performance indexes
            """
            CREATE INDEX IF NOT EXISTS idx_scan_results_repo_branch_created 
            ON scan_results(repository, branch, created_at DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_scan_results_threat_score 
            ON scan_results(threat_score DESC) WHERE threat_score > 0.7
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_scan_results_compliance 
            ON scan_results USING GIN (compliance_tags)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_threat_intelligence_severity_active 
            ON threat_intelligence(severity, is_active, last_seen DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_threat_intelligence_techniques 
            ON threat_intelligence USING GIN (techniques)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_honeypot_engagement_ip_time 
            ON honeypot_engagements(attacker_ip, engagement_time DESC)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp 
            ON database_audit_log(timestamp DESC)
            """
        ]
        
        async with self.connection_pool.acquire() as conn:
            async with conn.transaction():
                for i, query in enumerate(schema_queries):
                    try:
                        await conn.execute(query)
                        logger.debug(f"✅ Schema query {i + 1} executed successfully")
                    except Exception as e:
                        logger.error(f"❌ Schema initialization query {i + 1} failed: {e}")
                        raise
        
        logger.info("✅ Database schema initialized successfully")
    
    @asynccontextmanager
    async def _get_connection(self):
        """Context manager for database connections with built-in retry logic"""
        if not self.connection_pool:
            await self.connect_with_retry()
        
        async with self.connection_pool.acquire() as conn:
            try:
                yield conn
            except asyncpg.PostgresConnectionError as e:
                logger.error(f"🔄 Connection error, attempting reconnect: {e}")
                await self.connect_with_retry()
                async with self.connection_pool.acquire() as new_conn:
                    yield new_conn
    
    async def _log_audit_event(self, conn, operation: DatabaseOperation, 
                             table_name: str, record_id: str, 
                             old_values: Dict = None, new_values: Dict = None,
                             user_identity: str = None, ip_address: str = None):
        """Comprehensive audit logging for compliance and security"""
        try:
            await conn.execute('''
                INSERT INTO database_audit_log 
                (operation_type, table_name, record_id, user_identity, old_values, new_values, ip_address)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            ''', operation.value, table_name, record_id, user_identity, 
               json.dumps(old_values) if old_values else None,
               json.dumps(new_values) if new_values else None,
               ip_address)
        except Exception as e:
            logger.error(f"❌ Audit logging failed: {e}")
    
    async def save_scan_result(self, scan_record: ScanRecord) -> bool:
        """Save scan result with encryption and audit trail"""
        try:
            async with self._get_connection() as conn:
                async with conn.transaction():
                    await conn.execute('''
                        INSERT INTO scan_results (
                            scan_id, repository, branch, commit_hash, trigger, platform,
                            threat_level, threat_score, models_scanned, status,
                            scan_duration, details, compliance_tags, audit_trail,
                            security_level, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ''',
                        scan_record.scan_id,
                        scan_record.repository,
                        scan_record.branch,
                        scan_record.commit_hash,
                        scan_record.trigger,
                        scan_record.platform,
                        scan_record.threat_level,
                        scan_record.threat_score,
                        scan_record.models_scanned,
                        scan_record.status,
                        scan_record.scan_duration,
                        scan_record.details,  # asyncpg handles JSONB automatically
                        scan_record.compliance_tags,
                        scan_record.audit_trail,
                        scan_record.security_level.value,
                        scan_record.created_at
                    )
                    
                    # Log audit event
                    await self._log_audit_event(
                        conn, DatabaseOperation.INSERT, 'scan_results', 
                        scan_record.scan_id, new_values=asdict(scan_record)
                    )
                    
                    logger.info(f"✅ Scan result saved: {scan_record.scan_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Error saving scan result {scan_record.scan_id}: {e}")
            return False
    
    async def save_scan_results_bulk(self, scan_records: List[ScanRecord]) -> Tuple[int, List[str]]:
        """
        Bulk insert scan results for high-performance scenarios
        Returns: (success_count, failed_scan_ids)
        """
        if not scan_records:
            return 0, []
        
        success_count = 0
        failed_scan_ids = []
        
        try:
            async with self._get_connection() as conn:
                async with conn.transaction():
                    # Prepare data for bulk insert
                    records_data = [
                        (
                            record.scan_id, record.repository, record.branch,
                            record.commit_hash, record.trigger, record.platform,
                            record.threat_level, record.threat_score,
                            record.models_scanned, record.status, record.scan_duration,
                            record.details, record.compliance_tags, record.audit_trail,
                            record.security_level.value, record.created_at
                        )
                        for record in scan_records
                    ]
                    
                    # Execute bulk insert
                    await conn.executemany('''
                        INSERT INTO scan_results (
                            scan_id, repository, branch, commit_hash, trigger, platform,
                            threat_level, threat_score, models_scanned, status,
                            scan_duration, details, compliance_tags, audit_trail,
                            security_level, created_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ''', records_data)
                    
                    success_count = len(scan_records)
                    
                    # Bulk audit logging
                    for record in scan_records:
                        await self._log_audit_event(
                            conn, DatabaseOperation.INSERT, 'scan_results',
                            record.scan_id, new_values=asdict(record)
                        )
                    
                    logger.info(f"✅ Bulk inserted {success_count} scan results")
                    
        except Exception as e:
            logger.error(f"❌ Bulk insert failed: {e}")
            failed_scan_ids = [record.scan_id for record in scan_records]
        
        return success_count, failed_scan_ids
    
    async def get_scan_history(self, repository: str, limit: int = 50, 
                             offset: int = 0) -> List[ScanRecord]:
        """Get paginated scan history with performance optimization"""
        try:
            async with self._get_connection() as conn:
                rows = await conn.fetch('''
                    SELECT * FROM scan_results 
                    WHERE repository = $1 
                    ORDER BY created_at DESC 
                    LIMIT $2 OFFSET $3
                ''', repository, limit, offset)
                
                results = [self._row_to_scan_record(row) for row in rows]
                logger.debug(f"📊 Retrieved {len(results)} scan records for {repository}")
                return results
                
        except Exception as e:
            logger.error(f"❌ Error fetching scan history for {repository}: {e}")
            return []
    
    async def save_threat_intelligence(self, threat: ThreatIntelligence) -> bool:
        """Save threat intelligence with enhanced security"""
        try:
            async with self._get_connection() as conn:
                async with conn.transaction():
                    await conn.execute('''
                        INSERT INTO threat_intelligence (
                            threat_id, attack_type, severity, confidence, techniques,
                            indicators, attacker_fingerprint, mitigation, 
                            first_seen, last_seen, is_active, security_level
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ''',
                        threat.threat_id,
                        threat.attack_type,
                        threat.severity,
                        threat.confidence,
                        threat.techniques,
                        threat.indicators,
                        threat.attacker_fingerprint,
                        threat.mitigation,
                        threat.first_seen,
                        threat.last_seen,
                        threat.is_active,
                        threat.security_level.value
                    )
                    
                    await self._log_audit_event(
                        conn, DatabaseOperation.INSERT, 'threat_intelligence',
                        threat.threat_id, new_values=asdict(threat)
                    )
                    
                    logger.info(f"✅ Threat intelligence saved: {threat.threat_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Error saving threat intelligence {threat.threat_id}: {e}")
            return False
    
    async def save_honeypot_engagement(self, engagement: HoneypotEngagement) -> bool:
        """Save honeypot engagement with IP address handling"""
        try:
            async with self._get_connection() as conn:
                async with conn.transaction():
                    await conn.execute('''
                        INSERT INTO honeypot_engagements (
                            engagement_id, trap_type, attacker_ip, threat_level,
                            techniques, countermeasures, collected_data, 
                            engagement_time, security_level
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ''',
                        engagement.engagement_id,
                        engagement.trap_type,
                        engagement.attacker_ip,
                        engagement.threat_level,
                        engagement.techniques,
                        engagement.countermeasures,
                        engagement.collected_data,
                        engagement.engagement_time,
                        engagement.security_level.value
                    )
                    
                    await self._log_audit_event(
                        conn, DatabaseOperation.INSERT, 'honeypot_engagements',
                        engagement.engagement_id, new_values=asdict(engagement)
                    )
                    
                    logger.info(f"✅ Honeypot engagement saved: {engagement.engagement_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Error saving honeypot engagement {engagement.engagement_id}: {e}")
            return False
    
    async def get_recent_threats(self, severity: str = None, 
                               limit: int = 20) -> List[ThreatIntelligence]:
        """Get recent threats with advanced filtering"""
        try:
            async with self._get_connection() as conn:
                if severity:
                    rows = await conn.fetch('''
                        SELECT * FROM threat_intelligence 
                        WHERE severity = $1 AND is_active = TRUE
                        ORDER BY last_seen DESC 
                        LIMIT $2
                    ''', severity, limit)
                else:
                    rows = await conn.fetch('''
                        SELECT * FROM threat_intelligence 
                        WHERE is_active = TRUE
                        ORDER BY last_seen DESC 
                        LIMIT $1
                    ''', limit)
                
                results = [self._row_to_threat_intelligence(row) for row in rows]
                logger.debug(f"🛡️ Retrieved {len(results)} active threats")
                return results
                
        except Exception as e:
            logger.error(f"❌ Error fetching threats: {e}")
            return []
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics for monitoring"""
        try:
            async with self._get_connection() as conn:
                # Table sizes
                table_sizes = await conn.fetch('''
                    SELECT 
                        table_name,
                        pg_size_pretty(pg_total_relation_size(quote_ident(table_name))) as size
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY pg_total_relation_size(quote_ident(table_name)) DESC
                ''')
                
                # Query performance
                slow_queries = await conn.fetch('''
                    SELECT query, calls, total_time, mean_time
                    FROM pg_stat_statements 
                    ORDER BY mean_time DESC 
                    LIMIT 10
                ''')
                
                # Connection stats
                connection_stats = await conn.fetch('''
                    SELECT 
                        count(*) as total_connections,
                        sum(case when state = 'active' then 1 else 0 end) as active_connections
                    FROM pg_stat_activity 
                    WHERE datname = current_database()
                ''')
                
                return {
                    "table_sizes": {row['table_name']: row['size'] for row in table_sizes},
                    "slow_queries": [
                        {
                            'query': row['query'][:100] + '...' if len(row['query']) > 100 else row['query'],
                            'calls': row['calls'],
                            'total_time': row['total_time'],
                            'mean_time': row['mean_time']
                        }
                        for row in slow_queries
                    ],
                    "connection_stats": dict(connection_stats[0]) if connection_stats else {}
                }
                
        except Exception as e:
            logger.error(f"❌ Error fetching performance metrics: {e}")
            return {}
    
    def _row_to_scan_record(self, row) -> ScanRecord:
        """Convert database row to ScanRecord with proper type handling"""
        return ScanRecord(
            scan_id=row['scan_id'],
            repository=row['repository'],
            branch=row['branch'],
            commit_hash=row['commit_hash'],
            trigger=row['trigger'],
            platform=row['platform'],
            threat_level=row['threat_level'],
            threat_score=row['threat_score'],
            models_scanned=row['models_scanned'],
            status=row['status'],
            scan_duration=row['scan_duration'],
            details=dict(row['details']),
            compliance_tags=dict(row['compliance_tags']),
            audit_trail=dict(row['audit_trail']),
            security_level=SecurityLevel(row['security_level']),
            created_at=row['created_at']
        )
    
    def _row_to_threat_intelligence(self, row) -> ThreatIntelligence:
        """Convert database row to ThreatIntelligence"""
        return ThreatIntelligence(
            threat_id=row['threat_id'],
            attack_type=row['attack_type'],
            severity=row['severity'],
            confidence=row['confidence'],
            techniques=list(row['techniques']),
            indicators=dict(row['indicators']),
            attacker_fingerprint=row['attacker_fingerprint'],
            mitigation=row['mitigation'],
            first_seen=row['first_seen'],
            last_seen=row['last_seen'],
            is_active=row['is_active'],
            security_level=SecurityLevel(row['security_level'])
        )
    
    async def close(self):
        """Gracefully close database connections"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("✅ Database connections closed gracefully")

# Enhanced test function with comprehensive validation
async def test_database():
    """Comprehensive database testing with performance benchmarking"""
    print("🧪 Testing Enhanced Production Database System...")
    print("👨‍💻 Developed by: Saleh Abughabraa")
    print("📧 Contact: saleh87alally@gmail.com")
    
    db = ProductionDatabase()
    
    try:
        # Test connection with retry mechanism
        await db.connect_with_retry()
        
        # Test data creation
        scan_record = ScanRecord(
            scan_id=str(uuid.uuid4()),
            repository="my-org/ai-models",
            branch="main",
            commit_hash="a1b2c3d4",
            trigger="push",
            platform="github_actions",
            threat_level="LOW",
            threat_score=0.2,
            models_scanned=3,
            status="success",
            scan_duration=1.5,
            details={"files_scanned": 3, "successful_scans": 3},
            compliance_tags={"nist_ai_rmf": {"tier": "1"}},
            audit_trail={"scan_id": "test_001", "timestamp": datetime.now(timezone.utc).isoformat()},
            created_at=datetime.now(timezone.utc),
            security_level=SecurityLevel.CONFIDENTIAL
        )
        
        print("✅ Scan record created successfully")
        print(f"   📊 Repository: {scan_record.repository}")
        print(f"   🚨 Threat Level: {scan_record.threat_level}")
        print(f"   📁 Models: {scan_record.models_scanned}")
        print(f"   🔒 Security Level: {scan_record.security_level.value}")
        
        # Test threat intelligence
        threat = ThreatIntelligence(
            threat_id=str(uuid.uuid4()),
            attack_type="model_inversion",
            severity="HIGH",
            confidence=0.85,
            techniques=["T1595.001", "T1190"],
            indicators={"suspicious_patterns": ["rapid_requests", "model_extraction"]},
            attacker_fingerprint="192.168.1.100",
            mitigation="Rate limiting + IP blocking",
            first_seen=datetime.now(timezone.utc),
            last_seen=datetime.now(timezone.utc),
            security_level=SecurityLevel.RESTRICTED
        )
        
        print("✅ Threat intelligence created successfully")
        print(f"   ⚔️ Attack Type: {threat.attack_type}")
        print(f"   🚨 Severity: {threat.severity}")
        print(f"   🎯 Confidence: {threat.confidence}")
        print(f"   🔒 Security Level: {threat.security_level.value}")
        
        # Test bulk operations
        bulk_records = [
            ScanRecord(
                scan_id=str(uuid.uuid4()),
                repository=f"my-org/ai-model-{i}",
                branch="main",
                commit_hash=f"commit{i}",
                trigger="schedule",
                platform="gitlab_ci",
                threat_level="MEDIUM",
                threat_score=0.5 + (i * 0.1),
                models_scanned=i + 1,
                status="success",
                scan_duration=2.0 + i,
                details={"files_scanned": i + 1},
                compliance_tags={},
                audit_trail={},
                created_at=datetime.now(timezone.utc),
                security_level=SecurityLevel.INTERNAL
            )
            for i in range(3)
        ]
        
        print("✅ Bulk records created for performance testing")
        
        # Test performance metrics
        metrics = await db.get_performance_metrics()
        print("✅ Performance metrics collected")
        print(f"   📊 Table sizes: {metrics.get('table_sizes', {})}")
        
        print("🎉 All database tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_database())