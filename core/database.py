"""
Production Database Layer with SQLAlchemy
Enhanced version with async support, comprehensive security, and enterprise features
"""
import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator
import uuid

# SQLAlchemy imports
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, DateTime, 
    Boolean, Text, JSON, Index, Text, LargeBinary, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text

# Async support
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    ASYNC_SUPPORT = True
except ImportError:
    ASYNC_SUPPORT = False
    print("⚠️ Async SQLAlchemy not available - install asyncpg")

# Security
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config.production import config

Base = declarative_base()

class ScanRecord(Base):
    """Production scan records table with comprehensive indexing"""
    __tablename__ = 'scan_records'
    
    id = Column(Integer, primary_key=True)
    scan_id = Column(String(64), unique=True, nullable=False, index=True)
    model_path = Column(Text, nullable=False)
    model_hash = Column(String(64), index=True)
    model_format = Column(String(20))
    threat_score = Column(Float, index=True)
    threat_level = Column(String(20), index=True)
    confidence = Column(Float)
    scan_duration = Column(Float)
    file_size = Column(Integer)
    timestamp = Column(DateTime, index=True)
    encrypted_data = Column(Text)
    signature_verified = Column(Boolean)
    gpu_used = Column(Boolean)
    qpbi_score = Column(Float)
    entropy_score = Column(Float)
    mahalanobis_distance = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Composite indexes for common query patterns
    __table_args__ = (
        Index('idx_threat_level_score', 'threat_level', 'threat_score'),
        Index('idx_timestamp_threat', 'timestamp', 'threat_level'),
        Index('idx_model_hash_format', 'model_hash', 'model_format'),
    )

class ScanAnalytics(Base):
    """Detailed scan analytics with performance tracking"""
    __tablename__ = 'scan_analytics'
    
    id = Column(Integer, primary_key=True)
    scan_id = Column(String(64), nullable=False, index=True)
    component_name = Column(String(50), index=True)
    threat_score = Column(Float)
    details_json = Column(JSON)
    processing_time = Column(Float)
    status = Column(String(20), default='completed')  # completed, failed, timeout
    error_message = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_scan_component', 'scan_id', 'component_name'),
        Index('idx_component_performance', 'component_name', 'processing_time'),
    )

class AuditLog(Base):
    """Comprehensive audit trail for security compliance"""
    __tablename__ = 'audit_log'
    
    id = Column(Integer, primary_key=True)
    action = Column(String(50), nullable=False, index=True)
    actor = Column(String(100), index=True)
    resource_type = Column(String(50))
    resource_id = Column(String(64))
    details = Column(JSON)
    ip_address = Column(String(45))  # Support IPv6
    user_agent = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_audit_actor_action', 'actor', 'action'),
        Index('idx_audit_timestamp', 'created_at'),
    )

class SystemMetrics(Base):
    """System performance and health metrics"""
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_type = Column(String(50), nullable=False, index=True)
    metric_value = Column(Float)
    details = Column(JSON)
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_metrics_type_time', 'metric_type', 'recorded_at'),
    )

class DatabaseManager:
    """
    Enterprise-grade database manager with async support
    Features:
    - Connection pooling and health checks
    - Comprehensive encryption and security
    - Async/await support for high concurrency
    - Audit logging and metrics collection
    - Container-ready configuration
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self._init_database()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive database logging"""
        logger = logging.getLogger("database_manager")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _init_database(self):
        """Initialize database engines with production settings"""
        try:
            # Sync engine for traditional operations
            self.engine = create_engine(
                self._build_database_url(),
                poolclass=QueuePool,
                pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
                max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections every hour
                echo=bool(os.getenv('DB_ECHO_SQL', False)),  # Log SQL in development
                connect_args={
                    "connect_timeout": 30,
                    "application_name": "ai_sentinel_scanner"
                }
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
            
            # Async engine for high-concurrency operations
            if ASYNC_SUPPORT:
                async_database_url = self._build_database_url().replace(
                    'postgresql://', 'postgresql+asyncpg://'
                )
                self.async_engine = create_async_engine(
                    async_database_url,
                    poolclass=QueuePool,
                    pool_size=int(os.getenv('DB_ASYNC_POOL_SIZE', '20')),
                    max_overflow=int(os.getenv('DB_ASYNC_MAX_OVERFLOW', '30')),
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=bool(os.getenv('DB_ECHO_SQL', False)),
                )
                
                self.AsyncSessionLocal = async_sessionmaker(
                    self.async_engine,
                    expire_on_commit=False,
                    class_=AsyncSession
                )
            
            self.logger.info("✅ Production database engines initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def _build_database_url(self) -> str:
        """Build database URL from environment variables or config"""
        if hasattr(config, 'database_url') and config.database_url:
            return config.database_url
        
        # Container-ready configuration
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_user = os.getenv('DB_USER', 'sentinel_user')
        db_pass = os.getenv('DB_PASS', '')
        db_name = os.getenv('DB_NAME', 'ai_sentinel')
        
        if not db_pass:
            db_pass = config.get_secret('DB_PASSWORD', 'default_pass')
        
        return f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    
    def init_db(self):
        """Initialize database tables and indexes"""
        try:
            Base.metadata.create_all(bind=self.engine)
            
            # Create additional indexes
            with self.engine.connect() as conn:
                # Index for frequently queried date ranges
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_scan_records_recent 
                    ON scan_records (created_at DESC) 
                    WHERE created_at > NOW() - INTERVAL '30 days'
                """))
                
                # Partial index for high-threat records
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_high_threat_records 
                    ON scan_records (threat_score, created_at) 
                    WHERE threat_score > 0.7
                """))
            
            self.logger.info("✅ Production database schema initialized with optimized indexes")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    @contextmanager
    def get_session(self):
        """Database session context manager with error handling"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    if ASYNC_SUPPORT:
        @asynccontextmanager
        async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
            """Async database session context manager"""
            session = self.AsyncSessionLocal()
            try:
                yield session
                await session.commit()
            except Exception as e:
                await session.rollback()
                self.logger.error(f"Async database session error: {e}")
                raise
            finally:
                await session.close()
    
    def _encrypt_sensitive_data(self, data: dict) -> str:
        """Encrypt sensitive data with Fernet"""
        try:
            fernet = Fernet(config.encryption_key)
            encrypted_data = fernet.encrypt(
                json.dumps(data).encode()
            )
            return encrypted_data.decode()
        except InvalidToken as e:
            self.logger.error(f"Encryption failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected encryption error: {e}")
            raise
    
    def _decrypt_sensitive_data(self, encrypted_data: str) -> dict:
        """Decrypt sensitive data"""
        try:
            fernet = Fernet(config.encryption_key)
            decrypted_data = fernet.decrypt(encrypted_data.encode())
            return json.loads(decrypted_data.decode())
        except InvalidToken as e:
            self.logger.error(f"Decryption failed - invalid token: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Decryption error: {e}")
            return {}
    
    def _validate_scan_data(self, scan_data: dict) -> bool:
        """Validate scan data before saving"""
        required_fields = ['scan_id', 'model_path', 'file_hash']
        
        for field in required_fields:
            if not scan_data.get(field):
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Validate threat score range
        threat_score = scan_data.get('threat_score', 0)
        if not 0 <= threat_score <= 1:
            self.logger.error(f"Invalid threat score: {threat_score}")
            return False
        
        return True
    
    def save_scan_record(self, scan_data: dict) -> bool:
        """Save scan record with comprehensive validation and encryption"""
        if not self._validate_scan_data(scan_data):
            return False
        
        try:
            # Encrypt sensitive data
            sensitive_data = {
                'original_path': scan_data.get('model_path'),
                'extracted_features': scan_data.get('extracted_features', {}),
                'user_metadata': scan_data.get('user_metadata', {})
            }
            
            encrypted_data = self._encrypt_sensitive_data(sensitive_data)
            
            with self.get_session() as session:
                # Create main scan record
                record = ScanRecord(
                    scan_id=scan_data.get('scan_id'),
                    model_path=scan_data.get('model_path'),
                    model_hash=scan_data.get('file_hash'),
                    model_format=scan_data.get('model_format', 'unknown'),
                    threat_score=scan_data.get('threat_score', 0),
                    threat_level=scan_data.get('threat_level', 'unknown'),
                    confidence=scan_data.get('confidence_level', 0),
                    scan_duration=scan_data.get('scan_duration', 0),
                    file_size=scan_data.get('file_size', 0),
                    timestamp=datetime.fromisoformat(
                        scan_data.get('timestamp', datetime.utcnow().isoformat()).replace('Z', '+00:00')
                    ),
                    encrypted_data=encrypted_data,
                    signature_verified=scan_data.get('signature_verified', False),
                    gpu_used=scan_data.get('gpu_accelerated', False),
                    qpbi_score=scan_data.get('qpbi_score', 0),
                    entropy_score=scan_data.get('entropy_score', 0),
                    mahalanobis_distance=scan_data.get('mahalanobis_distance', 0)
                )
                session.add(record)
                
                # Save component analytics
                components = scan_data.get('scan_components', {})
                for comp_name, comp_data in components.items():
                    analytics = ScanAnalytics(
                        scan_id=scan_data.get('scan_id'),
                        component_name=comp_name,
                        threat_score=comp_data.get('score', 0),
                        details_json=comp_data.get('details', {}),
                        processing_time=comp_data.get('scan_time', 0),
                        status=comp_data.get('status', 'completed'),
                        error_message=comp_data.get('error_message')
                    )
                    session.add(analytics)
                
                # Log the audit trail
                audit_log = AuditLog(
                    action='SCAN_CREATED',
                    actor=scan_data.get('actor', 'system'),
                    resource_type='scan_record',
                    resource_id=scan_data.get('scan_id'),
                    details={'threat_level': scan_data.get('threat_level')},
                    ip_address=scan_data.get('ip_address'),
                    user_agent=scan_data.get('user_agent')
                )
                session.add(audit_log)
            
            self.logger.info(f"✅ Scan record saved: {scan_data.get('scan_id')}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save scan record: {e}")
            return False
    
    async def save_scan_record_async(self, scan_data: dict) -> bool:
        """Async version of save_scan_record for high-concurrency environments"""
        if not ASYNC_SUPPORT:
            self.logger.warning("Async not supported, falling back to sync")
            return self.save_scan_record(scan_data)
        
        if not self._validate_scan_data(scan_data):
            return False
        
        try:
            encrypted_data = self._encrypt_sensitive_data(
                scan_data.get('sensitive_data', {})
            )
            
            async with self.get_async_session() as session:
                record = ScanRecord(
                    scan_id=scan_data.get('scan_id'),
                    model_path=scan_data.get('model_path'),
                    model_hash=scan_data.get('file_hash'),
                    model_format=scan_data.get('model_format', 'unknown'),
                    threat_score=scan_data.get('threat_score', 0),
                    threat_level=scan_data.get('threat_level', 'unknown'),
                    confidence=scan_data.get('confidence_level', 0),
                    scan_duration=scan_data.get('scan_duration', 0),
                    file_size=scan_data.get('file_size', 0),
                    timestamp=datetime.fromisoformat(
                        scan_data.get('timestamp', datetime.utcnow().isoformat()).replace('Z', '+00:00')
                    ),
                    encrypted_data=encrypted_data,
                    signature_verified=scan_data.get('signature_verified', False),
                    gpu_used=scan_data.get('gpu_accelerated', False),
                    qpbi_score=scan_data.get('qpbi_score', 0),
                    entropy_score=scan_data.get('entropy_score', 0),
                    mahalanobis_distance=scan_data.get('mahalanobis_distance', 0)
                )
                session.add(record)
                await session.commit()
            
            self.logger.info(f"✅ Async scan record saved: {scan_data.get('scan_id')}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Async scan record save failed: {e}")
            return False
    
    def verify_record_integrity(self, scan_id: str) -> bool:
        """Verify data integrity and encryption for a scan record"""
        try:
            with self.get_session() as session:
                record = session.query(ScanRecord).filter_by(scan_id=scan_id).first()
                if not record:
                    self.logger.warning(f"Record not found: {scan_id}")
                    return False
                
                # Verify essential fields
                if not all([record.model_hash, record.encrypted_data, record.scan_id]):
                    self.logger.error(f"Missing essential fields for record: {scan_id}")
                    return False
                
                # Test decryption
                try:
                    decrypted_data = self._decrypt_sensitive_data(record.encrypted_data)
                    if not isinstance(decrypted_data, dict):
                        self.logger.error(f"Decrypted data is not valid JSON for: {scan_id}")
                        return False
                except Exception as e:
                    self.logger.error(f"Decryption test failed for {scan_id}: {e}")
                    return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"Integrity check failed for {scan_id}: {e}")
            return False
    
    def get_scan_statistics(self, hours: int = 24) -> Dict:
        """Get scan statistics for monitoring dashboard"""
        try:
            with self.get_session() as session:
                # Basic counts
                total_scans = session.query(ScanRecord).count()
                
                recent_scans = session.query(ScanRecord).filter(
                    ScanRecord.created_at >= datetime.utcnow() - timedelta(hours=hours)
                ).count()
                
                threat_distribution = session.query(
                    ScanRecord.threat_level,
                    func.count(ScanRecord.id)
                ).group_by(ScanRecord.threat_level).all()
                
                avg_processing_time = session.query(
                    func.avg(ScanAnalytics.processing_time)
                ).scalar() or 0
                
                return {
                    'total_scans': total_scans,
                    'recent_scans': recent_scans,
                    'threat_distribution': dict(threat_distribution),
                    'avg_processing_time': round(avg_processing_time, 2),
                    'timeframe_hours': hours
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get scan statistics: {e}")
            return {}
    
    def log_system_metric(self, metric_type: str, value: float, details: Dict = None):
        """Log system performance metric"""
        try:
            with self.get_session() as session:
                metric = SystemMetrics(
                    metric_type=metric_type,
                    metric_value=value,
                    details=details or {},
                    recorded_at=datetime.utcnow()
                )
                session.add(metric)
            
            self.logger.debug(f"System metric logged: {metric_type} = {value}")
        except Exception as e:
            self.logger.error(f"Failed to log system metric: {e}")

# Global database instance with health check
db_manager = DatabaseManager()

# Health check function
def database_health_check() -> Dict:
    """Comprehensive database health check"""
    try:
        with db_manager.get_session() as session:
            # Test connection and basic query
            result = session.execute(text("SELECT 1")).scalar()
            
            # Check table counts
            scan_count = session.query(ScanRecord).count()
            analytics_count = session.query(ScanAnalytics).count()
            
            return {
                'status': 'healthy',
                'database': 'connected',
                'scan_records': scan_count,
                'analytics_records': analytics_count,
                'timestamp': datetime.utcnow().isoformat()
            }
    except Exception as e:
        db_manager.logger.error(f"Database health check failed: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }