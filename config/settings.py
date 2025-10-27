"""
🎯 System Configuration Settings
📦 Centralized configuration management for AI Model Sentinel
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Business Logic: 
   - Manages all system configuration in one place
   - Supports different environments (dev, staging, prod)
   - Provides type-safe configuration access
   - Enables feature flags and runtime configuration
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
from datetime import datetime, timezone

# Try to import security and secret management libraries
try:
    from decouple import config as env_config
    DECOUPLE_AVAILABLE = True
except ImportError:
    DECOUPLE_AVAILABLE = False

try:
    import hvac  # HashiCorp Vault
    VAULT_AVAILABLE = True
except ImportError:
    VAULT_AVAILABLE = False

# Import models for integration
try:
    from audit_models import AuditLog, AuditAction
    from security_models import SecurityEvent, EventSeverity
    from user_models import UserRole, Tenant
except ImportError:
    # Fallback definitions if imports fail
    class AuditAction(str, Enum):
        CONFIG_CHANGE = "config_change"
    
    class EventSeverity(str, Enum):
        HIGH = "high"
    
    class UserRole(str, Enum):
        TENANT_ADMIN = "tenant_admin"


class Environment(str, Enum):
    """🌍 Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseType(str, Enum):
    """🗄️ Supported database types"""
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    MYSQL = "mysql"


class StorageProvider(str, Enum):
    """☁️ Supported cloud storage providers"""
    AWS_S3 = "aws_s3"
    AZURE_BLOB = "azure_blob"
    GCP_STORAGE = "gcp_storage"
    BACKBLAZE_B2 = "backblaze_b2"
    LOCAL = "local"


class AnalyticsProvider(str, Enum):
    """📊 Supported analytics providers"""
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"


@dataclass
class SecretManager:
    """🔐 Secure secret management with multiple backends"""
    
    @staticmethod
    def get_secret(key: str, default: str = "") -> str:
        """Get secret from secure source with fallback"""
        # Try environment variables first
        value = os.getenv(key, default)
        
        if value == default and DECOUPLE_AVAILABLE:
            value = env_config(key, default=default)
        
        # Try HashiCorp Vault if available
        if value == default and VAULT_AVAILABLE:
            value = SecretManager._get_from_vault(key, default)
        
        return value
    
    @staticmethod
    def _get_from_vault(key: str, default: str) -> str:
        """Get secret from HashiCorp Vault"""
        try:
            # Vault configuration would go here
            # client = hvac.Client(url=os.getenv('VAULT_URL'))
            # secret = client.secrets.kv.read_secret_version(path=key)
            # return secret['data']['data']['value']
            return default
        except Exception:
            return default


@dataclass
class DatabaseConfig:
    """🗄️ Database configuration settings"""
    
    # Database type and connection
    database_type: DatabaseType = field(default=DatabaseType.POSTGRESQL)
    host: str = field(default="localhost")
    port: int = field(default=5432)
    database: str = field(default="ai_sentinel")
    username: str = field(default="postgres")
    password: str = field(default="")
    
    # Connection pooling
    pool_min_size: int = field(default=5)
    pool_max_size: int = field(default=20)
    pool_timeout: int = field(default=30)
    
    # SSL and security
    ssl_mode: str = field(default="require")  # Enforce SSL in production
    ssl_cert: Optional[str] = field(default=None)
    ssl_key: Optional[str] = field(default=None)
    ssl_ca: Optional[str] = field(default=None)
    
    # Tenant isolation
    tenant_prefix: bool = field(default=True)
    
    def get_connection_string(self, tenant_id: Optional[str] = None) -> str:
        """Generate database connection string with tenant support"""
        db_name = self.database
        if tenant_id and self.tenant_prefix:
            db_name = f"{tenant_id}_{self.database}"
        
        if self.database_type == DatabaseType.POSTGRESQL:
            ssl_params = f"?sslmode={self.ssl_mode}" if self.ssl_mode != "prefer" else ""
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{db_name}{ssl_params}"
        elif self.database_type == DatabaseType.SQLITE:
            return f"sqlite:///./storage/{db_name}.db"
        else:
            return f"{self.database_type.value}://{self.username}:{self.password}@{self.host}:{self.port}/{db_name}"
    
    def load_from_env(self):
        """Load database config from environment variables"""
        self.host = SecretManager.get_secret("DB_HOST", "localhost")
        self.port = int(SecretManager.get_secret("DB_PORT", "5432"))
        self.database = SecretManager.get_secret("DB_NAME", "ai_sentinel")
        self.username = SecretManager.get_secret("DB_USER", "postgres")
        self.password = SecretManager.get_secret("DB_PASSWORD", "")
        
        # Enforce SSL in production
        env = os.getenv("ENVIRONMENT", "development")
        if env == "production":
            self.ssl_mode = "require"


@dataclass
class SecurityConfig:
    """🔐 Security and authentication configuration"""
    
    # JWT Settings with key rotation support
    jwt_secret: str = field(default="")
    jwt_algorithm: str = field(default="HS256")
    jwt_expiration_minutes: int = field(default=60)
    jwt_rotation_days: int = field(default=30)
    previous_jwt_secrets: List[str] = field(default_factory=list)
    
    # Password policy with breach detection
    password_min_length: int = field(default=12)
    password_require_uppercase: bool = field(default=True)
    password_require_lowercase: bool = field(default=True)
    password_require_numbers: bool = field(default=True)
    password_require_special: bool = field(default=True)
    password_breach_check: bool = field(default=True)
    
    # Rate limiting with tenant isolation
    rate_limit_requests: int = field(default=100)
    rate_limit_window: int = field(default=900)  # 15 minutes
    tenant_rate_limits: Dict[str, int] = field(default_factory=dict)
    
    # Encryption with key rotation
    encryption_key: str = field(default="")
    encryption_algorithm: str = field(default="AES-256-GCM")
    key_rotation_days: int = field(default=90)
    
    # Session management
    session_timeout_minutes: int = field(default=480)  # 8 hours
    max_concurrent_sessions: int = field(default=5)
    
    # HTTPS enforcement
    require_https: bool = field(default=True)
    hsts_max_age: int = field(default=31536000)  # 1 year
    
    def load_from_env(self):
        """Load security config from environment variables"""
        self.jwt_secret = SecretManager.get_secret("JWT_SECRET", "")
        self.encryption_key = SecretManager.get_secret("ENCRYPTION_KEY", "")
        
        # Enforce stronger settings in production
        env = os.getenv("ENVIRONMENT", "development")
        if env == "production":
            self.password_min_length = 14
            self.require_https = True
            self.jwt_expiration_minutes = 30  # Shorter sessions in production
    
    def rotate_encryption_key(self) -> bool:
        """Rotate encryption key and keep previous for decryption"""
        try:
            if self.encryption_key:
                self.previous_jwt_secrets.append(self.encryption_key)
            
            # Generate new key (in production, use proper key generation)
            new_key = hashlib.sha256(os.urandom(32)).hexdigest()
            self.encryption_key = new_key
            return True
        except Exception:
            return False


@dataclass
class APIConfig:
    """🌐 API and web server configuration"""
    
    # Server settings
    host: str = field(default="0.0.0.0")
    port: int = field(default=8000)
    debug: bool = field(default=False)
    reload: bool = field(default=False)
    
    # HTTPS configuration
    ssl_certfile: Optional[str] = field(default=None)
    ssl_keyfile: Optional[str] = field(default=None)
    
    # CORS settings with tenant origins
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["*"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    tenant_cors_origins: Dict[str, List[str]] = field(default_factory=dict)
    
    # API documentation
    docs_enabled: bool = field(default=True)
    docs_path: str = field(default="/docs")
    redoc_path: str = field(default="/redoc")
    
    # Rate limiting with tenant isolation
    api_rate_limit: str = field(default="100/minute")
    tenant_rate_limits: Dict[str, str] = field(default_factory=dict)
    max_request_size: str = field(default="10MB")
    
    def load_from_env(self):
        """Load API config from environment variables"""
        self.host = os.getenv("API_HOST", "0.0.0.0")
        self.port = int(os.getenv("API_PORT", "8000"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # Enable HTTPS in production
        env = os.getenv("ENVIRONMENT", "development")
        if env == "production":
            self.ssl_certfile = os.getenv("SSL_CERT_FILE")
            self.ssl_keyfile = os.getenv("SSL_KEY_FILE")


@dataclass
class AnalyticsConfig:
    """📊 Analytics and monitoring configuration"""
    
    # Provider configuration
    provider: AnalyticsProvider = field(default=AnalyticsProvider.SNOWFLAKE)
    
    # Snowflake configuration
    snowflake_account: str = field(default="")
    snowflake_user: str = field(default="")
    snowflake_password: str = field(default="")
    snowflake_database: str = field(default="AI_SENTINEL")
    snowflake_warehouse: str = field(default="COMPUTE_WH")
    snowflake_schema: str = field(default="PUBLIC")
    
    # Google BigQuery
    bigquery_dataset: str = field(default="ai_sentinel")
    bigquery_credentials_path: str = field(default="")
    
    # Redis for real-time analytics
    redis_url: str = field(default="redis://localhost:6379")
    
    # Data retention with tenant isolation
    data_retention_days: int = field(default=365)
    analytics_retention_days: int = field(default=730)  # 2 years
    tenant_retention_policies: Dict[str, int] = field(default_factory=dict)
    
    # Performance
    batch_size: int = field(default=1000)
    max_concurrent_queries: int = field(default=10)
    query_timeout: int = field(default=300)  # 5 minutes
    
    def load_from_env(self):
        """Load analytics config from environment variables"""
        self.snowflake_account = SecretManager.get_secret("SNOWFLAKE_ACCOUNT", "")
        self.snowflake_user = SecretManager.get_secret("SNOWFLAKE_USER", "")
        self.snowflake_password = SecretManager.get_secret("SNOWFLAKE_PASSWORD", "")
        self.redis_url = SecretManager.get_secret("REDIS_URL", "redis://localhost:6379")


@dataclass
class StorageConfig:
    """💾 Cloud storage configuration"""
    
    # Primary storage provider
    primary_provider: StorageProvider = field(default=StorageProvider.LOCAL)
    failover_providers: List[StorageProvider] = field(default_factory=list)
    
    # AWS S3
    aws_access_key_id: str = field(default="")
    aws_secret_access_key: str = field(default="")
    aws_region: str = field(default="us-east-1")
    s3_bucket: str = field(default="ai-sentinel-backups")
    
    # Backblaze B2 Configuration
    b2_account_id: str = field(default="")
    b2_application_key: str = field(default="")
    b2_bucket: str = field(default="ai-sentinel-backups")
    b2_endpoint: str = field(default="s3.us-east-005.backblazeb2.com")
    
    # Azure Blob Storage
    azure_connection_string: str = field(default="")
    azure_container: str = field(default="backups")
    
    # Google Cloud Storage
    gcp_project_id: str = field(default="")
    gcp_bucket: str = field(default="ai-sentinel-backups")
    gcp_credentials_path: str = field(default="")
    
    # Local storage with tenant isolation
    local_storage_path: str = field(default="./storage")
    max_local_storage_gb: int = field(default=50)
    tenant_storage_quotas: Dict[str, int] = field(default_factory=dict)  # GB per tenant
    
    # Backup configuration
    auto_backup_enabled: bool = field(default=True)
    backup_interval_hours: int = field(default=24)
    backup_retention_days: int = field(default=30)
    
    def load_from_env(self):
        """Load storage config from environment variables"""
        self.aws_access_key_id = SecretManager.get_secret("AWS_ACCESS_KEY_ID", "")
        self.aws_secret_access_key = SecretManager.get_secret("AWS_SECRET_ACCESS_KEY", "")
        self.s3_bucket = os.getenv("S3_BUCKET", "ai-sentinel-backups")
        
        # Backblaze B2 configuration from your provided info
        self.b2_bucket = "ai-sentinel-backups"
        self.b2_endpoint = "s3.us-east-005.backblazeb2.com"
        
        # Set primary provider based on available credentials
        if self.aws_access_key_id and self.aws_secret_access_key:
            self.primary_provider = StorageProvider.AWS_S3
        elif self.b2_account_id and self.b2_application_key:
            self.primary_provider = StorageProvider.BACKBLAZE_B2
        else:
            self.primary_provider = StorageProvider.LOCAL


@dataclass
class TenantSpecificConfig:
    """🏢 Tenant-specific configuration overrides"""
    
    tenant_id: str
    features: Dict[str, bool] = field(default_factory=dict)
    storage_quota_gb: int = field(default=10)
    rate_limit_requests: int = field(default=100)
    data_retention_days: int = field(default=365)
    cors_origins: List[str] = field(default_factory=list)
    
    # Tenant-specific feature flags
    enabled_modules: List[str] = field(default_factory=lambda: [
        "security_scanning", "threat_intelligence", "compliance_reporting"
    ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "tenant_id": self.tenant_id,
            "features": self.features,
            "storage_quota_gb": self.storage_quota_gb,
            "rate_limit_requests": self.rate_limit_requests,
            "data_retention_days": self.data_retention_days,
            "cors_origins": self.cors_origins,
            "enabled_modules": self.enabled_modules
        }


@dataclass
class SystemSettings:
    """
    ⚙️ Main system configuration container
    💡 Centralized settings management for the entire application
    """
    
    # Environment
    environment: Environment = field(default=Environment.DEVELOPMENT)
    app_name: str = field(default="AI Model Sentinel")
    app_version: str = field(default="2.0.0")
    
    # Component configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    api: APIConfig = field(default_factory=APIConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Tenant-specific configurations
    tenant_configs: Dict[str, TenantSpecificConfig] = field(default_factory=dict)
    
    # Global feature flags
    features: Dict[str, bool] = field(default_factory=lambda: {
        "multi_tenant": True,
        "big_data_analytics": True,
        "real_time_monitoring": True,
        "advanced_threat_detection": True,
        "compliance_reporting": True,
        "api_rate_limiting": True,
        "auto_backup": True,
        "key_rotation": True,
        "breach_detection": True
    })
    
    # Logging with tenant isolation
    log_level: str = field(default="INFO")
    log_format: str = field(default="json")
    tenant_log_levels: Dict[str, str] = field(default_factory=dict)
    
    # Runtime configuration tracking
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    config_version: str = field(default="1.0")
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # Environment
        env_value = os.getenv("ENVIRONMENT", "development")
        self.environment = Environment(env_value)
        
        # Load component configurations
        self.database.load_from_env()
        self.security.load_from_env()
        self.api.load_from_env()
        self.analytics.load_from_env()
        self.storage.load_from_env()
        
        # Update last modified timestamp
        self.last_updated = datetime.now(timezone.utc)
        
        logging.info(f"✅ Settings initialized for {self.environment.value} environment")
    
    def get_tenant_config(self, tenant_id: str) -> TenantSpecificConfig:
        """Get tenant-specific configuration with fallback to defaults"""
        if tenant_id in self.tenant_configs:
            return self.tenant_configs[tenant_id]
        
        # Create default tenant config
        default_config = TenantSpecificConfig(tenant_id=tenant_id)
        self.tenant_configs[tenant_id] = default_config
        return default_config
    
    def update_tenant_config(self, tenant_id: str, config: TenantSpecificConfig) -> bool:
        """Update tenant-specific configuration"""
        try:
            self.tenant_configs[tenant_id] = config
            self.last_updated = datetime.now(timezone.utc)
            
            # Log configuration change for audit
            self._log_config_change(f"tenant_config_update:{tenant_id}", "success")
            return True
        except Exception as e:
            self._log_config_change(f"tenant_config_update:{tenant_id}", "failed", str(e))
            return False
    
    def is_feature_enabled(self, feature: str, tenant_id: Optional[str] = None) -> bool:
        """Check if a feature is enabled (with tenant override)"""
        # Check tenant-specific feature flag first
        if tenant_id and tenant_id in self.tenant_configs:
            tenant_features = self.tenant_configs[tenant_id].features
            if feature in tenant_features:
                return tenant_features[feature]
        
        # Fall back to global feature flag
        return self.features.get(feature, False)
    
    def get_tenant_log_level(self, tenant_id: str) -> str:
        """Get log level for specific tenant"""
        return self.tenant_log_levels.get(tenant_id, self.log_level)
    
    def _log_config_change(self, action: str, status: str, error: str = ""):
        """Log configuration changes for audit purposes"""
        # This would integrate with AuditLog and SecurityEvent models
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "status": status,
            "error": error,
            "config_version": self.config_version
        }
        
        # In production, this would create proper AuditLog and SecurityEvent records
        if status == "failed" and error:
            logging.error(f"Configuration change failed: {action} - {error}")
        else:
            logging.info(f"Configuration change: {action} - {status}")


# Global settings instance with lazy initialization
_settings_instance: Optional[SystemSettings] = None


def get_settings() -> SystemSettings:
    """Get or initialize the global settings instance"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = SystemSettings()
        _settings_instance.load_from_env()
    return _settings_instance


def initialize_settings() -> SystemSettings:
    """Initialize settings from environment variables"""
    return get_settings()


# Initialize settings when module is imported
settings = initialize_settings()