"""
🎯 Configuration Module Package
📦 Contains system configuration and settings management
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Description:
   This package handles all system configuration:
   - Application settings and environment variables
   - Database configuration
   - Security settings and feature flags
   - Deployment configurations
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
from functools import wraps

# Try to import validation libraries
try:
    from pydantic import BaseSettings, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

try:
    import marshmallow as mm
    from marshmallow import Schema, fields, validate, ValidationError
    MARSHMALLOW_AVAILABLE = True
except ImportError:
    MARSHMALLOW_AVAILABLE = False

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = metrics = None

# Prometheus imports
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Structured JSON logging configuration
class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        # Mask sensitive data
        self._mask_sensitive_data(log_entry)
        
        return json.dumps(log_entry)

    def _mask_sensitive_data(self, log_entry: Dict[str, Any]):
        """Mask sensitive information in logs"""
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential', 'api_key']
        for key in list(log_entry.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                log_entry[key] = '***MASKED***'

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Initialize OpenTelemetry if available
if OTEL_AVAILABLE:
    trace.set_tracer_provider(TracerProvider())
    metrics.set_meter_provider(MeterProvider())
    
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    
    # Create metrics
    config_loaded = meter.create_counter(
        "config_loaded_total",
        description="Total number of configuration loads"
    )
    config_errors = meter.create_counter(
        "config_errors_total", 
        description="Total number of configuration errors"
    )
    config_validation_errors = meter.create_counter(
        "config_validation_errors_total",
        description="Total number of configuration validation errors"
    )

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    CONFIG_LOADED_TOTAL = Counter(
        'config_loaded_total', 
        'Total configuration loads', 
        ['config_type']
    )
    CONFIG_ERRORS_TOTAL = Counter(
        'config_errors_total', 
        'Total configuration errors',
        ['error_type']
    )
    CONFIG_VALIDATION_ERRORS_TOTAL = Counter(
        'config_validation_errors_total',
        'Total configuration validation errors'
    )
    CONFIG_HEALTH = Gauge(
        'config_health_status',
        'Health status of configuration components',
        ['component']
    )

# Secrets Management Integration
class SecretsManager:
    """Manages secrets from various cloud providers and vaults"""
    
    @staticmethod
    def load_from_aws_secrets_manager(secret_name: str) -> Optional[Dict[str, Any]]:
        """Load secrets from AWS Secrets Manager"""
        try:
            # Implementation would use boto3
            # import boto3
            # client = boto3.client('secretsmanager')
            # response = client.get_secret_value(SecretId=secret_name)
            # return json.loads(response['SecretString'])
            logger.info(f"Loading secrets from AWS Secrets Manager: {secret_name}")
            return None
        except Exception as e:
            logger.error(f"AWS Secrets Manager error: {e}")
            return None
    
    @staticmethod
    def load_from_azure_key_vault(secret_name: str) -> Optional[Dict[str, Any]]:
        """Load secrets from Azure Key Vault"""
        try:
            # Implementation would use azure-keyvault-secrets
            logger.info(f"Loading secrets from Azure Key Vault: {secret_name}")
            return None
        except Exception as e:
            logger.error(f"Azure Key Vault error: {e}")
            return None
    
    @staticmethod
    def load_from_gcp_secret_manager(secret_name: str) -> Optional[Dict[str, Any]]:
        """Load secrets from GCP Secret Manager"""
        try:
            # Implementation would use google-cloud-secret-manager
            logger.info(f"Loading secrets from GCP Secret Manager: {secret_name}")
            return None
        except Exception as e:
            logger.error(f"GCP Secret Manager error: {e}")
            return None
    
    @staticmethod
    def load_from_hashicorp_vault(secret_path: str) -> Optional[Dict[str, Any]]:
        """Load secrets from HashiCorp Vault"""
        try:
            # Implementation would use hvac
            logger.info(f"Loading secrets from HashiCorp Vault: {secret_path}")
            return None
        except Exception as e:
            logger.error(f"HashiCorp Vault error: {e}")
            return None

    @staticmethod
    def get_secret(secret_name: str, default: str = "") -> str:
        """
        Unified method to get secrets from various sources
        Priority: Environment Variables -> Cloud Secrets -> Default
        """
        # First try environment variable
        env_value = os.getenv(secret_name)
        if env_value:
            return env_value
        
        # Then try cloud providers (simplified implementation)
        cloud_secrets = [
            SecretsManager.load_from_aws_secrets_manager,
            SecretsManager.load_from_azure_key_vault,
            SecretsManager.load_from_gcp_secret_manager,
            SecretsManager.load_from_hashicorp_vault
        ]
        
        for secret_loader in cloud_secrets:
            try:
                secrets = secret_loader(secret_name)
                if secrets and secret_name in secrets:
                    return secrets[secret_name]
            except Exception:
                continue
        
        # Return default if nothing found
        return default

# Environment Variable Management with Fallbacks
class EnvironmentConfig:
    def __init__(self):
        self.config = {"ml_engine": {"enabled": True}, "fusion_engine": {"enabled": True}, "quantum_engine": {"enabled": True}, "global": {"security_level": "classified_tier_1"}}
    def get(self, key, default=None):
        return self.config.get(key, default) if hasattr(self, "config") else default
    """Enhanced environment variable management with fallbacks and validation"""
    
    @staticmethod
    def get_env(key: str, default: Any = None, required: bool = False) -> Any:
        """
        Get environment variable with fallback and validation
        
        Args:
            key: Environment variable name
            default: Default value if not found
            required: Whether the variable is required
            
        Returns:
            Environment variable value or default
        """
        value = os.getenv(key)
        
        if value is None:
            if required and default is None:
                error_msg = f"Required environment variable {key} is not set"
                logger.error(error_msg)
                if PROMETHEUS_AVAILABLE:
                    CONFIG_ERRORS_TOTAL.labels(error_type='missing_required').inc()
                raise ValueError(error_msg)
            
            logger.warning(
                f"Environment variable {key} not found, using default",
                extra={'extra_fields': {'key': key, 'default_value': str(default)}}
            )
            return default
        
        # Log successful retrieval (masking sensitive data)
        log_value = value if not any(sensitive in key.lower() for sensitive in 
                                   ['password', 'secret', 'key', 'token']) else '***MASKED***'
        
        logger.debug(
            f"Loaded environment variable",
            extra={'extra_fields': {'key': key, 'value': log_value}}
        )
        
        return value
    
    @staticmethod
    def get_env_bool(key: str, default: bool = False) -> bool:
        """Get boolean environment variable"""
        value = EnvironmentConfig.get_env(key, default)
        if isinstance(value, bool):
            return value
        return str(value).lower() in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def get_env_int(key: str, default: int = 0) -> int:
        """Get integer environment variable"""
        value = EnvironmentConfig.get_env(key, default)
        try:
            return int(value)
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid integer value for {key}: {value}")
            if PROMETHEUS_AVAILABLE:
                CONFIG_ERRORS_TOTAL.labels(error_type='invalid_type').inc()
            return default
    
    @staticmethod
    def get_env_list(key: str, default: List[str] = None, separator: str = ',') -> List[str]:
        """Get list environment variable"""
        if default is None:
            default = []
        value = EnvironmentConfig.get_env(key, '')
        if not value:
            return default
        return [item.strip() for item in value.split(separator)]

# Configuration Validation
class ConfigValidator:
    """Configuration validation using Pydantic or Marshmallow"""
    
    @staticmethod
    def validate_with_pydantic(config_dict: Dict[str, Any], config_class: Any) -> Any:
        """Validate configuration using Pydantic"""
        if not PYDANTIC_AVAILABLE:
            logger.warning("Pydantic not available, skipping validation")
            return config_dict
        
        try:
            validated_config = config_class(**config_dict)
            logger.info("Configuration validated successfully with Pydantic")
            return validated_config
        except Exception as e:
            logger.error(f"Pydantic validation error: {e}")
            if PROMETHEUS_AVAILABLE:
                CONFIG_VALIDATION_ERRORS_TOTAL.inc()
            raise
    
    @staticmethod
    def validate_with_marshmallow(config_dict: Dict[str, Any], schema: Any) -> Dict[str, Any]:
        """Validate configuration using Marshmallow"""
        if not MARSHMALLOW_AVAILABLE:
            logger.warning("Marshmallow not available, skipping validation")
            return config_dict
        
        try:
            validated_config = schema().load(config_dict)
            logger.info("Configuration validated successfully with Marshmallow")
            return validated_config
        except ValidationError as e:
            logger.error(f"Marshmallow validation error: {e.messages}")
            if PROMETHEUS_AVAILABLE:
                CONFIG_VALIDATION_ERRORS_TOTAL.inc()
            raise

# Health Check and Monitoring
class ConfigHealthChecker:
    """Health checking for configuration components"""
    
    @staticmethod
    def check_database_config() -> Dict[str, Any]:
        """Check database configuration health"""
        try:
            # Try to import database config dynamically
            from . import database_config
            return {"status": "healthy", "component": "database_config"}
        except Exception as e:
            return {"status": "unhealthy", "component": "database_config", "error": str(e)}
    
    @staticmethod
    def check_security_config() -> Dict[str, Any]:
        """Check security configuration health"""
        try:
            # Try to import security config dynamically
            from . import security_config
            return {"status": "healthy", "component": "security_config"}
        except Exception as e:
            return {"status": "unhealthy", "component": "security_config", "error": str(e)}
    
    @staticmethod
    def check_feature_flags() -> Dict[str, Any]:
        """Check feature flags health"""
        try:
            # Try to import feature flags dynamically, but handle missing module gracefully
            from . import feature_flags
            return {"status": "healthy", "component": "feature_flags"}
        except ImportError:
            # Feature flags module is optional
            return {"status": "healthy", "component": "feature_flags", "note": "module_not_implemented"}
        except Exception as e:
            return {"status": "unhealthy", "component": "feature_flags", "error": str(e)}
    
    @staticmethod
    def check_settings() -> Dict[str, Any]:
        """Check main settings health"""
        try:
            from . import settings
            return {"status": "healthy", "component": "settings"}
        except Exception as e:
            return {"status": "unhealthy", "component": "settings", "error": str(e)}

def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for all configuration components
    
    Returns:
        Dict containing health status of all components
    """
    health_status = {
        "version": __version__,
        "overall_status": "healthy",
        "timestamp": time.time(),
        "components": {},
        "validation": {
            "pydantic_available": PYDANTIC_AVAILABLE,
            "marshmallow_available": MARSHMALLOW_AVAILABLE
        }
    }
    
    # Check all components
    components = {
        "settings": ConfigHealthChecker.check_settings,
        "database_config": ConfigHealthChecker.check_database_config,
        "security_config": ConfigHealthChecker.check_security_config,
        "feature_flags": ConfigHealthChecker.check_feature_flags,
    }
    
    unhealthy_components = []
    
    for name, check_func in components.items():
        try:
            component_status = check_func()
            health_status["components"][name] = component_status
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                status_value = 1 if component_status["status"] == "healthy" else 0
                CONFIG_HEALTH.labels(component=name).set(status_value)
            
            # Only consider unhealthy if it's not a missing but optional module
            if (component_status["status"] != "healthy" and 
                component_status.get("note") != "module_not_implemented"):
                unhealthy_components.append(name)
                
        except Exception as e:
            health_status["components"][name] = {
                "status": "error", 
                "error": str(e)
            }
            unhealthy_components.append(name)
    
    # Update overall status - be more tolerant about feature_flags
    if (unhealthy_components and 
        len(unhealthy_components) == 1 and 
        "feature_flags" in unhealthy_components):
        health_status["overall_status"] = "healthy"  # Feature flags are optional
    elif unhealthy_components:
        health_status["overall_status"] = "degraded"
    
    if all(comp["status"] in ["unhealthy", "error"] for comp in health_status["components"].values()):
        health_status["overall_status"] = "unhealthy"
    
    # Log health check result
    logger.info(
        "Configuration health check completed",
        extra={
            'extra_fields': {
                'overall_status': health_status["overall_status"],
                'unhealthy_components': unhealthy_components,
                'total_components': len(components)
            }
        }
    )
    
    return health_status

# Monitoring initialization
def initialize_monitoring(enable_prometheus: bool = True, prometheus_port: int = 9090):
    """Initialize monitoring and observability components"""
    
    if enable_prometheus and PROMETHEUS_AVAILABLE:
        try:
            start_http_server(prometheus_port)
            logger.info(
                f"Prometheus metrics server started on port {prometheus_port}",
                extra={'extra_fields': {'port': prometheus_port}}
            )
        except Exception as e:
            logger.error(
                f"Failed to start Prometheus server: {e}",
                extra={'extra_fields': {'error': str(e)}}
            )

# Package imports with error handling
try:
    from . import settings
    CONFIG_LOADED = True
    if PROMETHEUS_AVAILABLE:
        CONFIG_LOADED_TOTAL.labels(config_type='settings').inc()
except ImportError as e:
    logger.error(f"Failed to import settings: {e}")
    CONFIG_LOADED = False

try:
    from . import database_config
    if PROMETHEUS_AVAILABLE:
        CONFIG_LOADED_TOTAL.labels(config_type='database_config').inc()
except ImportError as e:
    logger.error(f"Failed to import database_config: {e}")

try:
    from . import security_config
    if PROMETHEUS_AVAILABLE:
        CONFIG_LOADED_TOTAL.labels(config_type='security_config').inc()
except ImportError as e:
    logger.error(f"Failed to import security_config: {e}")

# Feature flags is optional - handle gracefully
try:
    from . import feature_flags
    if PROMETHEUS_AVAILABLE:
        CONFIG_LOADED_TOTAL.labels(config_type='feature_flags').inc()
    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False
    logger.info("Feature flags module not available - this is optional")

__all__ = [
    "settings", 
    "database_config", 
    "security_config", 
    "EnvironmentConfig",
    "SecretsManager", 
    "ConfigValidator",
    "health_check",
    "initialize_monitoring"
]

# Add feature_flags to exports only if available
if FEATURE_FLAGS_AVAILABLE:
    __all__.append("feature_flags")

__version__ = "2.0.0"

# Auto-initialize monitoring if enabled (use different port to avoid conflicts)
if os.getenv('CONFIG_ENABLE_MONITORING', 'false').lower() == 'true':
    initialize_monitoring(prometheus_port=9090)

# Perform initial health check
initial_health = health_check()
logger.info(
    f"Configuration package v{__version__} initialized",
    extra={
        'extra_fields': {
            'overall_health': initial_health["overall_status"],
            'pydantic_available': PYDANTIC_AVAILABLE,
            'marshmallow_available': MARSHMALLOW_AVAILABLE,
            'otel_available': OTEL_AVAILABLE,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'feature_flags_available': FEATURE_FLAGS_AVAILABLE
        }
    }
)

# SentinelConfig for engine compatibility
from sentinel_config import SentinelConfig, sentinel_config
