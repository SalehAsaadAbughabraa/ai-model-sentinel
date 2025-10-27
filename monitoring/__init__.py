"""
🎯 Storage Module Package
📦 Contains cloud storage and caching components
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Description:
   This package handles all storage and caching functionality:
   - Multi-cloud storage (AWS S3, Azure, GCP)
   - Caching layers (Redis, in-memory)
   - Backup and recovery management
   - File and object storage operations
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
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
        sensitive_keys = ['password', 'secret', 'key', 'token', 'credential']
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
    storage_requests = meter.create_counter(
        "storage_requests_total",
        description="Total number of storage requests"
    )
    cache_hits = meter.create_counter(
        "cache_hits_total", 
        description="Total number of cache hits"
    )
    backup_failures = meter.create_counter(
        "backup_failures_total",
        description="Total number of backup failures"
    )
    request_duration = meter.create_histogram(
        "request_duration_seconds",
        description="Duration of storage requests"
    )

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    STORAGE_REQUESTS_TOTAL = Counter(
        'storage_requests_total', 
        'Total storage requests', 
        ['provider', 'operation']
    )
    CACHE_HITS_TOTAL = Counter(
        'cache_hits_total', 
        'Total cache hits', 
        ['cache_type']
    )
    BACKUP_FAILURES_TOTAL = Counter(
        'backup_failures_total', 
        'Total backup failures'
    )
    REQUEST_DURATION = Histogram(
        'request_duration_seconds',
        'Request duration in seconds',
        ['provider', 'operation']
    )
    STORAGE_HEALTH = Gauge(
        'storage_health_status',
        'Health status of storage providers',
        ['provider']
    )

# Encryption utilities
class SecurityManager:
    """Handles encryption and security compliance"""
    
    @staticmethod
    def encrypt_data(data: bytes, key: Optional[str] = None) -> bytes:
        """Encrypt data using AES256 or KMS"""
        # Implementation would use cryptography library or AWS KMS
        # Placeholder for actual encryption logic
        return data
    
    @staticmethod
    def decrypt_data(encrypted_data: bytes, key: Optional[str] = None) -> bytes:
        """Decrypt data"""
        # Placeholder for actual decryption logic
        return encrypted_data
    
    @staticmethod
    def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in dictionaries"""
        masked_data = data.copy()
        sensitive_fields = ['password', 'secret', 'api_key', 'access_key', 'secret_key']
        
        for field in sensitive_fields:
            if field in masked_data:
                masked_data[field] = '***MASKED***'
                
        return masked_data

# Decorators for observability
def observe_operation(operation_name: str, provider: str = "unknown"):
    """Decorator for adding observability to storage operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Create span for distributed tracing
            span = None
            if OTEL_AVAILABLE:
                tracer = trace.get_tracer(__name__)
                with tracer.start_as_current_span(f"{provider}.{operation_name}") as span:
                    span.set_attribute("operation", operation_name)
                    span.set_attribute("provider", provider)
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            duration = time.time() - start_time
            
            # Update metrics
            if OTEL_AVAILABLE:
                storage_requests.add(1, {"operation": operation_name, "provider": provider})
                request_duration.record(duration, {"operation": operation_name, "provider": provider})
            
            if PROMETHEUS_AVAILABLE:
                STORAGE_REQUESTS_TOTAL.labels(provider=provider, operation=operation_name).inc()
                REQUEST_DURATION.labels(provider=provider, operation=operation_name).observe(duration)
            
            # Structured logging
            logger.info(
                f"Storage operation completed",
                extra={
                    'extra_fields': {
                        'operation': operation_name,
                        'provider': provider,
                        'duration_seconds': duration,
                        'status': 'success'
                    }
                }
            )
            
            return result
        return wrapper
    return decorator

# Health check and monitoring
class HealthChecker:
    """Comprehensive health checking for all storage components"""
    
    @staticmethod
    def check_aws_s3() -> Dict[str, Any]:
        """Check AWS S3 health status"""
        try:
            # Implementation would use boto3 to check S3 connectivity
            return {"status": "healthy", "service": "aws_s3"}
        except Exception as e:
            return {"status": "unhealthy", "service": "aws_s3", "error": str(e)}
    
    @staticmethod
    def check_azure_storage() -> Dict[str, Any]:
        """Check Azure Storage health status"""
        try:
            # Implementation would use azure-storage-blob to check connectivity
            return {"status": "healthy", "service": "azure_storage"}
        except Exception as e:
            return {"status": "unhealthy", "service": "azure_storage", "error": str(e)}
    
    @staticmethod
    def check_redis() -> Dict[str, Any]:
        """Check Redis health status"""
        try:
            # Implementation would use redis-py to check connectivity
            return {"status": "healthy", "service": "redis"}
        except Exception as e:
            return {"status": "unhealthy", "service": "redis", "error": str(e)}
    
    @staticmethod
    def check_gcp_storage() -> Dict[str, Any]:
        """Check GCP Storage health status"""
        try:
            # Implementation would use google-cloud-storage to check connectivity
            return {"status": "healthy", "service": "gcp_storage"}
        except Exception as e:
            return {"status": "unhealthy", "service": "gcp_storage", "error": str(e)}

def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for all storage components
    
    Returns:
        Dict containing health status of all components
    """
    health_status = {
        "version": __version__,
        "overall_status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }
    
    # Check all components
    components = {
        "aws_s3": HealthChecker.check_aws_s3,
        "azure_storage": HealthChecker.check_azure_storage,
        "gcp_storage": HealthChecker.check_gcp_storage,
        "redis": HealthChecker.check_redis,
    }
    
    unhealthy_components = []
    
    for name, check_func in components.items():
        try:
            component_status = check_func()
            health_status["components"][name] = component_status
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                status_value = 1 if component_status["status"] == "healthy" else 0
                STORAGE_HEALTH.labels(provider=name).set(status_value)
            
            if component_status["status"] != "healthy":
                unhealthy_components.append(name)
                
        except Exception as e:
            health_status["components"][name] = {
                "status": "error", 
                "error": str(e)
            }
            unhealthy_components.append(name)
    
    # Update overall status
    if unhealthy_components:
        health_status["overall_status"] = "degraded"
    if all(comp["status"] in ["unhealthy", "error"] for comp in health_status["components"].values()):
        health_status["overall_status"] = "unhealthy"
    
    # Log health check result
    logger.info(
        "Health check completed",
        extra={
            'extra_fields': {
                'overall_status': health_status["overall_status"],
                'unhealthy_components': unhealthy_components,
                'total_components': len(components)
            }
        }
    )
    
    return health_status

# Failover and redundancy management
class FailoverManager:
    """Manages failover between different storage providers and cache layers"""
    
    @staticmethod
    def get_cloud_failover_chain() -> List[str]:
        """Get the failover chain for cloud providers"""
        # Configurable failover order - can be set via environment variables
        return os.getenv('STORAGE_FAILOVER_CHAIN', 'aws_s3,azure_storage,gcp_storage').split(',')
    
    @staticmethod
    def get_cache_failover_chain() -> List[str]:
        """Get the failover chain for cache layers"""
        return ['redis', 'memory']

# Initialize package components
def initialize_monitoring(enable_prometheus: bool = True, prometheus_port: int = 8000):
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

# Package imports and exports
try:
    from . import cloud
except ImportError:
    pass

try:
    from . import cache
except ImportError:
    pass

try:
    from . import backup
except ImportError:
    pass

__all__ = [
    "cloud", 
    "cache", 
    "backup",
    "health_check",
    "initialize_monitoring",
    "observe_operation",
    "SecurityManager",
    "FailoverManager"
]

__version__ = "2.0.0"

# Auto-initialize monitoring if enabled
if os.getenv('STORAGE_ENABLE_MONITORING', 'true').lower() == 'true':
    initialize_monitoring()

# Perform initial health check
logger.info(
    f"Storage package v{__version__} initialized",
    extra={
        'extra_fields': {
            'otel_available': OTEL_AVAILABLE,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'monitoring_enabled': os.getenv('STORAGE_ENABLE_MONITORING', 'true')
        }
    }
)