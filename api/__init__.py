"""
🎯 API Module Package
📦 Contains REST API and web interface components
👨‍💻 Author: Saleh Abughabraa
📧 Contact: saleh87alally@gmail.com
🚀 Version: 2.0.0

DESCRIPTION:
    This package handles all API and web interface functionality for AI Model Sentinel,
    providing robust, secure, and scalable REST API solutions.

MODULES:
    - v1, v2: API version endpoints with full versioning support
    - middleware: Security, logging, metrics, and CORS middleware
    - docs: OpenAPI/Swagger documentation with examples
    - security: Advanced authentication and authorization
    - monitoring: Performance metrics and observability

FEATURES:
    - Lazy loading for improved startup performance
    - Advanced security with OAuth2 + JWT and automatic token refresh
    - Comprehensive rate limiting and protection against attacks
    - OpenAPI documentation with request/response examples
    - Prometheus metrics and OpenTelemetry tracing
    - Input validation with Pydantic models
    - Cloud-native deployment support

USAGE:
    >>> from api import v1, middleware, security
    >>> from api.monitoring import setup_metrics, setup_tracing

EXAMPLES:
    # Initialize API with monitoring
    from api import initialize_api
    from api.security import setup_authentication
    
    app = initialize_api()
    setup_authentication(app)
    setup_metrics(app)

NOTES:
    - All modules support lazy loading to optimize memory usage
    - Comprehensive security measures implemented by default
    - Designed for cloud-native environments and high availability
"""

import importlib
import warnings
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version
        __version__ = version("ai-model-sentinel-api")
    except ImportError:
        __version__ = "2.0.0"

__author__ = "Saleh Abughabraa"
__all__ = [
    "v1", "v2", "middleware", "docs", "security", "monitoring",
    "__version__", "__author__",
    "initialize_api", "get_api_info", "setup_rate_limiting",
    "validate_request", "create_api_response"
]

# Module configuration with performance and security settings
_MODULE_CONFIG = {
    "v1": {
        "module": ".v1",
        "dependencies": ["fastapi", "pydantic"],
        "lazy_load": True,
        "description": "API Version 1 endpoints",
        "rate_limit": "100/minute",
        "critical_components": ["router", "endpoints"]
    },
    "v2": {
        "module": ".v2", 
        "dependencies": ["fastapi", "pydantic"],
        "lazy_load": True,
        "description": "API Version 2 endpoints (latest)",
        "rate_limit": "200/minute",
        "critical_components": ["router", "endpoints"]
    },
    "middleware": {
        "module": ".middleware",
        "dependencies": ["fastapi", "starlette"],
        "lazy_load": False,
        "description": "Security, logging, and CORS middleware",
        "critical_components": ["SecurityMiddleware", "LoggingMiddleware", "CORSManager"]
    },
    "docs": {
        "module": ".docs",
        "dependencies": ["fastapi", "swagger-ui"],
        "lazy_load": True,
        "description": "OpenAPI/Swagger documentation",
        "critical_components": ["setup_swagger", "generate_openapi_spec"]
    },
    "security": {
        "module": ".security",
        "dependencies": ["authlib", "python-jose", "passlib"],
        "lazy_load": False,
        "description": "Authentication and authorization",
        "critical_components": ["OAuth2Manager", "JWTManager", "PermissionValidator"]
    },
    "monitoring": {
        "module": ".monitoring",
        "dependencies": ["prometheus_client", "opentelemetry"],
        "lazy_load": True,
        "description": "Metrics, tracing, and observability",
        "critical_components": ["setup_metrics", "setup_tracing", "MetricsCollector"]
    }
}

# Cache for loaded modules and performance tracking
_import_cache = {}
_performance_metrics = {
    "module_load_times": {},
    "request_counts": {},
    "error_counts": {}
}

class APIPerformanceTracker:
    """Track API performance metrics and load times."""
    
    @staticmethod
    def track_module_load(module_name: str, load_time: float):
        _performance_metrics["module_load_times"][module_name] = load_time
    
    @staticmethod
    def track_request(endpoint: str, status_code: int):
        if endpoint not in _performance_metrics["request_counts"]:
            _performance_metrics["request_counts"][endpoint] = 0
        _performance_metrics["request_counts"][endpoint] += 1
        
        if status_code >= 400:
            if endpoint not in _performance_metrics["error_counts"]:
                _performance_metrics["error_counts"][endpoint] = 0
            _performance_metrics["error_counts"][endpoint] += 1

def __getattr__(name: str) -> Any:
    """
    Lazy import implementation for API modules with performance tracking.
    
    Args:
        name (str): Name of the module or component to import
        
    Returns:
        Any: The requested module, class, or function
    """
    import time
    
    if name in _MODULE_CONFIG:
        start_time = time.time()
        module = _load_module(name)
        load_time = time.time() - start_time
        APIPerformanceTracker.track_module_load(name, load_time)
        return module
    
    # Check for components in modules
    for module_name, config in _MODULE_CONFIG.items():
        if name in config.get("critical_components", []):
            parent_module = _load_module(module_name)
            return getattr(parent_module, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> List[str]:
    """Return list of available attributes for tab completion."""
    all_attributes = list(__all__)
    for config in _MODULE_CONFIG.values():
        all_attributes.extend(config.get("critical_components", []))
    return sorted(set(all_attributes))

def _load_module(module_name: str) -> Any:
    """Load module with dependency checking and error handling."""
    if module_name in _import_cache:
        return _import_cache[module_name]
    
    config = _MODULE_CONFIG[module_name]
    
    # Check dependencies
    missing_deps = _check_dependencies(config["dependencies"])
    if missing_deps:
        raise ImportError(
            f"Missing dependencies for {module_name}: {missing_deps}. "
            f"Install with: pip install {' '.join(missing_deps)}"
        )
    
    try:
        module = importlib.import_module(config["module"], package=__name__)
        _import_cache[module_name] = module
        return module
    except ImportError as e:
        raise ImportError(f"Failed to import module {module_name}: {e}") from e

def _check_dependencies(dependencies: List[str]) -> List[str]:
    """Check if all dependencies are available."""
    missing = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing.append(dep)
    return missing

# Core API Management Functions
def initialize_api(
    version: str = "v2",
    enable_docs: bool = True,
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    rate_limiting: bool = True
) -> Any:
    """
    Initialize the API application with configured settings.
    
    Args:
        version: API version to use (v1, v2)
        enable_docs: Enable OpenAPI documentation
        enable_metrics: Enable Prometheus metrics
        enable_tracing: Enable OpenTelemetry tracing
        rate_limiting: Enable rate limiting
        
    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI
        from .middleware import setup_middleware
        from .security import setup_authentication
        
        # Create FastAPI application with lifespan support
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            if enable_metrics:
                monitoring = _load_module("monitoring")
                monitoring.setup_metrics(app)
            
            if enable_tracing:
                monitoring = _load_module("monitoring")
                monitoring.setup_tracing(app)
            
            yield
            
            # Shutdown
            # Cleanup connections and resources
        
        app = FastAPI(
            title="AI Model Sentinel API",
            version=__version__,
            description="Secure and scalable API for AI Model Sentinel",
            lifespan=lifespan
        )
        
        # Setup middleware and security
        setup_middleware(app)
        setup_authentication(app)
        
        # Include versioned routers
        api_version = _load_module(version)
        app.include_router(api_version.router, prefix=f"/api/{version}")
        
        # Setup documentation
        if enable_docs:
            docs_module = _load_module("docs")
            docs_module.setup_swagger(app)
        
        # Setup rate limiting
        if rate_limiting:
            from .middleware import RateLimiter
            RateLimiter.setup(app)
        
        return app
        
    except ImportError as e:
        raise RuntimeError(f"Failed to initialize API: {e}") from e

def setup_rate_limiting(
    app: Any,
    default_limit: str = "100/minute",
    strategy: str = "fixed-window",
    storage_url: Optional[str] = None
) -> None:
    """
    Configure rate limiting for the API.
    
    Args:
        app: FastAPI application instance
        default_limit: Default rate limit string
        strategy: Rate limiting strategy (fixed-window, moving-window, token-bucket)
        storage_url: Storage URL for distributed rate limiting
    """
    try:
        from .middleware import RateLimiter
        RateLimiter.configure(
            app=app,
            default_limit=default_limit,
            strategy=strategy,
            storage_url=storage_url
        )
    except ImportError as e:
        warnings.warn(f"Rate limiting not available: {e}", UserWarning)

def validate_request(data: Any, validator: Any) -> Dict[str, Any]:
    """
    Validate request data using Pydantic models.
    
    Args:
        data: Request data to validate
        validator: Pydantic model class for validation
        
    Returns:
        Dict with validation results
    """
    try:
        validated_data = validator(**data)
        return {
            "valid": True,
            "data": validated_data.dict(),
            "errors": []
        }
    except Exception as e:
        return {
            "valid": False,
            "data": {},
            "errors": [str(e)]
        }

def create_api_response(
    success: bool,
    data: Any = None,
    message: str = "",
    status_code: int = 200,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create standardized API response.
    
    Args:
        success: Whether the request was successful
        data: Response data
        message: Response message
        status_code: HTTP status code
        metadata: Additional metadata
        
    Returns:
        Standardized response dictionary
    """
    response = {
        "success": success,
        "message": message,
        "data": data or {},
        "timestamp": importlib.import_module("datetime").datetime.utcnow().isoformat(),
        "version": __version__
    }
    
    if metadata:
        response["metadata"] = metadata
    
    # Track response for monitoring
    APIPerformanceTracker.track_request("create_api_response", status_code)
    
    return response

def get_api_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the API setup.
    
    Returns:
        API information dictionary
    """
    dependencies_status = {}
    for module_name, config in _MODULE_CONFIG.items():
        missing_deps = _check_dependencies(config["dependencies"])
        dependencies_status[module_name] = {
            "missing_dependencies": missing_deps,
            "all_satisfied": len(missing_deps) == 0,
            "lazy_loaded": config["lazy_load"],
            "loaded": module_name in _import_cache
        }
    
    return {
        "version": __version__,
        "author": __author__,
        "available_versions": ["v1", "v2"],
        "performance_metrics": _performance_metrics,
        "dependencies_status": dependencies_status,
        "modules_loaded": list(_import_cache.keys())
    }

# Security configuration
class SecurityConfig:
    """Security configuration for API endpoints."""
    
    OAUTH2_SCHEME = "oauth2"
    JWT_ALGORITHM = "RS256"
    TOKEN_EXPIRY = 3600  # 1 hour
    REFRESH_TOKEN_EXPIRY = 86400  # 24 hours
    
    RATE_LIMITS = {
        "auth": "10/minute",
        "api": "100/minute",
        "admin": "1000/minute"
    }
    
    CORS_ORIGINS = [
        "http://localhost:3000",
        "https://yourdomain.com"
    ]

# Initialize critical modules on import
try:
    _load_module("security")
    _load_module("middleware")
except ImportError as e:
    warnings.warn(f"Failed to load critical modules: {e}", UserWarning)