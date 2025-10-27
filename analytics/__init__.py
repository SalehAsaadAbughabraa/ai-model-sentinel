"""
🎯 Analytics Module Package
📦 Contains data processing and business intelligence components
👨‍💻 Author: Saleh Abughabraa
📧 Contact: saleh87alally@gmail.com
🚀 Version: 2.0.0

DESCRIPTION:
    This package handles all analytics and data processing for AI Model Sentinel,
    including big data integration, visualization, and machine learning.

MODULES:
    - bigdata: Integration with Snowflake, BigQuery, and other big data platforms
    - visualization: Data visualization, reporting, and dashboard components
    - ml: Machine learning models, anomaly detection, and AI algorithms

OPTIONAL DEPENDENCIES:
    - bigdata: Requires snowflake-connector-python, google-cloud-bigquery
    - visualization: Requires plotly, dash, matplotlib
    - ml: Requires scikit-learn, tensorflow, xgboost

USAGE:
    >>> from analytics import bigdata, visualization, ml
    >>> # Or import specific components as needed

EXAMPLES:
    # Lazy loading example
    from analytics.bigdata import SnowflakeConnector
    from analytics.ml import AnomalyDetector
    
    # On-demand initialization
    connector = SnowflakeConnector()
    detector = AnomalyDetector()

NOTES:
    - Uses lazy imports to improve performance and memory usage
    - Optional modules can be disabled if dependencies are missing
    - Version is managed centrally through package metadata
"""

import importlib
import warnings
from typing import Dict, Any, Optional

try:
    from ._version import __version__
except ImportError:
    try:
        from importlib.metadata import version
        __version__ = version("ai-model-sentinel-analytics")
    except ImportError:
        __version__ = "2.0.0"

__author__ = "Saleh Abughabraa"
__all__ = [
    "bigdata", 
    "visualization", 
    "ml",
    "__version__", 
    "__author__",
    "get_module_info",
    "check_dependencies",
    "initialize_module"
]

# Module configuration with dependencies and optional flags
_MODULE_CONFIG = {
    "bigdata": {
        "module": ".bigdata",
        "dependencies": ["snowflake.connector", "google.cloud.bigquery"],
        "optional": False,
        "description": "Big Data integration (Snowflake, BigQuery)",
        "critical_components": ["SnowflakeConnector", "BigQueryClient"]
    },
    "visualization": {
        "module": ".visualization", 
        "dependencies": ["plotly", "dash", "matplotlib"],
        "optional": True,
        "description": "Data visualization and reporting",
        "critical_components": ["DashboardBuilder", "ReportGenerator"]
    },
    "ml": {
        "module": ".ml",
        "dependencies": ["sklearn", "tensorflow", "xgboost"],
        "optional": True,
        "description": "Machine learning and anomaly detection",
        "critical_components": ["AnomalyDetector", "MLPipeline"]
    }
}

# Cache for loaded modules and initialization status
_import_cache = {}
_module_initialized = {}
_optional_modules_disabled = set()

def __getattr__(name: str) -> Any:
    """
    Lazy import implementation for analytics modules.
    
    Args:
        name (str): Name of the module or component to import
        
    Returns:
        Any: The requested module, class, or function
        
    Raises:
        AttributeError: If the requested name is not found
        ImportError: If required dependencies are missing for non-optional modules
    """
    # Check if it's a main module
    if name in _MODULE_CONFIG:
        return _load_module(name)
    
    # Check if it's a component from a specific module
    for module_name, config in _MODULE_CONFIG.items():
        if name in config.get("critical_components", []):
            parent_module = _load_module(module_name)
            try:
                return getattr(parent_module, name)
            except AttributeError as e:
                raise AttributeError(
                    f"Component '{name}' not found in module '{module_name}'. "
                    f"Available components: {config.get('critical_components', [])}"
                ) from e
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list:
    """
    Return list of available attributes for tab completion.
    
    Returns:
        list: All available modules and their critical components
    """
    all_attributes = list(__all__)
    for config in _MODULE_CONFIG.values():
        all_attributes.extend(config.get("critical_components", []))
    return sorted(set(all_attributes))

def _load_module(module_name: str) -> Any:
    """
    Load a module with dependency checking and optional module handling.
    
    Args:
        module_name (str): Name of the module to load
        
    Returns:
        Any: The loaded module
        
    Raises:
        ImportError: If required dependencies are missing for non-optional modules
    """
    if module_name in _import_cache:
        return _import_cache[module_name]
    
    config = _MODULE_CONFIG[module_name]
    
    # Check dependencies
    missing_deps = _check_module_dependencies(module_name)
    
    if missing_deps and not config["optional"]:
        raise ImportError(
            f"Required dependencies for '{module_name}' are missing: {missing_deps}. "
            f"Install with: pip install {' '.join(missing_deps)}"
        )
    elif missing_deps and config["optional"]:
        if module_name not in _optional_modules_disabled:
            warnings.warn(
                f"Optional module '{module_name}' is disabled due to missing dependencies: {missing_deps}. "
                f"Install with: pip install {' '.join(missing_deps)} to enable this module.",
                UserWarning,
                stacklevel=2
            )
            _optional_modules_disabled.add(module_name)
        
        # Create a dummy module for optional disabled modules
        dummy_module = type('DisabledModule', (), {
            '__doc__': f"Module '{module_name}' is disabled due to missing dependencies: {missing_deps}",
            '__missing_deps__': missing_deps,
            '__warn_on_use__': lambda: warnings.warn(
                f"Using disabled module '{module_name}'. Functionality will be limited.",
                UserWarning,
                stacklevel=2
            )
        })
        _import_cache[module_name] = dummy_module
        return dummy_module
    
    try:
        module = importlib.import_module(config["module"], package=__name__)
        _import_cache[module_name] = module
        return module
    except ImportError as e:
        if config["optional"]:
            warnings.warn(
                f"Failed to import optional module '{module_name}': {e}",
                UserWarning,
                stacklevel=2
            )
            # Return dummy module for optional modules
            dummy_module = type('FailedModule', (), {
                '__doc__': f"Module '{module_name}' failed to import: {e}",
                '__import_error__': e
            })
            _import_cache[module_name] = dummy_module
            return dummy_module
        else:
            raise ImportError(
                f"Failed to import required module '{module_name}': {e}"
            ) from e

def _check_module_dependencies(module_name: str) -> list:
    """
    Check if all dependencies for a module are available.
    
    Args:
        module_name (str): Name of the module to check
        
    Returns:
        list: List of missing dependencies
    """
    config = _MODULE_CONFIG[module_name]
    missing_deps = []
    
    for dep in config["dependencies"]:
        try:
            importlib.import_module(dep)
        except ImportError:
            missing_deps.append(dep)
    
    return missing_deps

def check_dependencies(module_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Check dependencies for all modules or a specific module.
    
    Args:
        module_name (str, optional): Specific module to check. If None, checks all.
        
    Returns:
        dict: Dependency status for each module
    """
    if module_name:
        if module_name not in _MODULE_CONFIG:
            raise ValueError(f"Unknown module: {module_name}")
        
        missing_deps = _check_module_dependencies(module_name)
        return {
            module_name: {
                "missing_dependencies": missing_deps,
                "all_satisfied": len(missing_deps) == 0,
                "optional": _MODULE_CONFIG[module_name]["optional"]
            }
        }
    
    result = {}
    for name in _MODULE_CONFIG:
        missing_deps = _check_module_dependencies(name)
        result[name] = {
            "missing_dependencies": missing_deps,
            "all_satisfied": len(missing_deps) == 0,
            "optional": _MODULE_CONFIG[name]["optional"]
        }
    
    return result

def initialize_module(module_name: str, init_params: Dict[str, Any] = None) -> bool:
    """
    Initialize a module with specific parameters on-demand.
    
    Args:
        module_name (str): Module to initialize
        init_params (dict): Initialization parameters
        
    Returns:
        bool: True if initialization was successful
    """
    if module_name not in _MODULE_CONFIG:
        raise ValueError(f"Unknown module: {module_name}")
    
    if init_params is None:
        init_params = {}
    
    module = _load_module(module_name)
    
    # Check if module has initialization function
    if hasattr(module, 'initialize'):
        try:
            result = module.initialize(**init_params)
            _module_initialized[module_name] = True
            return result if isinstance(result, bool) else True
        except Exception as e:
            warnings.warn(
                f"Failed to initialize module '{module_name}': {e}",
                UserWarning,
                stacklevel=2
            )
            return False
    
    _module_initialized[module_name] = True
    return True

def get_module_info() -> Dict[str, Any]:
    """
    Get information about all available modules.
    
    Returns:
        dict: Module information including status and dependencies
    """
    info = {}
    dependencies_status = check_dependencies()
    
    for name, config in _MODULE_CONFIG.items():
        info[name] = {
            "description": config["description"],
            "optional": config["optional"],
            "dependencies": config["dependencies"],
            "loaded": name in _import_cache,
            "initialized": _module_initialized.get(name, False),
            "disabled": name in _optional_modules_disabled,
            "dependencies_status": dependencies_status[name]
        }
    
    return info

# Perform initial dependency check
def _initial_dependency_check():
    """Perform initial dependency check and warn about missing optional deps."""
    deps_status = check_dependencies()
    
    for module_name, status in deps_status.items():
        if status["missing_dependencies"] and _MODULE_CONFIG[module_name]["optional"]:
            warnings.warn(
                f"Optional module '{module_name}' has missing dependencies: "
                f"{status['missing_dependencies']}. Some features may be unavailable.",
                UserWarning,
                stacklevel=1
            )

_initial_dependency_check()