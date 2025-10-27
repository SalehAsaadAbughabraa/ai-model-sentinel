"""
🎯 Documentation Module Package
📦 Contains all system documentation and guides
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Description:
   This package contains comprehensive documentation:
   - API documentation and references
   - Deployment guides and tutorials
   - User guides and manuals
   - Developer guides and contribution docs
   - Automated documentation generation and publishing
"""

import json
import logging
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import webbrowser
from pathlib import Path

# Try to import documentation frameworks
try:
    import sphinx
    from sphinx.application import Sphinx
    SPHINX_AVAILABLE = True
except ImportError:
    SPHINX_AVAILABLE = False

try:
    import mkdocs
    MKDOCS_AVAILABLE = True
except ImportError:
    MKDOCS_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Prometheus imports for monitoring
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
            
        return json.dumps(log_entry)

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(StructuredFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Prometheus metrics for documentation monitoring
if PROMETHEUS_AVAILABLE:
    DOCS_BUILDS_TOTAL = Counter(
        'docs_builds_total', 
        'Total documentation builds', 
        ['builder', 'status']
    )
    DOCS_BUILD_DURATION = Histogram(
        'docs_build_duration_seconds',
        'Documentation build duration',
        ['builder']
    )
    DOCS_SERVE_REQUESTS = Counter(
        'docs_serve_requests_total',
        'Total documentation serve requests',
        ['host', 'port']
    )
    DOCS_DEPLOYMENTS_TOTAL = Counter(
        'docs_deployments_total',
        'Total documentation deployments',
        ['platform', 'status']
    )
    DOCS_HEALTH = Gauge(
        'docs_health_status',
        'Documentation system health status',
        ['component']
    )

# Documentation Builders Enum
class DocBuilder(Enum):
    SPHINX = "sphinx"
    MKDOCS = "mkdocs"
    BOTH = "both"

# Deployment Platforms Enum
class DeploymentPlatform(Enum):
    GITHUB_PAGES = "github_pages"
    AWS_S3 = "aws_s3"
    NETLIFY = "netlify"
    GITLAB_PAGES = "gitlab_pages"
    AZURE_STORAGE = "azure_storage"

# Documentation Result Class
@dataclass
class BuildResult:
    builder: DocBuilder
    success: bool
    duration: float
    output_path: str
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

# Documentation Manager
class DocumentationManager:
    """Main documentation management class with automated generation and publishing"""
    
    def __init__(self):
        self.docs_dir = Path("docs")
        self.build_dir = Path("docs_build")
        self.config_dir = Path("docs_config")
        
        # Create directories if they don't exist
        self.docs_dir.mkdir(exist_ok=True)
        self.build_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup documentation-specific logging"""
        self.logger = logging.getLogger(f"{__name__}.manager")
    
    def generate_sphinx_docs(self, source_dir: str = "docs/source", 
                           build_dir: str = "docs_build/sphinx") -> BuildResult:
        """Generate documentation using Sphinx"""
        start_time = time.time()
        
        try:
            source_path = Path(source_dir)
            build_path = Path(build_dir)
            
            # Create directories if they don't exist
            source_path.mkdir(parents=True, exist_ok=True)
            build_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize Sphinx if conf.py doesn't exist
            conf_file = source_path / "conf.py"
            if not conf_file.exists():
                self._init_sphinx_config(str(source_path))
            
            # Build Sphinx documentation
            cmd = [
                "sphinx-build",
                "-b", "html",
                str(source_path),
                str(build_path),
                "-v"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            duration = time.time() - start_time
            
            success = result.returncode == 0
            warnings = [line for line in result.stderr.split('\n') if 'WARNING' in line]
            
            build_result = BuildResult(
                builder=DocBuilder.SPHINX,
                success=success,
                duration=duration,
                output_path=str(build_path),
                error_message=result.stderr if not success else None,
                warnings=warnings
            )
            
            # Update metrics
            if PROMETHEUS_AVAILABLE:
                status = "success" if success else "failure"
                DOCS_BUILDS_TOTAL.labels(builder="sphinx", status=status).inc()
                DOCS_BUILD_DURATION.labels(builder="sphinx").observe(duration)
            
            self.logger.info(
                f"Sphinx documentation build {'succeeded' if success else 'failed'}",
                extra={
                    'extra_fields': {
                        'duration': duration,
                        'warnings_count': len(warnings),
                        'output_path': str(build_path)
                    }
                }
            )
            
            return build_result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Sphinx build failed: {e}")
            
            if PROMETHEUS_AVAILABLE:
                DOCS_BUILDS_TOTAL.labels(builder="sphinx", status="error").inc()
            
            return BuildResult(
                builder=DocBuilder.SPHINX,
                success=False,
                duration=duration,
                output_path="",
                error_message=str(e)
            )
    
    def generate_mkdocs_docs(self, config_file: str = "mkdocs.yml", 
                           build_dir: str = "docs_build/mkdocs") -> BuildResult:
        """Generate documentation using MkDocs"""
        start_time = time.time()
        
        try:
            build_path = Path(build_dir)
            build_path.mkdir(parents=True, exist_ok=True)
            
            # Create mkdocs.yml if it doesn't exist
            if not Path(config_file).exists():
                self._init_mkdocs_config(config_file)
            
            # Build MkDocs documentation
            cmd = ["mkdocs", "build", "--site-dir", str(build_path), "--verbose"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            duration = time.time() - start_time
            
            success = result.returncode == 0
            warnings = [line for line in result.stderr.split('\n') if 'WARNING' in line]
            
            build_result = BuildResult(
                builder=DocBuilder.MKDOCS,
                success=success,
                duration=duration,
                output_path=str(build_path),
                error_message=result.stderr if not success else None,
                warnings=warnings
            )
            
            # Update metrics
            if PROMETHEUS_AVAILABLE:
                status = "success" if success else "failure"
                DOCS_BUILDS_TOTAL.labels(builder="mkdocs", status=status).inc()
                DOCS_BUILD_DURATION.labels(builder="mkdocs").observe(duration)
            
            self.logger.info(
                f"MkDocs documentation build {'succeeded' if success else 'failed'}",
                extra={
                    'extra_fields': {
                        'duration': duration,
                        'warnings_count': len(warnings),
                        'output_path': str(build_path)
                    }
                }
            )
            
            return build_result
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"MkDocs build failed: {e}")
            
            if PROMETHEUS_AVAILABLE:
                DOCS_BUILDS_TOTAL.labels(builder="mkdocs", status="error").inc()
            
            return BuildResult(
                builder=DocBuilder.MKDOCS,
                success=False,
                duration=duration,
                output_path="",
                error_message=str(e)
            )
    
    def _init_sphinx_config(self, source_dir: str):
        """Initialize Sphinx configuration"""
        try:
            # Create basic Sphinx configuration
            conf_content = '''
project = 'System Documentation'
copyright = '2024, Saleh Abughabraa'
author = 'Saleh Abughabraa'
version = '2.0.0'
release = '2.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
'''
            conf_file = Path(source_dir) / "conf.py"
            conf_file.parent.mkdir(parents=True, exist_ok=True)
            conf_file.write_text(conf_content)
            
            # Create index.rst
            index_content = '''
Welcome to System Documentation
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api
   deployment
   user_guide
   developer_guide

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
'''
            index_file = Path(source_dir) / "index.rst"
            index_file.write_text(index_content)
            
            self.logger.info("Initialized Sphinx configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Sphinx config: {e}")
    
    def _init_mkdocs_config(self, config_file: str):
        """Initialize MkDocs configuration"""
        try:
            config = {
                'site_name': 'System Documentation',
                'site_description': 'Comprehensive system documentation',
                'site_author': 'Saleh Abughabraa',
                'repo_url': 'https://github.com/your-username/your-repo',
                'nav': [
                    {'Home': 'index.md'},
                    {'API': 'api.md'},
                    {'Deployment': 'deployment.md'},
                    {'User Guide': 'user_guide.md'},
                    {'Developer Guide': 'developer_guide.md'}
                ],
                'theme': 'readthedocs',
                'markdown_extensions': [
                    'toc',
                    'tables',
                    'fenced_code',
                    'codehilite'
                ]
            }
            
            if YAML_AVAILABLE:
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            else:
                # Fallback to basic YAML writing
                with open(config_file, 'w') as f:
                    f.write("site_name: System Documentation\\n")
                    f.write("site_description: Comprehensive system documentation\\n")
                    # ... more basic config
            
            self.logger.info("Initialized MkDocs configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MkDocs config: {e}")
    
    def serve_docs(self, builder: DocBuilder = DocBuilder.MKDOCS, 
                   host: str = "localhost", port: int = 8000) -> bool:
        """Serve documentation locally for preview"""
        try:
            if builder == DocBuilder.SPHINX:
                # For Sphinx, we need to build first then serve static files
                build_result = self.generate_sphinx_docs()
                if not build_result.success:
                    return False
                
                # Serve static files using Python HTTP server
                os.chdir(build_result.output_path)
                cmd = [sys.executable, "-m", "http.server", str(port), "--bind", host]
                
            else:  # MkDocs
                cmd = ["mkdocs", "serve", "--dev-addr", f"{host}:{port}"]
            
            self.logger.info(
                f"Serving {builder.value} documentation on http://{host}:{port}"
            )
            
            if PROMETHEUS_AVAILABLE:
                DOCS_SERVE_REQUESTS.labels(host=host, port=str(port)).inc()
            
            # Run the server
            subprocess.run(cmd, cwd=os.getcwd())
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to serve documentation: {e}")
            return False
    
    def deploy_docs(self, platform: DeploymentPlatform, 
                   build_dir: str = "docs_build/mkdocs") -> bool:
        """Deploy documentation to specified platform"""
        start_time = time.time()
        
        try:
            build_path = Path(build_dir)
            if not build_path.exists():
                self.logger.error(f"Build directory not found: {build_dir}")
                return False
            
            success = False
            deployment_cmd = []
            
            if platform == DeploymentPlatform.GITHUB_PAGES:
                deployment_cmd = ["mkdocs", "gh-deploy", "--force"]
                success = self._run_deployment_command(deployment_cmd)
                
            elif platform == DeploymentPlatform.AWS_S3:
                bucket_name = os.getenv('AWS_S3_BUCKET', 'my-docs-bucket')
                deployment_cmd = [
                    "aws", "s3", "sync", str(build_path), 
                    f"s3://{bucket_name}", "--delete"
                ]
                success = self._run_deployment_command(deployment_cmd)
                
            elif platform == DeploymentPlatform.NETLIFY:
                site_id = os.getenv('NETLIFY_SITE_ID')
                if site_id:
                    deployment_cmd = [
                        "netlify", "deploy", 
                        "--dir", str(build_path),
                        "--prod", 
                        "--site", site_id
                    ]
                    success = self._run_deployment_command(deployment_cmd)
            
            duration = time.time() - start_time
            
            # Update metrics
            if PROMETHEUS_AVAILABLE:
                status = "success" if success else "failure"
                DOCS_DEPLOYMENTS_TOTAL.labels(platform=platform.value, status=status).inc()
            
            self.logger.info(
                f"Deployment to {platform.value} {'succeeded' if success else 'failed'}",
                extra={
                    'extra_fields': {
                        'platform': platform.value,
                        'duration': duration,
                        'success': success
                    }
                }
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            
            if PROMETHEUS_AVAILABLE:
                DOCS_DEPLOYMENTS_TOTAL.labels(platform=platform.value, status="error").inc()
            
            return False
    
    def _run_deployment_command(self, cmd: List[str]) -> bool:
        """Run deployment command and return success status"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Deployment command failed: {e}")
            return False

# CI/CD Integration
class CICDIntegration:
    """CI/CD integration for automated documentation publishing"""
    
    @staticmethod
    def github_actions_workflow() -> Dict[str, Any]:
        """Generate GitHub Actions workflow for documentation"""
        return {
            "name": "Deploy Documentation",
            "on": {
                "push": {
                    "branches": ["main"],
                    "paths": ["docs/**", "mkdocs.yml", "sphinx/**"]
                }
            },
            "jobs": {
                "deploy": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"uses": "actions/checkout@v2"},
                        {"uses": "actions/setup-python@v2"},
                        {"run": "pip install mkdocs sphinx"},
                        {"run": "python -m documentation generate --builder mkdocs"},
                        {"run": "python -m documentation deploy --platform github_pages"}
                    ]
                }
            }
        }
    
    @staticmethod
    def gitlab_ci_config() -> Dict[str, Any]:
        """Generate GitLab CI configuration for documentation"""
        return {
            "pages": {
                "stage": "deploy",
                "script": [
                    "pip install mkdocs",
                    "python -m documentation generate --builder mkdocs",
                    "mv docs_build/mkdocs public"
                ],
                "artifacts": {"paths": ["public"]},
                "only": ["main"]
            }
        }

# Health Check and Monitoring
class DocumentationHealthChecker:
    """Health checking for documentation system"""
    
    @staticmethod
    def check_sphinx() -> Dict[str, Any]:
        """Check Sphinx health status"""
        try:
            import sphinx
            return {"status": "healthy", "component": "sphinx", "version": sphinx.__version__}
        except Exception as e:
            return {"status": "unhealthy", "component": "sphinx", "error": str(e)}
    
    @staticmethod
    def check_mkdocs() -> Dict[str, Any]:
        """Check MkDocs health status"""
        try:
            import mkdocs
            return {"status": "healthy", "component": "mkdocs", "version": mkdocs.__version__}
        except Exception as e:
            return {"status": "unhealthy", "component": "mkdocs", "error": str(e)}
    
    @staticmethod
    def check_build_directories() -> Dict[str, Any]:
        """Check build directories health"""
        try:
            directories = ["docs", "docs_build", "docs_config"]
            status = {}
            for dir_name in directories:
                path = Path(dir_name)
                status[dir_name] = {
                    "exists": path.exists(),
                    "writable": os.access(path, os.W_OK) if path.exists() else False
                }
            return {"status": "healthy", "component": "directories", "details": status}
        except Exception as e:
            return {"status": "unhealthy", "component": "directories", "error": str(e)}

def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for documentation system
    
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
        "sphinx": DocumentationHealthChecker.check_sphinx,
        "mkdocs": DocumentationHealthChecker.check_mkdocs,
        "directories": DocumentationHealthChecker.check_build_directories,
    }
    
    unhealthy_components = []
    
    for name, check_func in components.items():
        try:
            component_status = check_func()
            health_status["components"][name] = component_status
            
            # Update Prometheus metrics
            if PROMETHEUS_AVAILABLE:
                status_value = 1 if component_status["status"] == "healthy" else 0
                DOCS_HEALTH.labels(component=name).set(status_value)
            
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
        "Documentation health check completed",
        extra={
            'extra_fields': {
                'overall_status': health_status["overall_status"],
                'unhealthy_components': unhealthy_components,
                'total_components': len(components)
            }
        }
    )
    
    return health_status

# Main documentation functions
def generate_docs(builder: DocBuilder = DocBuilder.BOTH) -> Dict[DocBuilder, BuildResult]:
    """
    Generate documentation using specified builder
    
    Args:
        builder: Documentation builder to use
    
    Returns:
        Dictionary of build results
    """
    manager = DocumentationManager()
    results = {}
    
    if builder in [DocBuilder.SPHINX, DocBuilder.BOTH]:
        results[DocBuilder.SPHINX] = manager.generate_sphinx_docs()
    
    if builder in [DocBuilder.MKDOCS, DocBuilder.BOTH]:
        results[DocBuilder.MKDOCS] = manager.generate_mkdocs_docs()
    
    return results

def serve_docs(builder: DocBuilder = DocBuilder.MKDOCS, 
               host: str = "localhost", port: int = 8000,
               open_browser: bool = True) -> bool:
    """
    Serve documentation locally for preview
    
    Args:
        builder: Documentation builder to serve
        host: Host to serve on
        port: Port to serve on
        open_browser: Whether to open browser automatically
    
    Returns:
        Success status
    """
    manager = DocumentationManager()
    success = manager.serve_docs(builder, host, port)
    
    if success and open_browser:
        try:
            webbrowser.open(f"http://{host}:{port}")
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")
    
    return success

# Package imports
try:
    from . import api
except ImportError as e:
    logger.warning(f"Could not import api documentation: {e}")

try:
    from . import deployment
except ImportError as e:
    logger.warning(f"Could not import deployment documentation: {e}")

try:
    from . import user_guide
except ImportError as e:
    logger.warning(f"Could not import user_guide documentation: {e}")

try:
    from . import developer_guide
except ImportError as e:
    logger.warning(f"Could not import developer_guide documentation: {e}")

# Export public API
__all__ = [
    "api", 
    "deployment", 
    "user_guide", 
    "developer_guide",
    "DocBuilder",
    "DeploymentPlatform",
    "DocumentationManager",
    "CICDIntegration",
    "generate_docs",
    "serve_docs",
    "health_check"
]

__version__ = "2.0.0"

# Initialize monitoring if enabled
if os.getenv('DOCS_ENABLE_MONITORING', 'false').lower() == 'true' and PROMETHEUS_AVAILABLE:
    try:
        start_http_server(8000)
        logger.info("Documentation monitoring enabled on port 8000")
    except Exception as e:
        logger.warning(f"Could not start monitoring server: {e}")

logger.info(f"Documentation package v{__version__} initialized")