# integration/__init__.py
"""
Integration Package - CI/CD and Cloud Integration System
Provides seamless integration with CI/CD pipelines and cloud platforms
"""

try:
    from .ci_cd_integration import CICDIntegration, CICDConfig, CICDPlatform, ScanTrigger, CICDContext
except ImportError as e:
    print(f"⚠️ CI/CD integration not available: {e}")

# Optional imports with error handling
try:
    from .cloud_integration import CloudIntegration
except ImportError:
    # Cloud integration is optional
    pass

__all__ = [
    'CICDIntegration',
    'CICDConfig', 
    'CICDPlatform',
    'ScanTrigger',
    'CICDContext'
]