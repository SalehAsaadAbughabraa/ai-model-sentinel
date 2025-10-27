"""
🎯 Security Module Package
📦 Contains all security and compliance components for AI Model Sentinel
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Description:
   This package provides comprehensive security management including:
   - Encryption and cryptographic utilities
   - Authentication and authorization systems
   - Compliance and audit management
   - Security monitoring and incident response
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Import all security components
from .encryption import EncryptionManager, encryption_manager, initialize_encryption
from .authentication import AuthenticationManager, auth_manager, initialize_authentication
from .compliance import ComplianceManager, compliance_manager, initialize_compliance

# Configure logging
logger = logging.getLogger(__name__)

class SecurityComponent(Enum):
    """Enum for security components"""
    ENCRYPTION = "encryption"
    AUTHENTICATION = "authentication" 
    COMPLIANCE = "compliance"

class SecuritySystemStatus(Enum):
    """Status of security system components"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    INITIALIZING = "initializing"

@dataclass
class ComponentHealth:
    """Health status of a security component"""
    component: SecurityComponent
    status: SecuritySystemStatus
    message: str
    response_time: float
    last_checked: str

@dataclass
class SecuritySystemHealth:
    """Overall health status of security system"""
    overall_status: SecuritySystemStatus
    components: Dict[SecurityComponent, ComponentHealth]
    timestamp: str

class SecurityManager:
    """
    Central security management class that orchestrates all security components
    and provides unified interfaces for the entire security system.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecurityManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.encryption_manager: Optional[EncryptionManager] = None
            self.auth_manager: Optional[AuthenticationManager] = None
            self.compliance_manager: Optional[ComplianceManager] = None
            self.tenant_context: Dict[str, Any] = {}
            self._health_status: SecuritySystemStatus = SecuritySystemStatus.UNHEALTHY
            self._component_health: Dict[SecurityComponent, ComponentHealth] = {}
            self._initialized = True
            logger.info("SecurityManager initialized")
    
    async def initialize_security_system(self, 
                                      config: Dict[str, Any],
                                      tenant_id: Optional[str] = None) -> bool:
        """
        Initialize the entire security system with proper error handling and health checks
        
        Args:
            config: Configuration for all security components
            tenant_id: Optional tenant ID for multi-tenant context
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting security system initialization...")
            
            # Set tenant context if provided
            if tenant_id:
                self.tenant_context["tenant_id"] = tenant_id
            
            # Initialize components with proper error handling
            initialization_tasks = []
            
            # Encryption Manager Initialization
            if config.get("encryption", {}).get("enabled", True):
                encryption_task = self._initialize_component(
                    SecurityComponent.ENCRYPTION,
                    initialize_encryption,
                    config.get("encryption", {})
                )
                initialization_tasks.append(encryption_task)
            
            # Authentication Manager Initialization  
            if config.get("authentication", {}).get("enabled", True):
                auth_task = self._initialize_component(
                    SecurityComponent.AUTHENTICATION,
                    initialize_authentication,
                    config.get("authentication", {})
                )
                initialization_tasks.append(auth_task)
            
            # Compliance Manager Initialization
            if config.get("compliance", {}).get("enabled", True):
                compliance_task = self._initialize_component(
                    SecurityComponent.COMPLIANCE,
                    initialize_compliance,
                    config.get("compliance", {})
                )
                initialization_tasks.append(compliance_task)
            
            # Execute all initializations concurrently
            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)
            
            # Check results and update health status
            successful_initializations = 0
            for i, result in enumerate(results):
                component = list(SecurityComponent)[i]
                if isinstance(result, Exception):
                    logger.error(f"Failed to initialize {component.value}: {result}")
                    self._update_component_health(
                        component, 
                        SecuritySystemStatus.UNHEALTHY,
                        f"Initialization failed: {str(result)}"
                    )
                else:
                    successful_initializations += 1
                    self._update_component_health(
                        component,
                        SecuritySystemStatus.HEALTHY,
                        "Component initialized successfully"
                    )
            
            # Determine overall system health
            total_components = len(initialization_tasks)
            if successful_initializations == total_components:
                self._health_status = SecuritySystemStatus.HEALTHY
            elif successful_initializations > 0:
                self._health_status = SecuritySystemStatus.DEGRADED
            else:
                self._health_status = SecuritySystemStatus.UNHEALTHY
            
            logger.info(f"Security system initialization completed. Status: {self._health_status.value}")
            return self._health_status != SecuritySystemStatus.UNHEALTHY
            
        except Exception as e:
            logger.error(f"Security system initialization failed: {e}")
            self._health_status = SecuritySystemStatus.UNHEALTHY
            return False
    
    async def _initialize_component(self, 
                                  component: SecurityComponent,
                                  init_function,
                                  config: Dict[str, Any]) -> Any:
        """
        Initialize a security component with proper async handling and monitoring
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Initializing {component.value}...")
            result = await init_function(config)
            
            # Set component instance
            if component == SecurityComponent.ENCRYPTION:
                self.encryption_manager = result
            elif component == SecurityComponent.AUTHENTICATION:
                self.auth_manager = result
            elif component == SecurityComponent.COMPLIANCE:
                self.compliance_manager = result
            
            response_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"{component.value} initialized in {response_time:.2f}s")
            
            return result
            
        except Exception as e:
            response_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"{component.value} initialization failed after {response_time:.2f}s: {e}")
            raise
    
    def _update_component_health(self, 
                               component: SecurityComponent,
                               status: SecuritySystemStatus,
                               message: str):
        """Update health status of a component"""
        self._component_health[component] = ComponentHealth(
            component=component,
            status=status,
            message=message,
            response_time=0.0,  # Will be updated during health checks
            last_checked=asyncio.get_event_loop().time()
        )
    
    async def health_check(self) -> SecuritySystemHealth:
        """
        Comprehensive health check for all security components
        
        Returns:
            SecuritySystemHealth: Detailed health status of all components
        """
        health_checks = []
        
        # Check encryption component
        if self.encryption_manager and hasattr(self.encryption_manager, 'health_check'):
            encryption_health = await self.encryption_manager.health_check()
            health_checks.append((SecurityComponent.ENCRYPTION, encryption_health))
        
        # Check authentication component
        if self.auth_manager and hasattr(self.auth_manager, 'health_check'):
            auth_health = await self.auth_manager.health_check()
            health_checks.append((SecurityComponent.AUTHENTICATION, auth_health))
        
        # Check compliance component
        if self.compliance_manager and hasattr(self.compliance_manager, 'health_check'):
            compliance_health = await self.compliance_manager.health_check()
            health_checks.append((SecurityComponent.COMPLIANCE, compliance_health))
        
        # Determine overall health status
        component_statuses = [health[1].get('status', 'unknown') for health in health_checks]
        
        if all(status == 'healthy' for status in component_statuses):
            overall_status = SecuritySystemStatus.HEALTHY
        elif any(status == 'unhealthy' for status in component_statuses):
            overall_status = SecuritySystemStatus.UNHEALTHY
        else:
            overall_status = SecuritySystemStatus.DEGRADED
        
        # Build component health details
        component_details = {}
        for component, health_data in health_checks:
            component_details[component] = ComponentHealth(
                component=component,
                status=SecuritySystemStatus(health_data.get('status', 'unhealthy')),
                message=health_data.get('message', 'No health information'),
                response_time=health_data.get('response_time', 0.0),
                last_checked=health_data.get('timestamp', '')
            )
        
        return SecuritySystemHealth(
            overall_status=overall_status,
            components=component_details,
            timestamp=str(asyncio.get_event_loop().time())
        )
    
    def set_tenant_context(self, tenant_id: str, user_id: Optional[str] = None):
        """
        Set tenant context for multi-tenant environment
        
        Args:
            tenant_id: Tenant identifier
            user_id: Optional user identifier
        """
        self.tenant_context = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # Propagate tenant context to all components
        if self.encryption_manager and hasattr(self.encryption_manager, 'set_tenant_context'):
            self.encryption_manager.set_tenant_context(tenant_id)
        
        if self.auth_manager and hasattr(self.auth_manager, 'set_tenant_context'):
            self.auth_manager.set_tenant_context(tenant_id, user_id)
        
        if self.compliance_manager and hasattr(self.compliance_manager, 'set_tenant_context'):
            self.compliance_manager.set_tenant_context(tenant_id, user_id)
    
    def get_tenant_context(self) -> Dict[str, Any]:
        """Get current tenant context"""
        return self.tenant_context.copy()
    
    async def encrypt_sensitive_data(self, data: str, context: Dict[str, Any] = None) -> str:
        """Encrypt sensitive data using encryption manager"""
        if not self.encryption_manager:
            raise RuntimeError("Encryption manager not initialized")
        
        tenant_context = context or self.tenant_context
        return await self.encryption_manager.encrypt(data, tenant_context)
    
    async def decrypt_sensitive_data(self, encrypted_data: str, context: Dict[str, Any] = None) -> str:
        """Decrypt sensitive data using encryption manager"""
        if not self.encryption_manager:
            raise RuntimeError("Encryption manager not initialized")
        
        tenant_context = context or self.tenant_context
        return await self.encryption_manager.decrypt(encrypted_data, tenant_context)
    
    async def authenticate_user(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate user using authentication manager"""
        if not self.auth_manager:
            raise RuntimeError("Authentication manager not initialized")
        
        return await self.auth_manager.authenticate(credentials, self.tenant_context)
    
    async def verify_permission(self, user_id: str, permission: str, resource: str) -> bool:
        """Verify user permission using authentication manager"""
        if not self.auth_manager:
            raise RuntimeError("Authentication manager not initialized")
        
        return await self.auth_manager.verify_permission(user_id, permission, resource, self.tenant_context)
    
    async def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event using compliance manager"""
        if not self.compliance_manager:
            raise RuntimeError("Compliance manager not initialized")
        
        event_data = {
            **details,
            "tenant_context": self.tenant_context,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        await self.compliance_manager.log_event(event_type, event_data)
    
    @property
    def is_healthy(self) -> bool:
        """Check if security system is healthy"""
        return self._health_status == SecuritySystemStatus.HEALTHY
    
    @property
    def status(self) -> SecuritySystemStatus:
        """Get current security system status"""
        return self._health_status

# Global security manager instance
security_manager = SecurityManager()

# Main initialization function
async def initialize_security_system(config: Dict[str, Any], 
                                   tenant_id: Optional[str] = None) -> bool:
    """
    Main function to initialize the entire security system
    
    Args:
        config: Security system configuration
        tenant_id: Optional tenant ID for multi-tenant setup
    
    Returns:
        bool: True if initialization successful
    """
    return await security_manager.initialize_security_system(config, tenant_id)

# Export all components and the security manager
__all__ = [
    # Core security manager
    "SecurityManager",
    "security_manager",
    "initialize_security_system",
    
    # Encryption
    "EncryptionManager",
    "encryption_manager", 
    "initialize_encryption",
    
    # Authentication
    "AuthenticationManager",
    "auth_manager",
    "initialize_authentication",
    
    # Compliance
    "ComplianceManager", 
    "compliance_manager",
    "initialize_compliance",
    
    # Types and enums
    "SecurityComponent",
    "SecuritySystemStatus", 
    "ComponentHealth",
    "SecuritySystemHealth"
]

__version__ = "2.0.0"