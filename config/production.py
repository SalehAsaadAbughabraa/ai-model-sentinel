"""
Production Configuration with Secure Secrets Management
Enhanced version with caching, key rotation, and comprehensive error handling
"""
import os
import base64
import logging
from functools import lru_cache
from typing import Optional, Union
from datetime import datetime, timedelta

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Secrets management clients
try:
    import hvac  # HashiCorp Vault client
except ImportError:
    hvac = None

try:
    import boto3  # AWS KMS
except ImportError:
    boto3 = None

# Custom exceptions
class SecretRetrievalError(Exception):
    """Custom exception for secret retrieval failures"""
    pass

class KeyRotationError(Exception):
    """Custom exception for key rotation failures"""
    pass

class SecureConfig:
    """
    Secure configuration management with Vault/KMS fallback
    
    Features:
    - Multi-layer secrets retrieval (Vault → KMS → Env Vars → Secure Fallback)
    - LRU caching for performance
    - Automatic key rotation support
    - Comprehensive logging and error handling
    - Development/production environment isolation
    """
    
    def __init__(self, env: str = 'production'):
        self.env = env
        self._secrets_client = None
        self._key_last_rotation = None
        self.logger = self._setup_logging()
        self._init_secrets_engine()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger("secure_config")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _init_secrets_engine(self):
        """Initialize secrets management with priority order"""
        clients_tried = []
        
        try:
            # Priority 1: HashiCorp Vault
            if hvac and os.getenv('VAULT_ADDR'):
                self._secrets_client = hvac.Client(
                    url=os.getenv('VAULT_ADDR'),
                    token=os.getenv('VAULT_TOKEN')
                )
                # Test connection
                if self._secrets_client.is_authenticated():
                    self.logger.info("✅ Vault secrets engine initialized and authenticated")
                    return
                else:
                    clients_tried.append("Vault (authentication failed)")
                    self._secrets_client = None
            
            # Priority 2: AWS KMS
            if boto3 and os.getenv('AWS_ACCESS_KEY_ID'):
                self._secrets_client = boto3.client('kms')
                # Test with a simple operation
                try:
                    self._secrets_client.list_aliases(Limit=1)
                    self.logger.info("✅ KMS secrets engine initialized")
                    return
                except Exception:
                    clients_tried.append("KMS (connection failed)")
                    self._secrets_client = None
            
            # Priority 3: Docker/Kubernetes secrets
            if self._try_docker_secrets():
                self.logger.info("✅ Docker/Kubernetes secrets initialized")
                return
            else:
                clients_tried.append("Docker/Kubernetes")
                
        except Exception as e:
            self.logger.error(f"❌ Secrets engine initialization failed: {e}")
        
        self.logger.warning(f"⚠️ No secure secrets engine available. Tried: {', '.join(clients_tried)}")
    
    def _try_docker_secrets(self) -> bool:
        """Try to load secrets from Docker/Kubernetes mounted volumes"""
        secrets_path = "/run/secrets"
        if os.path.exists(secrets_path):
            # Mark as available but don't set as primary client
            return True
        return False
    
    @lru_cache(maxsize=128)
    def get_secret(self, secret_name: str, default: str = None, use_cache: bool = True) -> str:
        """
        Get secret from secure storage with caching
        
        Args:
            secret_name: Name of the secret to retrieve
            default: Fallback value if secret cannot be retrieved
            use_cache: Whether to use cached result (disable for real-time retrieval)
        
        Returns:
            Secret value as string
        
        Raises:
            SecretRetrievalError: When secret retrieval fails in production
        """
        # Bypass cache if requested
        if not use_cache:
            self.get_secret.cache_clear()
        
        secret_value = None
        retrieval_method = None
        
        try:
            # Method 1: HashiCorp Vault
            if self._secrets_client and hasattr(self._secrets_client, 'read'):
                try:
                    secret = self._secrets_client.read(f"secret/ai_sentinel/{secret_name}")
                    if secret and 'data' in secret:
                        secret_value = secret['data']['value']
                        retrieval_method = "Vault"
                except Exception as e:
                    self.logger.debug(f"Vault retrieval failed for {secret_name}: {e}")
            
            # Method 2: AWS KMS
            if not secret_value and self._secrets_client and hasattr(self._secrets_client, 'decrypt'):
                try:
                    env_value = os.getenv(secret_name)
                    if env_value:
                        response = self._secrets_client.decrypt(
                            CiphertextBlob=base64.b64decode(env_value)
                        )
                        secret_value = response['Plaintext'].decode()
                        retrieval_method = "KMS"
                except Exception as e:
                    self.logger.debug(f"KMS retrieval failed for {secret_name}: {e}")
            
            # Method 3: Docker/Kubernetes secrets
            if not secret_value:
                docker_secret_path = f"/run/secrets/{secret_name}"
                if os.path.exists(docker_secret_path):
                    try:
                        with open(docker_secret_path, 'r') as f:
                            secret_value = f.read().strip()
                            retrieval_method = "Docker/K8s"
                    except Exception as e:
                        self.logger.debug(f"Docker secret retrieval failed: {e}")
            
            # Method 4: Environment variables
            if not secret_value:
                env_value = os.getenv(secret_name)
                if env_value:
                    secret_value = env_value
                    retrieval_method = "Environment"
            
            # Method 5: Default fallback
            if not secret_value and default is not None:
                secret_value = default
                retrieval_method = "Default"
                if self.env == 'production':
                    self.logger.warning(f"Using default value for secret: {secret_name}")
            
            # Validation
            if secret_value is None:
                error_msg = f"Secret '{secret_name}' not found and no default provided"
                if self.env == 'production':
                    raise SecretRetrievalError(error_msg)
                else:
                    self.logger.error(error_msg)
                    return None
            
            if retrieval_method:
                self.logger.debug(f"Retrieved secret '{secret_name}' via {retrieval_method}")
            
            return secret_value
            
        except Exception as e:
            error_msg = f"Failed to retrieve secret '{secret_name}': {str(e)}"
            if self.env == 'production':
                raise SecretRetrievalError(error_msg) from e
            else:
                self.logger.error(error_msg)
                return default
    
    def rotate_encryption_key(self) -> bool:
        """
        Rotate encryption key in secure storage
        
        Returns:
            bool: True if rotation successful
        
        Raises:
            KeyRotationError: If rotation fails in production
        """
        try:
            new_key = Fernet.generate_key()
            encoded_key = base64.b64encode(new_key).decode()
            
            if self._secrets_client and hasattr(self._secrets_client, 'write'):
                # Vault rotation
                self._secrets_client.write(
                    "secret/ai_sentinel/ENCRYPTION_KEY", 
                    {"value": encoded_key}
                )
            elif self._secrets_client and hasattr(self._secrets_client, 'encrypt'):
                # KMS rotation - would need proper key management setup
                self.logger.warning("KMS key rotation requires manual setup")
                return False
            else:
                raise KeyRotationError("No secure storage available for key rotation")
            
            self._key_last_rotation = datetime.now()
            self.get_secret.cache_clear()  # Clear cache after rotation
            
            self.logger.info("✅ Encryption key rotated successfully")
            return True
            
        except Exception as e:
            error_msg = f"Key rotation failed: {str(e)}"
            if self.env == 'production':
                raise KeyRotationError(error_msg) from e
            else:
                self.logger.error(error_msg)
                return False
    
    def should_rotate_key(self, rotation_days: int = 90) -> bool:
        """Check if key should be rotated based on time"""
        if not self._key_last_rotation:
            return True
        
        rotation_period = timedelta(days=rotation_days)
        return datetime.now() - self._key_last_rotation > rotation_period
    
    @property
    def encryption_key(self) -> bytes:
        """Get encryption key securely with production safety"""
        if self.env == 'production':
            key = self.get_secret('ENCRYPTION_KEY')
            if not key:
                raise SecretRetrievalError(
                    "ENCRYPTION_KEY is required in production environment"
                )
        else:
            key = self.get_secret('ENCRYPTION_KEY')
        
        if key and key.startswith('base64:'):
            return base64.b64decode(key[7:])
        elif key:
            return key.encode()
        else:
            # Development fallback with warning
            self.logger.warning("⚠️ Using generated key - NOT FOR PRODUCTION")
            return Fernet.generate_key()
    
    @property
    def database_url(self) -> str:
        """Get database URL with environment-specific handling"""
        if self.env == 'production':
            return self.get_secret(
                'DATABASE_URL', 
                'postgresql://user:pass@localhost/ai_sentinel'
            )
        else:
            return self.get_secret('DATABASE_URL', 'sqlite:///ai_sentinel_dev.db')
    
    @property
    def jwt_secret(self) -> str:
        """Get JWT secret with production validation"""
        secret = self.get_secret('JWT_SECRET', 'dev-secret-change-in-production')
        if self.env == 'production' and secret == 'dev-secret-change-in-production':
            self.logger.warning("Using default JWT secret in production!")
        return secret
    
    @property
    def signing_key_path(self) -> str:
        """Get signing key path"""
        return self.get_secret('SIGNING_KEY_PATH', '/secrets/signing_key.pem')
    
    def get_all_configs(self) -> dict:
        """Get all configuration values (for debugging/auditing)"""
        configs = {
            'environment': self.env,
            'database_url': '[REDACTED]' if 'pass' in str(self.database_url) else self.database_url,
            'has_jwt_secret': bool(self.jwt_secret and self.jwt_secret != 'dev-secret-change-in-production'),
            'has_signing_key': os.path.exists(self.signing_key_path),
            'secrets_engine': type(self._secrets_client).__name__ if self._secrets_client else 'None',
            'key_last_rotation': self._key_last_rotation,
        }
        return configs

# Global config instance with environment detection
config = SecureConfig(os.getenv('ENV', 'development'))

# Optional: Auto-rotate key if needed in production
if os.getenv('ENV') == 'production' and config.should_rotate_key():
    config.logger.info("Key rotation check: considering rotation for production key")