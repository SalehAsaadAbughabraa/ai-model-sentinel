"""
🎯 Encryption and Security Utilities
📦 Provides cryptographic functions for data protection and security
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Business Logic: 
   - Implements strong encryption for sensitive data
   - Provides secure key management and rotation
   - Supports multiple encryption algorithms
   - Ensures data integrity and confidentiality
   - Multi-tenant key isolation and management
"""

import os
import base64
import logging
import asyncio
import secrets
from typing import Optional, Union, Dict, Any, List
from datetime import datetime, timezone, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
import jwt
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings, SecretManager


logger = logging.getLogger("EncryptionManager")


class KeyManager:
    """
    🔑 Advanced key management with rotation and multi-tenant support
    💡 Provides secure key generation, storage, and rotation
    """
    
    def __init__(self):
        self.current_keys: Dict[str, Any] = {}
        self.key_versions: Dict[str, List[str]] = {}
        self.key_cache: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialized = False
        
    async def initialize(self):
        """Initialize key manager asynchronously"""
        if self._initialized:
            return
            
        # Initialize default tenant
        await self.generate_fernet_key("default")
        self._initialized = True
        
    def generate_key_id(self, tenant_id: str, key_type: str) -> str:
        """Generate unique key identifier"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{tenant_id}_{key_type}_{timestamp}"
    
    async def generate_fernet_key(self, tenant_id: str = "default") -> str:
        """Generate Fernet key for tenant"""
        key_id = self.generate_key_id(tenant_id, "fernet")
        key = Fernet.generate_key()
        
        self.current_keys[key_id] = {
            "key": key,
            "created_at": datetime.now(timezone.utc),
            "tenant_id": tenant_id,
            "type": "fernet"
        }
        
        # Store previous versions for key rotation
        if tenant_id not in self.key_versions:
            self.key_versions[tenant_id] = []
        self.key_versions[tenant_id].append(key_id)
        
        logger.info(f"Generated Fernet key for tenant {tenant_id}")
        return key_id
    
    async def get_fernet_key(self, key_id: str) -> Optional[bytes]:
        """Retrieve Fernet key by ID"""
        if key_id in self.key_cache:
            return self.key_cache[key_id]
        
        key_data = self.current_keys.get(key_id)
        if key_data:
            self.key_cache[key_id] = key_data["key"]
            return key_data["key"]
        
        return None
    
    async def rotate_tenant_keys(self, tenant_id: str) -> bool:
        """Rotate all keys for a tenant"""
        try:
            # Generate new keys
            new_fernet_key_id = await self.generate_fernet_key(tenant_id)
            
            # Keep only last 3 versions
            if tenant_id in self.key_versions:
                self.key_versions[tenant_id] = self.key_versions[tenant_id][-3:]
            
            logger.info(f"Rotated keys for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed for tenant {tenant_id}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """Key management health check"""
        return {
            "status": "healthy",
            "total_keys": len(self.current_keys),
            "tenants_with_keys": len(self.key_versions),
            "key_cache_size": len(self.key_cache),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class MultiLayerEncryption:
    """
    🛡️ Multi-layer encryption with AES-GCM and Fernet
    💡 Provides field-level and record-level encryption
    """
    
    def __init__(self, key_manager: KeyManager):
        self.key_manager = key_manager
        self.aes_cache: Dict[str, Any] = {}
    
    async def encrypt_aes_gcm(self, plaintext: str, tenant_id: str = "default", 
                            associated_data: bytes = b"") -> Dict[str, str]:
        """Encrypt using AES-GCM with authentication"""
        try:
            # Generate unique key for this operation
            key = os.urandom(32)
            nonce = os.urandom(12)
            
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
            encryptor = cipher.encryptor()
            
            if associated_data:
                encryptor.authenticate_additional_data(associated_data)
            
            ciphertext = encryptor.update(plaintext.encode()) + encryptor.finalize()
            
            return {
                "ciphertext": base64.urlsafe_b64encode(ciphertext).decode(),
                "key": base64.urlsafe_b64encode(key).decode(),
                "nonce": base64.urlsafe_b64encode(nonce).decode(),
                "tag": base64.urlsafe_b64encode(encryptor.tag).decode()
            }
            
        except Exception as e:
            logger.error(f"AES-GCM encryption failed: {e}")
            raise
    
    async def decrypt_aes_gcm(self, encrypted_data: Dict[str, str], 
                            associated_data: bytes = b"") -> str:
        """Decrypt AES-GCM encrypted data"""
        try:
            key = base64.urlsafe_b64decode(encrypted_data["key"])
            nonce = base64.urlsafe_b64decode(encrypted_data["nonce"])
            ciphertext = base64.urlsafe_b64decode(encrypted_data["ciphertext"])
            tag = base64.urlsafe_b64decode(encrypted_data["tag"])
            
            cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
            decryptor = cipher.decryptor()
            
            if associated_data:
                decryptor.authenticate_additional_data(associated_data)
            
            plaintext = decryptor.update(ciphertext) + decryptor.finalize()
            return plaintext.decode()
            
        except Exception as e:
            logger.error(f"AES-GCM decryption failed: {e}")
            raise
    
    async def encrypt_field_level(self, field_value: str, field_name: str, 
                                tenant_id: str = "default") -> str:
        """Field-level encryption with context binding"""
        # Add context to prevent field swapping
        contextual_data = f"{tenant_id}:{field_name}:{field_value}"
        
        # Use Fernet for field-level encryption
        fernet_key_id = await self._get_tenant_fernet_key(tenant_id)
        fernet_key = await self.key_manager.get_fernet_key(fernet_key_id)
        
        if not fernet_key:
            raise ValueError(f"No Fernet key found for tenant {tenant_id}")
        
        cipher = Fernet(fernet_key)
        encrypted_data = cipher.encrypt(contextual_data.encode())
        return base64.urlsafe_b64encode(encrypted_data).decode()
    
    async def decrypt_field_level(self, encrypted_value: str, field_name: str,
                                tenant_id: str = "default") -> Optional[str]:
        """Field-level decryption with context verification"""
        try:
            fernet_key_id = await self._get_tenant_fernet_key(tenant_id)
            fernet_key = await self.key_manager.get_fernet_key(fernet_key_id)
            
            if not fernet_key:
                return None
            
            cipher = Fernet(fernet_key)
            encrypted_data = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_data = cipher.decrypt(encrypted_data).decode()
            
            # Verify context
            parts = decrypted_data.split(':', 2)
            if len(parts) != 3 or parts[0] != tenant_id or parts[1] != field_name:
                logger.warning(f"Field context mismatch: {parts}")
                return None
            
            return parts[2]
            
        except Exception as e:
            logger.error(f"Field-level decryption failed: {e}")
            return None
    
    async def _get_tenant_fernet_key(self, tenant_id: str) -> str:
        """Get or create Fernet key for tenant"""
        # Look for existing key
        for key_id, key_data in self.key_manager.current_keys.items():
            if key_data.get("tenant_id") == tenant_id and key_data.get("type") == "fernet":
                return key_id
        
        # Create new key if none exists
        return await self.key_manager.generate_fernet_key(tenant_id)


class AdvancedPasswordManager:
    """
    🔐 Advanced password management with Argon2 and bcrypt support
    💡 Provides secure password hashing and verification
    """
    
    def __init__(self):
        self.argon2_available = False
        self.bcrypt_available = False
        self._initialize_hashers()
    
    def _initialize_hashers(self):
        """Initialize password hashing libraries"""
        try:
            import argon2
            self.argon2_available = True
            self.argon2_hasher = argon2.PasswordHasher()
        except ImportError:
            self.argon2_available = False
            logger.warning("Argon2 not available, using fallback algorithms")
        
        try:
            import bcrypt
            self.bcrypt_available = True
            self.bcrypt = bcrypt
        except ImportError:
            self.bcrypt_available = False
            logger.warning("bcrypt not available, using fallback algorithms")
    
    async def hash_password(self, password: str, algorithm: str = "argon2") -> str:
        """Hash password using specified algorithm"""
        if algorithm == "argon2" and self.argon2_available:
            return await self._hash_argon2(password)
        elif algorithm == "bcrypt" and self.bcrypt_available:
            return await self._hash_bcrypt(password)
        else:
            return await self._hash_pbkdf2(password)  # Fallback
    
    async def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            if password_hash.startswith("argon2"):
                return await self._verify_argon2(password, password_hash)
            elif password_hash.startswith("bcrypt"):
                return await self._verify_bcrypt(password, password_hash)
            elif password_hash.startswith("pbkdf2"):
                return await self._verify_pbkdf2(password, password_hash)
            else:
                logger.warning("Unknown password hash format")
                return False
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    async def _hash_argon2(self, password: str) -> str:
        """Hash using Argon2"""
        if not self.argon2_available:
            raise RuntimeError("Argon2 not available")
        
        loop = asyncio.get_event_loop()
        hashed = await loop.run_in_executor(
            None, self.argon2_hasher.hash, password
        )
        return f"argon2${hashed}"
    
    async def _verify_argon2(self, password: str, password_hash: str) -> bool:
        """Verify Argon2 hash"""
        if not self.argon2_available:
            return False
        
        try:
            actual_hash = password_hash.split('$', 1)[1]
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.argon2_hasher.verify, actual_hash, password
            )
        except Exception:
            return False
    
    async def _hash_bcrypt(self, password: str) -> str:
        """Hash using bcrypt"""
        if not self.bcrypt_available:
            raise RuntimeError("bcrypt not available")
        
        loop = asyncio.get_event_loop()
        hashed = await loop.run_in_executor(
            None, self.bcrypt.hashpw, password.encode(), self.bcrypt.gensalt()
        )
        return f"bcrypt${hashed.decode()}"
    
    async def _verify_bcrypt(self, password: str, password_hash: str) -> bool:
        """Verify bcrypt hash"""
        if not self.bcrypt_available:
            return False
        
        try:
            actual_hash = password_hash.split('$', 1)[1]
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.bcrypt.checkpw, password.encode(), actual_hash.encode()
            )
        except Exception:
            return False
    
    async def _hash_pbkdf2(self, password: str) -> str:
        """Hash using PBKDF2 (fallback)"""
        salt = secrets.token_bytes(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return f"pbkdf2_sha256${base64.urlsafe_b64encode(salt).decode()}${key.decode()}"
    
    async def _verify_pbkdf2(self, password: str, password_hash: str) -> bool:
        """Verify PBKDF2 hash"""
        try:
            parts = password_hash.split('$')
            if len(parts) != 3 or parts[0] != 'pbkdf2_sha256':
                return False
            
            salt = base64.urlsafe_b64decode(parts[1].encode())
            stored_key = parts[2]
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
                backend=default_backend()
            )
            computed_key = base64.urlsafe_b64encode(kdf.derive(password.encode())).decode()
            
            return secrets.compare_digest(stored_key, computed_key)
        except Exception:
            return False


class TokenManager:
    """
    🎫 Advanced token management with JWT and encryption
    💡 Provides secure token generation, validation, and management
    """
    
    def __init__(self, encryption_manager: 'EncryptionManager'):
        self.encryption_manager = encryption_manager
        self.token_blacklist: set = set()
    
    async def generate_jwt_token(self, payload: Dict[str, Any], 
                               tenant_id: str = "default",
                               expires_minutes: int = 60) -> str:
        """Generate JWT token with encryption"""
        # Add standard claims
        now = datetime.now(timezone.utc)
        payload.update({
            "iss": "AI_Model_Sentinel",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
            "tenant_id": tenant_id,
            "jti": secrets.token_urlsafe(16)  # Unique token ID
        })
        
        # Encrypt sensitive payload data
        encrypted_payload = await self._encrypt_token_payload(payload, tenant_id)
        
        # Sign with tenant-specific secret
        secret = await self._get_tenant_secret(tenant_id)
        token = jwt.encode(encrypted_payload, secret, algorithm="HS256")
        
        logger.debug(f"Generated JWT token for tenant {tenant_id}")
        return token
    
    async def verify_jwt_token(self, token: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            if token in self.token_blacklist:
                logger.warning("Token is blacklisted")
                return None
            
            secret = await self._get_tenant_secret(tenant_id)
            payload = jwt.decode(token, secret, algorithms=["HS256"])
            
            # Decrypt token payload
            decrypted_payload = await self._decrypt_token_payload(payload, tenant_id)
            
            # Validate expiration
            current_timestamp = datetime.now(timezone.utc).timestamp()
            if decrypted_payload.get("exp", 0) < current_timestamp:
                logger.warning("Token has expired")
                return None
            
            logger.debug(f"Verified JWT token for tenant {tenant_id}")
            return decrypted_payload
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke token by adding to blacklist"""
        try:
            self.token_blacklist.add(token)
            logger.info("Token revoked successfully")
            return True
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False
    
    async def _encrypt_token_payload(self, payload: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        """Encrypt sensitive token payload data"""
        encrypted_payload = payload.copy()
        
        # Encrypt sensitive fields
        sensitive_fields = ['user_id', 'email', 'permissions']
        for field in sensitive_fields:
            if field in encrypted_payload:
                encrypted_value = await self.encryption_manager.encrypt_field_level(
                    str(encrypted_payload[field]), field, tenant_id
                )
                encrypted_payload[field] = encrypted_value
        
        return encrypted_payload
    
    async def _decrypt_token_payload(self, payload: Dict[str, Any], tenant_id: str) -> Dict[str, Any]:
        """Decrypt token payload data"""
        decrypted_payload = payload.copy()
        
        # Decrypt sensitive fields
        sensitive_fields = ['user_id', 'email', 'permissions']
        for field in sensitive_fields:
            if field in decrypted_payload:
                decrypted_value = await self.encryption_manager.decrypt_field_level(
                    decrypted_payload[field], field, tenant_id
                )
                if decrypted_value:
                    # Convert back to appropriate type if needed
                    if field == 'permissions' and isinstance(decrypted_value, str):
                        try:
                            import json
                            decrypted_payload[field] = json.loads(decrypted_value)
                        except:
                            decrypted_payload[field] = decrypted_value
                    else:
                        decrypted_payload[field] = decrypted_value
        
        return decrypted_payload
    
    async def _get_tenant_secret(self, tenant_id: str) -> str:
        """Get tenant-specific JWT secret"""
        # In production, this would fetch from secure storage
        base_secret = getattr(settings.security, 'jwt_secret', 'fallback-secret-key-change-in-production')
        return f"{base_secret}:{tenant_id}"


class EncryptionManager:
    """
    🔐 Comprehensive encryption manager for AI Model Sentinel
    💡 Provides multiple encryption methods and secure key management with multi-tenant support
    """
    
    def __init__(self):
        self.key_manager = KeyManager()
        self.multi_layer_encryption = MultiLayerEncryption(self.key_manager)
        self.password_manager = AdvancedPasswordManager()
        self.token_manager = TokenManager(self)
        self.fernet_cipher: Optional[Fernet] = None
        self.performance_metrics: Dict[str, List[float]] = {
            "encryption_times": [],
            "decryption_times": [],
            "hashing_times": []
        }
        self._initialized = False
        
    async def initialize(self):
        """Initialize encryption manager asynchronously"""
        if self._initialized:
            return
            
        await self._initialize_ciphers()
        self._initialized = True
    
    async def _initialize_ciphers(self) -> None:
        """Initialize encryption ciphers with configuration from settings"""
        try:
            # Derive Fernet key from configuration for backward compatibility
            encryption_key = getattr(settings.security, 'encryption_key', 'default-encryption-key-change-in-production')
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'ai_sentinel_salt',
                iterations=100000,
                backend=default_backend()
            )
            fernet_key = base64.urlsafe_b64encode(kdf.derive(encryption_key.encode()))
            self.fernet_cipher = Fernet(fernet_key)
            
            # Initialize key manager
            await self.key_manager.initialize()
            
            logger.info("Encryption ciphers initialized successfully")
            
        except Exception as e:
            logger.error(f"Encryption initialization failed: {e}")
            raise
    
    async def encrypt_string(self, plaintext: str, tenant_id: str = "default") -> str:
        """
        🔒 Encrypt a string using Fernet symmetric encryption
        💡 Suitable for encrypting sensitive configuration and user data
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use tenant-specific encryption
            encrypted = await self.multi_layer_encryption.encrypt_field_level(
                plaintext, "generic_string", tenant_id
            )
            
            # Update performance metrics
            self._update_performance_metrics("encryption_times", start_time)
            
            return encrypted
            
        except Exception as e:
            logger.error(f"String encryption failed for tenant {tenant_id}: {e}")
            raise
    
    async def decrypt_string(self, encrypted_text: str, tenant_id: str = "default") -> str:
        """
        🔓 Decrypt a string encrypted with Fernet
        💡 Returns original plaintext or raises exception on failure
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use tenant-specific decryption
            decrypted = await self.multi_layer_encryption.decrypt_field_level(
                encrypted_text, "generic_string", tenant_id
            )
            
            if decrypted is None:
                raise ValueError("Decryption failed or context mismatch")
            
            # Update performance metrics
            self._update_performance_metrics("decryption_times", start_time)
            
            return decrypted
            
        except Exception as e:
            logger.error(f"String decryption failed for tenant {tenant_id}: {e}")
            raise
    
    async def encrypt_dictionary(self, data: dict, tenant_id: str = "default") -> dict:
        """
        📁 Encrypt all string values in a dictionary with field-level encryption
        💡 Useful for encrypting configuration objects and sensitive data structures
        """
        encrypted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str) and self._should_encrypt_field(key):
                encrypted_data[key] = await self.encrypt_string(value, tenant_id)
            elif isinstance(value, dict):
                encrypted_data[key] = await self.encrypt_dictionary(value, tenant_id)
            elif isinstance(value, list):
                encrypted_data[key] = await self._encrypt_list(value, tenant_id)
            else:
                encrypted_data[key] = value
        
        return encrypted_data
    
    async def decrypt_dictionary(self, encrypted_data: dict, tenant_id: str = "default") -> dict:
        """
        📁 Decrypt all encrypted string values in a dictionary
        💡 Reverses encrypt_dictionary operation
        """
        decrypted_data = {}
        
        for key, value in encrypted_data.items():
            if isinstance(value, str):
                try:
                    decrypted_data[key] = await self.decrypt_string(value, tenant_id)
                except:
                    # If decryption fails, assume it wasn't encrypted
                    decrypted_data[key] = value
            elif isinstance(value, dict):
                decrypted_data[key] = await self.decrypt_dictionary(value, tenant_id)
            elif isinstance(value, list):
                decrypted_data[key] = await self._decrypt_list(value, tenant_id)
            else:
                decrypted_data[key] = value
        
        return decrypted_data
    
    async def _encrypt_list(self, data_list: list, tenant_id: str) -> list:
        """Encrypt list items"""
        encrypted_list = []
        for item in data_list:
            if isinstance(item, str) and self._should_encrypt_field("list_item"):
                encrypted_list.append(await self.encrypt_string(item, tenant_id))
            elif isinstance(item, dict):
                encrypted_list.append(await self.encrypt_dictionary(item, tenant_id))
            elif isinstance(item, list):
                encrypted_list.append(await self._encrypt_list(item, tenant_id))
            else:
                encrypted_list.append(item)
        return encrypted_list
    
    async def _decrypt_list(self, data_list: list, tenant_id: str) -> list:
        """Decrypt list items"""
        decrypted_list = []
        for item in data_list:
            if isinstance(item, str):
                try:
                    decrypted_list.append(await self.decrypt_string(item, tenant_id))
                except:
                    decrypted_list.append(item)
            elif isinstance(item, dict):
                decrypted_list.append(await self.decrypt_dictionary(item, tenant_id))
            elif isinstance(item, list):
                decrypted_list.append(await self._decrypt_list(item, tenant_id))
            else:
                decrypted_list.append(item)
        return decrypted_list
    
    def _should_encrypt_field(self, field_name: str) -> bool:
        """Determine if a field should be encrypted"""
        # Define sensitive fields that should always be encrypted
        sensitive_fields = {
            'password', 'secret', 'key', 'token', 'credential', 'api_key',
            'access_key', 'secret_key', 'private_key', 'jwt_secret'
        }
        return any(sensitive in field_name.lower() for sensitive in sensitive_fields)
    
    def _update_performance_metrics(self, metric_type: str, start_time: float):
        """Update performance metrics"""
        duration = asyncio.get_event_loop().time() - start_time
        self.performance_metrics[metric_type].append(duration)
        
        # Keep only last 100 measurements
        if len(self.performance_metrics[metric_type]) > 100:
            self.performance_metrics[metric_type] = self.performance_metrics[metric_type][-100:]
    
    # Delegate to specialized managers
    async def hash_password(self, password: str, algorithm: str = "argon2") -> str:
        """Hash password using advanced algorithms"""
        start_time = asyncio.get_event_loop().time()
        result = await self.password_manager.hash_password(password, algorithm)
        self._update_performance_metrics("hashing_times", start_time)
        return result
    
    async def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        return await self.password_manager.verify_password(password, password_hash)
    
    async def generate_jwt_token(self, payload: Dict[str, Any], 
                               tenant_id: str = "default",
                               expires_minutes: int = 60) -> str:
        """Generate JWT token"""
        return await self.token_manager.generate_jwt_token(payload, tenant_id, expires_minutes)
    
    async def verify_jwt_token(self, token: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        return await self.token_manager.verify_jwt_token(token, tenant_id)
    
    async def revoke_token(self, token: str) -> bool:
        """Revoke JWT token"""
        return await self.token_manager.revoke_token(token)
    
    async def encrypt_field_level(self, field_value: str, field_name: str, 
                                tenant_id: str = "default") -> str:
        """Field-level encryption"""
        return await self.multi_layer_encryption.encrypt_field_level(
            field_value, field_name, tenant_id
        )
    
    async def decrypt_field_level(self, encrypted_value: str, field_name: str,
                                tenant_id: str = "default") -> Optional[str]:
        """Field-level decryption"""
        return await self.multi_layer_encryption.decrypt_field_level(
            encrypted_value, field_name, tenant_id
        )
    
    async def rotate_tenant_keys(self, tenant_id: str) -> bool:
        """Rotate keys for a tenant"""
        return await self.key_manager.rotate_tenant_keys(tenant_id)
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def generate_hmac_signature(self, data: str, key: Optional[str] = None) -> str:
        """Generate HMAC signature for data integrity verification"""
        if key is None:
            key = getattr(settings.security, 'encryption_key', 'default-key')
        
        h = hmac.HMAC(key.encode(), hashes.SHA256(), backend=default_backend())
        h.update(data.encode())
        return base64.urlsafe_b64encode(h.finalize()).decode()
    
    def verify_hmac_signature(self, data: str, signature: str, key: Optional[str] = None) -> bool:
        """Verify HMAC signature for data integrity"""
        try:
            expected_signature = self.generate_hmac_signature(data, key)
            return secrets.compare_digest(signature, expected_signature)
        except Exception:
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ❤️ Comprehensive encryption system health check
        💡 Verifies all encryption components and performance metrics
        """
        try:
            # Ensure initialized
            if not self._initialized:
                await self.initialize()
            
            # Test basic encryption/decryption
            test_data = "AI Model Sentinel Health Check"
            encrypted = await self.encrypt_string(test_data)
            decrypted = await self.decrypt_string(encrypted)
            
            # Test password hashing
            test_password = "test_password_123"
            password_hash = await self.hash_password(test_password)
            password_valid = await self.verify_password(test_password, password_hash)
            
            # Test JWT tokens
            test_payload = {"user_id": "test_user", "role": "admin"}
            jwt_token = await self.generate_jwt_token(test_payload)
            jwt_valid = await self.verify_jwt_token(jwt_token) is not None
            
            # Calculate performance metrics
            avg_encryption_time = sum(self.performance_metrics["encryption_times"]) / len(self.performance_metrics["encryption_times"]) if self.performance_metrics["encryption_times"] else 0
            avg_decryption_time = sum(self.performance_metrics["decryption_times"]) / len(self.performance_metrics["decryption_times"]) if self.performance_metrics["decryption_times"] else 0
            
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "encryption_decryption": "working" if decrypted == test_data else "failed",
                    "password_hashing": "working" if password_valid else "failed",
                    "jwt_tokens": "working" if jwt_valid else "failed",
                    "key_management": self.key_manager.health_check()
                },
                "performance": {
                    "avg_encryption_time_ms": round(avg_encryption_time * 1000, 2),
                    "avg_decryption_time_ms": round(avg_decryption_time * 1000, 2),
                    "total_operations": sum(len(times) for times in self.performance_metrics.values())
                },
                "tenant_metrics": {
                    "total_tenants": len(self.key_manager.key_versions),
                    "total_keys": len(self.key_manager.current_keys)
                }
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global encryption manager instance
encryption_manager = EncryptionManager()


async def initialize_encryption() -> bool:
    """
    🚀 Initialize encryption system
    💡 Main entry point for encryption setup
    """
    try:
        await encryption_manager.initialize()
        
        health = await encryption_manager.health_check()
        if health["status"] == "healthy":
            logger.info("Encryption system initialized successfully")
            return True
        else:
            logger.error("Encryption system health check failed")
            return False
    except Exception as e:
        logger.error(f"Encryption system initialization failed: {e}")
        return False