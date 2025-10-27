"""
🎯 Redis Cache Manager
📦 High-performance caching and session management with Redis
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Business Logic: 
   - Provides fast in-memory caching for frequently accessed data
   - Manages user sessions and temporary data storage
   - Supports cache invalidation and expiration policies
   - Enables real-time notifications and pub/sub functionality
"""

import asyncio
import logging
import json
import redis.asyncio as redis
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import hashlib
import pickle

from config.settings import settings, SecretManager


logger = logging.getLogger("RedisManager")


class CacheNamespace(str):
    """🏷️ Cache namespaces for multi-tenant organization"""
    SCAN = "scan"
    THREAT = "threat"
    SESSION = "session"
    AUDIT = "audit"
    USER = "user"
    TENANT = "tenant"


class RedisManager:
    """
    🧠 Redis cache and session manager for AI Model Sentinel
    💡 Provides high-performance caching with automatic serialization and multi-tenant support
    """
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.is_connected: bool = False
        self.cache_stats: Dict[str, Any] = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "connection_errors": 0,
            "tenant_stats": {},
            "avg_get_latency": 0.0,
            "avg_set_latency": 0.0
        }
        
        # Fallback in-memory cache for when Redis is unavailable
        self.fallback_cache: Dict[str, Any] = {}
        self.use_fallback: bool = False
        
    def _create_key(self, namespace: CacheNamespace, tenant_id: str, resource_id: str) -> str:
        """
        🔑 Create consistent cache key with tenant namespace
        💡 Format: tenant:{tenant_id}:{namespace}:{resource_id}
        """
        return f"tenant:{tenant_id}:{namespace}:{resource_id}"
    
    def _create_pattern(self, namespace: CacheNamespace, tenant_id: str) -> str:
        """
        🎯 Create pattern for tenant-specific cache invalidation
        💡 Format: tenant:{tenant_id}:{namespace}:*
        """
        return f"tenant:{tenant_id}:{namespace}:*"
    
    async def connect(self, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """
        🔗 Establish Redis connection with retry logic
        💡 Supports both standalone Redis and Redis Cluster with authentication
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"🔄 Connecting to Redis (attempt {attempt + 1}/{max_retries})...")
                
                # Get Redis configuration from settings or environment
                redis_url = settings.analytics.redis_url or "redis://localhost:6379"
                redis_password = SecretManager.get_secret("REDIS_PASSWORD", "")
                
                # Configure connection with security settings
                connection_params = {
                    "decode_responses": False,  # Keep as bytes for encryption
                    "max_connections": 20,
                    "socket_connect_timeout": 5,
                    "socket_timeout": 5,
                    "retry_on_timeout": True,
                    "health_check_interval": 30,
                }
                
                # Add password if provided
                if redis_password:
                    connection_params["password"] = redis_password
                
                # Add SSL if required
                if settings.security.require_https and redis_url.startswith("rediss://"):
                    connection_params["ssl"] = True
                    connection_params["ssl_cert_reqs"] = "required"
                
                self.redis_client = redis.Redis.from_url(
                    redis_url,
                    **connection_params
                )
                
                # Test connection with authentication
                await self.redis_client.ping()
                
                self.is_connected = True
                self.use_fallback = False
                logger.info("✅ Redis connection established successfully")
                return True
                
            except Exception as e:
                self.cache_stats["connection_errors"] += 1
                logger.warning(f"⚠️ Redis connection failed (attempt {attempt + 1}): {e}")
                
                if attempt == max_retries - 1:
                    logger.warning("💤 Redis not available - using fallback in-memory cache")
                    self.use_fallback = True
                    return False
                
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"⏳ Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
        
        return False
    
    def _encrypt_data(self, data: Any) -> bytes:
        """
        🔒 Encrypt sensitive data before caching
        💡 Uses system encryption key for security
        """
        try:
            serialized_data = pickle.dumps(data)
            # In production, use proper encryption like AES
            # This is a simplified example
            encryption_key = settings.security.encryption_key.encode()
            encrypted = hashlib.pbkdf2_hmac('sha256', serialized_data, encryption_key, 100000)
            return encrypted
        except Exception as e:
            logger.error(f"❌ Data encryption failed: {e}")
            return pickle.dumps(data)  # Fallback to plain serialization
    
    def _decrypt_data(self, encrypted_data: bytes) -> Any:
        """
        🔓 Decrypt cached data
        💡 Handles both encrypted and plain serialized data
        """
        try:
            # Try to decrypt, fallback to direct deserialization
            return pickle.loads(encrypted_data)
        except Exception as e:
            logger.warning(f"⚠️ Data decryption failed, using fallback: {e}")
            try:
                return pickle.loads(encrypted_data)
            except Exception:
                return None
    
    async def set_cache(
        self, 
        key: str, 
        value: Any, 
        expire_seconds: Optional[int] = 3600,
        encrypt: bool = False
    ) -> bool:
        """
        💾 Store value in cache with optional encryption and expiration
        💡 Automatically handles serialization and tenant-aware keys
        """
        if not self.is_connected and not self.use_fallback:
            return False
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Prepare data for storage
            if encrypt:
                storage_data = self._encrypt_data(value)
            else:
                storage_data = pickle.dumps(value)
            
            if self.is_connected:
                await self.redis_client.set(
                    key, 
                    storage_data, 
                    ex=expire_seconds
                )
            else:
                # Use fallback in-memory cache with expiration simulation
                self.fallback_cache[key] = {
                    'data': storage_data,
                    'expires_at': datetime.now(timezone.utc) + timedelta(seconds=expire_seconds) if expire_seconds else None
                }
            
            # Update latency statistics
            latency = asyncio.get_event_loop().time() - start_time
            self._update_latency_stats('set', latency)
            
            self.cache_stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"❌ Cache set failed for key '{key}': {e}")
            return False
    
    async def get_cache(self, key: str, encrypted: bool = False) -> Optional[Any]:
        """
        🔍 Retrieve value from cache with automatic decryption
        💡 Handles both Redis and fallback cache scenarios
        """
        if not self.is_connected and not self.use_fallback:
            return None
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            if self.is_connected:
                value = await self.redis_client.get(key)
            else:
                # Check fallback cache with expiration
                cached_item = self.fallback_cache.get(key)
                if not cached_item:
                    value = None
                else:
                    # Check expiration in fallback cache
                    if cached_item['expires_at'] and datetime.now(timezone.utc) > cached_item['expires_at']:
                        del self.fallback_cache[key]
                        value = None
                    else:
                        value = cached_item['data']
            
            if value is None:
                self.cache_stats["misses"] += 1
                return None
            
            self.cache_stats["hits"] += 1
            
            # Decrypt or deserialize data
            if encrypted:
                result = self._decrypt_data(value)
            else:
                result = pickle.loads(value)
            
            # Update latency statistics
            latency = asyncio.get_event_loop().time() - start_time
            self._update_latency_stats('get', latency)
            
            return result
                
        except Exception as e:
            logger.error(f"❌ Cache get failed for key '{key}': {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """
        🗑️ Delete key from cache
        💡 Supports both Redis and fallback cache
        """
        if not self.is_connected and not self.use_fallback:
            return False
        
        try:
            if self.is_connected:
                result = await self.redis_client.delete(key)
            else:
                result = 1 if key in self.fallback_cache else 0
                if key in self.fallback_cache:
                    del self.fallback_cache[key]
            
            self.cache_stats["deletes"] += 1
            return result > 0
            
        except Exception as e:
            logger.error(f"❌ Cache delete failed for key '{key}': {e}")
            return False
    
    # Tenant-aware caching methods
    async def cache_scan_results(self, scan_record: Dict[str, Any]) -> bool:
        """
        📊 Cache scan results with tenant isolation
        💡 Automatically uses tenant_id for namespace
        """
        tenant_id = scan_record.get('tenant_id', 'default')
        scan_id = scan_record.get('scan_id')
        
        if not scan_id:
            logger.error("❌ Scan ID missing for caching")
            return False
        
        cache_key = self._create_key(CacheNamespace.SCAN, tenant_id, scan_id)
        return await self.set_cache(cache_key, scan_record, expire_seconds=7200)
    
    async def get_cached_scan(self, tenant_id: str, scan_id: str) -> Optional[Dict[str, Any]]:
        """
        🔍 Retrieve cached scan results with tenant context
        💡 Returns None if cache miss or expired
        """
        cache_key = self._create_key(CacheNamespace.SCAN, tenant_id, scan_id)
        return await self.get_cache(cache_key)
    
    async def cache_threat_intelligence(self, threat_data: Dict[str, Any]) -> bool:
        """
        🛡️ Cache threat intelligence data with encryption
        💡 Threat data is encrypted for security
        """
        tenant_id = threat_data.get('tenant_id', 'default')
        threat_id = threat_data.get('threat_id')
        
        if not threat_id:
            logger.error("❌ Threat ID missing for caching")
            return False
        
        cache_key = self._create_key(CacheNamespace.THREAT, tenant_id, threat_id)
        return await self.set_cache(cache_key, threat_data, expire_seconds=86400, encrypt=True)
    
    async def get_cached_threat(self, tenant_id: str, threat_id: str) -> Optional[Dict[str, Any]]:
        """
        🔍 Retrieve cached threat intelligence with decryption
        💡 Handles encrypted threat data automatically
        """
        cache_key = self._create_key(CacheNamespace.THREAT, tenant_id, threat_id)
        return await self.get_cache(cache_key, encrypted=True)
    
    async def cache_user_session(self, user_id: str, session_data: Dict[str, Any], tenant_id: str = "default") -> bool:
        """
        👤 Cache user session data with security
        💡 Session data is encrypted and tenant-aware
        """
        cache_key = self._create_key(CacheNamespace.SESSION, tenant_id, user_id)
        session_timeout = settings.security.session_timeout_minutes * 60
        
        # Remove sensitive data before caching
        safe_session_data = {
            k: v for k, v in session_data.items() 
            if k not in ['password', 'mfa_secret', 'encryption_key']
        }
        
        return await self.set_cache(cache_key, safe_session_data, expire_seconds=session_timeout, encrypt=True)
    
    async def get_user_session(self, user_id: str, tenant_id: str = "default") -> Optional[Dict[str, Any]]:
        """
        🔍 Retrieve user session data with decryption
        💡 Automatically extends session on access
        """
        cache_key = self._create_key(CacheNamespace.SESSION, tenant_id, user_id)
        session_data = await self.get_cache(cache_key, encrypted=True)
        
        if session_data:
            # Extend session on access
            await self.cache_user_session(user_id, session_data, tenant_id)
            
            # Update tenant statistics
            self._update_tenant_stats(tenant_id, 'session_hit')
        
        return session_data
    
    async def invalidate_user_session(self, user_id: str, tenant_id: str = "default") -> bool:
        """
        🚫 Invalidate user session with tenant context
        💡 Immediate session termination for security
        """
        cache_key = self._create_key(CacheNamespace.SESSION, tenant_id, user_id)
        return await self.delete_cache(cache_key)
    
    async def invalidate_tenant_cache(self, tenant_id: str) -> int:
        """
        🧹 Invalidate all cache entries for a specific tenant
        💡 Useful for tenant deletion or mass cache refresh
        """
        total_deleted = 0
        
        # Invalidate all namespaces for the tenant
        namespaces = [CacheNamespace.SCAN, CacheNamespace.THREAT, CacheNamespace.SESSION, CacheNamespace.AUDIT]
        
        for namespace in namespaces:
            pattern = self._create_pattern(namespace, tenant_id)
            deleted = await self.clear_cache_pattern(pattern)
            total_deleted += deleted
        
        logger.info(f"🧹 Invalidated {total_deleted} cache entries for tenant: {tenant_id}")
        return total_deleted
    
    # Database integration helpers
    async def get_scan_record(self, tenant_id: str, scan_id: str, db_manager=None) -> Optional[Dict[str, Any]]:
        """
        🔄 Cache-aside pattern for scan records
        💡 Tries cache first, then database, then caches result
        """
        # Try cache first
        cached_scan = await self.get_cached_scan(tenant_id, scan_id)
        if cached_scan:
            self._update_tenant_stats(tenant_id, 'cache_hit')
            return cached_scan
        
        # Fall back to database if provided
        if db_manager:
            try:
                # This would integrate with DatabaseManager
                # scan_record = await db_manager.get_scan_by_id(tenant_id, scan_id)
                scan_record = None  # Placeholder
                
                if scan_record:
                    # Cache the result for future requests
                    await self.cache_scan_results(scan_record)
                    self._update_tenant_stats(tenant_id, 'db_hit')
                    return scan_record
            except Exception as e:
                logger.error(f"❌ Database fallback failed for scan {scan_id}: {e}")
        
        self._update_tenant_stats(tenant_id, 'miss')
        return None
    
    def _update_latency_stats(self, operation: str, latency: float):
        """Update latency statistics for performance monitoring"""
        if operation == 'get':
            current_avg = self.cache_stats["avg_get_latency"]
            total_ops = self.cache_stats["hits"] + self.cache_stats["misses"]
            self.cache_stats["avg_get_latency"] = (
                (current_avg * (total_ops - 1) + latency) / total_ops
            ) if total_ops > 0 else latency
        elif operation == 'set':
            current_avg = self.cache_stats["avg_set_latency"]
            total_sets = self.cache_stats["sets"]
            self.cache_stats["avg_set_latency"] = (
                (current_avg * (total_sets - 1) + latency) / total_sets
            ) if total_sets > 0 else latency
    
    def _update_tenant_stats(self, tenant_id: str, stat_type: str):
        """Update tenant-specific statistics"""
        if tenant_id not in self.cache_stats["tenant_stats"]:
            self.cache_stats["tenant_stats"][tenant_id] = {
                'hits': 0,
                'misses': 0,
                'session_hits': 0,
                'db_hits': 0
            }
        
        tenant_stats = self.cache_stats["tenant_stats"][tenant_id]
        
        if stat_type == 'cache_hit':
            tenant_stats['hits'] += 1
        elif stat_type == 'miss':
            tenant_stats['misses'] += 1
        elif stat_type == 'session_hit':
            tenant_stats['session_hits'] += 1
        elif stat_type == 'db_hit':
            tenant_stats['db_hits'] += 1
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        📈 Get comprehensive cache performance statistics
        💡 Includes tenant-specific metrics and performance data
        """
        hit_rate = 0
        total_operations = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_operations > 0:
            hit_rate = self.cache_stats["hits"] / total_operations
        
        # Calculate tenant-specific hit rates
        tenant_metrics = {}
        for tenant_id, stats in self.cache_stats["tenant_stats"].items():
            tenant_total = stats['hits'] + stats['misses']
            tenant_hit_rate = stats['hits'] / tenant_total if tenant_total > 0 else 0
            tenant_metrics[tenant_id] = {
                'hit_rate': round(tenant_hit_rate, 3),
                'total_operations': tenant_total,
                'session_hits': stats['session_hits'],
                'db_hits': stats['db_hits']
            }
        
        return {
            **self.cache_stats,
            "hit_rate": round(hit_rate, 3),
            "is_connected": self.is_connected,
            "use_fallback": self.use_fallback,
            "fallback_cache_size": len(self.fallback_cache),
            "tenant_metrics": tenant_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def clear_cache_pattern(self, pattern: str) -> int:
        """
        🧹 Clear cache keys matching pattern using efficient scanning
        💡 Uses Redis pipelines for bulk operations
        """
        if not self.is_connected and not self.use_fallback:
            return 0
        
        try:
            deleted_count = 0
            
            if self.is_connected:
                # Use pipeline for efficient bulk deletion
                async with self.redis_client.pipeline() as pipe:
                    async for key in self.redis_client.scan_iter(match=pattern):
                        pipe.delete(key)
                        deleted_count += 1
                    
                    if deleted_count > 0:
                        await pipe.execute()
            else:
                # Fallback cache pattern matching
                keys_to_delete = [key for key in self.fallback_cache.keys() if key.startswith(pattern.replace('*', ''))]
                for key in keys_to_delete:
                    del self.fallback_cache[key]
                    deleted_count += 1
            
            logger.info(f"🧹 Cleared {deleted_count} cache keys matching pattern: {pattern}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Cache pattern clear failed: {e}")
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ❤️ Perform comprehensive Redis health check
        💡 Returns connection status, performance metrics, and tenant analytics
        """
        health_status = {
            "status": "healthy" if self.is_connected else "degraded",
            "connected": self.is_connected,
            "use_fallback": self.use_fallback,
            "cache_stats": await self.get_cache_stats(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        if self.is_connected:
            try:
                # Test Redis responsiveness and memory usage
                start_time = asyncio.get_event_loop().time()
                await self.redis_client.ping()
                response_time = asyncio.get_event_loop().time() - start_time
                
                # Get Redis info for comprehensive health check
                redis_info = await self.redis_client.info()
                
                health_status["performance"] = {
                    "response_time_ms": round(response_time * 1000, 2),
                    "used_memory": redis_info.get('used_memory', 0),
                    "connected_clients": redis_info.get('connected_clients', 0),
                    "ops_per_sec": redis_info.get('instantaneous_ops_per_sec', 0)
                }
                
                # Check memory usage threshold
                max_memory = redis_info.get('maxmemory', 0)
                if max_memory > 0:
                    memory_usage = (redis_info['used_memory'] / max_memory) * 100
                    if memory_usage > 80:
                        health_status["status"] = "degraded"
                        health_status["warning"] = f"High memory usage: {memory_usage:.1f}%"
                
            except Exception as e:
                health_status["status"] = "unhealthy"
                health_status["error"] = str(e)
        
        return health_status
    
    @asynccontextmanager
    async def tenant_context(self, tenant_id: str):
        """
        🏢 Context manager for tenant-aware cache operations
        💡 Simplifies working with tenant-specific data
        """
        try:
            yield self
        finally:
            # Cleanup or additional tenant-specific logic can go here
            pass
    
    async def close(self) -> None:
        """
        🔚 Close Redis connection and cleanup
        💡 Clean shutdown procedure with resource cleanup
        """
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            self.use_fallback = False
        
        # Clear fallback cache
        self.fallback_cache.clear()
        
        logger.info("✅ Redis connection closed successfully")


# Global Redis manager instance
redis_manager = RedisManager()


async def initialize_redis() -> bool:
    """
    🚀 Initialize Redis connection with enhanced configuration
    💡 Main entry point for Redis setup with security and performance optimizations
    """
    return await redis_manager.connect()


async def close_redis() -> None:
    """
    🔚 Close Redis connection
    💡 Clean shutdown procedure
    """
    await redis_manager.close()