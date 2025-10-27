"""
AI Model Sentinel v2.0.0 - Advanced Database Management
Production-Grade Database Layer with Connection Pooling and Monitoring
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, Dict, Any
from datetime import datetime

import asyncpg
from asyncpg import Connection, Pool
import redis.asyncio as redis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool

from app import config
from app.monitoring.metrics import MetricsCollector

# Initialize metrics
metrics = MetricsCollector()

class DatabaseManager:
    """Advanced database management with connection pooling and monitoring"""
    
    _pool: Optional[Pool] = None
    _redis_client: Optional[redis.Redis] = None
    _sqlalchemy_engine = None
    _sqlalchemy_session = None
    
    @classmethod
    async def initialize(cls):
        """Initialize database connections"""
        try:
            # Initialize PostgreSQL connection pool
            cls._pool = await asyncpg.create_pool(
                dsn=os.getenv("DATABASE_URL"),
                min_size=5,
                max_size=20,
                max_inactive_connection_lifetime=300,
                command_timeout=60,
                server_settings={
                    'application_name': 'ai-sentinel-v2',
                    'jit': 'off'
                }
            )
            
            # Initialize Redis client
            cls._redis_client = redis.from_url(
                os.getenv("REDIS_URL"),
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True
            )
            
            # Initialize SQLAlchemy for complex queries
            cls._sqlalchemy_engine = create_async_engine(
                os.getenv("DATABASE_URL").replace("postgresql://", "postgresql+asyncpg://"),
                poolclass=AsyncAdaptedQueuePool,
                pool_size=config.config.DB_POOL_SIZE,
                max_overflow=config.config.DB_MAX_OVERFLOW,
                pool_recycle=config.config.DB_POOL_RECYCLE,
                echo=False
            )
            
            cls._sqlalchemy_session = async_sessionmaker(
                cls._sqlalchemy_engine,
                expire_on_commit=False,
                class_=AsyncSession
            )
            
            # Test connections
            await cls.test_connections()
            logging.info("✅ Database connections initialized successfully")
            
        except Exception as e:
            logging.error(f"❌ Database initialization failed: {e}")
            raise
    
    @classmethod
    async def test_connections(cls):
        """Test database connections"""
        # Test PostgreSQL
        async with cls._pool.acquire() as conn:
            await conn.execute("SELECT 1")
        
        # Test Redis
        await cls._redis_client.ping()
        
        # Test SQLAlchemy
        async with cls._sqlalchemy_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    
    @classmethod
    async def close(cls):
        """Close database connections"""
        if cls._pool:
            await cls._pool.close()
        if cls._redis_client:
            await cls._redis_client.close()
        if cls._sqlalchemy_engine:
            await cls._sqlalchemy_engine.dispose()
        
        logging.info("✅ Database connections closed")
    
    @classmethod
    @asynccontextmanager
    async def get_connection(cls) -> AsyncGenerator[Connection, None]:
        """Get database connection from pool"""
        if not cls._pool:
            raise RuntimeError("Database not initialized")
        
        start_time = datetime.now()
        metrics.record_db_connection()
        
        try:
            async with cls._pool.acquire() as connection:
                yield connection
                metrics.record_db_success()
                
        except Exception as e:
            metrics.record_db_error()
            logging.error(f"Database connection error: {e}")
            raise
        
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            metrics.record_db_duration(duration)
    
    @classmethod
    def get_redis(cls) -> redis.Redis:
        """Get Redis client"""
        if not cls._redis_client:
            raise RuntimeError("Redis not initialized")
        return cls._redis_client
    
    @classmethod
    @asynccontextmanager
    async def get_session(cls) -> AsyncGenerator[AsyncSession, None]:
        """Get SQLAlchemy session"""
        if not cls._sqlalchemy_session:
            raise RuntimeError("SQLAlchemy not initialized")
        
        async with cls._sqlalchemy_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
    
    @classmethod
    async def execute_query(cls, query: str, *args) -> Any:
        """Execute query with monitoring"""
        start_time = datetime.now()
        metrics.record_db_query()
        
        try:
            async with cls.get_connection() as conn:
                result = await conn.execute(query, *args)
                metrics.record_db_success()
                return result
                
        except Exception as e:
            metrics.record_db_error()
            logging.error(f"Query execution failed: {e}")
            raise
        
        finally:
            duration = (datetime.now() - start_time).total_seconds()
            metrics.record_db_duration(duration)
    
    @classmethod
    async def fetch_row(cls, query: str, *args) -> Optional[Dict]:
        """Fetch single row with monitoring"""
        async with cls.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    @classmethod
    async def fetch_all(cls, query: str, *args) -> list:
        """Fetch all rows with monitoring"""
        async with cls.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    @classmethod
    async def health_check(cls) -> Dict[str, Any]:
        """Comprehensive database health check"""
        health_status = {
            "postgresql": "unknown",
            "redis": "unknown",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Check PostgreSQL
            async with cls.get_connection() as conn:
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    health_status["postgresql"] = "healthy"
                else:
                    health_status["postgresql"] = "unhealthy"
        
        except Exception as e:
            health_status["postgresql"] = f"unhealthy: {str(e)}"
        
        try:
            # Check Redis
            await cls._redis_client.ping()
            health_status["redis"] = "healthy"
        
        except Exception as e:
            health_status["redis"] = f"unhealthy: {str(e)}"
        
        return health_status
    
    @classmethod
    async def get_database_stats(cls) -> Dict[str, Any]:
        """Get database performance statistics"""
        stats = {}
        
        try:
            async with cls.get_connection() as conn:
                # Database size
                db_size = await conn.fetchval("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """)
                
                # Table statistics
                table_stats = await conn.fetch("""
                    SELECT 
                        schemaname,
                        tablename,
                        attname AS column_name,
                        n_distinct,
                    FROM pg_stats
                    WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
                    LIMIT 50
                """)
                
                # Connection count
                connections = await conn.fetchval("""
                    SELECT COUNT(*) FROM pg_stat_activity 
                    WHERE datname = current_database()
                """)
                
                stats.update({
                    "database_size": db_size,
                    "active_connections": connections,
                    "table_statistics": [dict(row) for row in table_stats]
                })
        
        except Exception as e:
            logging.error(f"Failed to get database stats: {e}")
        
        return stats

# FastAPI dependency
async def get_db() -> AsyncGenerator[Connection, None]:
    """Database dependency for FastAPI"""
    async with DatabaseManager.get_connection() as connection:
        yield connection

# Redis client instance
redis_client = DatabaseManager.get_redis

# SQLAlchemy session dependency
async def get_sqlalchemy_session() -> AsyncGenerator[AsyncSession, None]:
    """SQLAlchemy session dependency for FastAPI"""
    async with DatabaseManager.get_session() as session:
        yield session

# Cache management
class CacheManager:
    """Advanced cache management with Redis"""
    
    def __init__(self):
        self.redis = DatabaseManager.get_redis()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        try:
            return await self.redis.get(key)
        except Exception as e:
            logging.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: str, expire: int = 3600) -> bool:
        """Set value in cache with expiration"""
        try:
            await self.redis.setex(key, expire, value)
            return True
        except Exception as e:
            logging.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logging.error(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return await self.redis.exists(key) > 0
        except Exception as e:
            logging.error(f"Cache exists error: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment counter in cache"""
        try:
            return await self.redis.incrby(key, amount)
        except Exception as e:
            logging.error(f"Cache increment error: {e}")
            return None
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics"""
        try:
            info = await self.redis.info()
            return {
                "used_memory": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
            }
        except Exception as e:
            logging.error(f"Failed to get cache stats: {e}")
            return {}

# Global cache instance
cache_manager = CacheManager()