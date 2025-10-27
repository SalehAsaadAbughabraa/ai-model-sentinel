"""
AI Model Sentinel v2.0.0 - Main Application Entry Point
Production-Grade Enterprise Security Fusion Engine
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

import sentry_sdk
from elasticapm import Client
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.security import get_current_user, rate_limiter
from app.database import DatabaseManager, get_db
from app.monitoring.metrics import MetricsCollector
from app.api.endpoints import router as api_router
from app import config

# Initialize monitoring
metrics = MetricsCollector()

# Initialize Sentry for error tracking
if os.getenv("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        environment=os.getenv("ENVIRONMENT", "production"),
        release=f"ai-sentinel@{config.__version__}",
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )

# Initialize Elastic APM
if os.getenv("ELASTIC_APM_SERVER_URL"):
    apm_client = Client({
        'SERVICE_NAME': 'ai-model-sentinel',
        'SERVER_URL': os.getenv("ELASTIC_APM_SERVER_URL"),
        'ENVIRONMENT': os.getenv("ENVIRONMENT", "production"),
        'CAPTURE_BODY': 'all'
    })

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    startup_time = time.time()
    
    # Startup
    logging.info("🚀 Starting AI Model Sentinel v2.0.0 - Enterprise Edition")
    
    # Initialize database connections
    await DatabaseManager.initialize()
    
    # Initialize metrics
    metrics.initialize()
    
    # Log startup completion
    startup_duration = time.time() - startup_time
    logging.info(f"✅ Startup completed in {startup_duration:.2f}s")
    metrics.record_startup_time(startup_duration)
    
    yield
    
    # Shutdown
    logging.info("🛑 Shutting down AI Model Sentinel")
    await DatabaseManager.close()

# Create FastAPI application
app = FastAPI(
    title="AI Model Sentinel v2.0.0",
    description="World's Most Advanced AI Security Fusion Engine - Enterprise Grade",
    version=config.__version__,
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    lifespan=lifespan,
    openapi_url="/api/v1/openapi.json"
)

# Security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware for monitoring
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Monitor all incoming requests"""
    start_time = time.time()
    metrics.record_request()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Record metrics
        metrics.record_response_time(process_time)
        metrics.record_status_code(response.status_code)
        
        # Add headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Sentinel-Version"] = config.__version__
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        metrics.record_error()
        logging.error(f"Request failed: {str(e)}")
        raise

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Custom docs endpoint with authentication
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(request: Request):
    return get_swagger_ui_html(
        openapi_url="/api/v1/openapi.json",
        title="AI Model Sentinel v2.0.0 - API Documentation",
        swagger_ui_parameters={"defaultModelsExpandDepth": -1}
    )

# Health check endpoint
@app.get("/health")
async def health_check(db: Any = Depends(get_db)):
    """Comprehensive health check"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": config.__version__,
        "environment": os.getenv("ENVIRONMENT", "production"),
        "components": {}
    }
    
    # Check database
    try:
        await db.execute("SELECT 1")
        health_status["components"]["database"] = "healthy"
    except Exception as e:
        health_status["components"]["database"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        from app.database import redis_client
        await redis_client.ping()
        health_status["components"]["redis"] = "healthy"
    except Exception as e:
        health_status["components"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    # Add metrics
    health_status["metrics"] = {
        "uptime": metrics.get_uptime(),
        "total_requests": metrics.get_total_requests(),
        "error_rate": metrics.get_error_rate()
    }
    
    return health_status

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return generate_latest()

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "AI Model Sentinel v2.0.0 - Enterprise Edition",
        "version": config.__version__,
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
        "documentation": "/docs"
    }

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logging.error(f"Global exception: {str(exc)}", exc_info=True)
    metrics.record_error()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request.headers.get("X-Request-ID", "unknown")
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=int(os.getenv("WORKERS", "4")),
        log_level="info",
        access_log=True
    )