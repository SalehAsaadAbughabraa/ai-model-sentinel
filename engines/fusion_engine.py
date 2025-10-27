"""
AI Model Sentinel v2.0.0 - Fusion Intelligence Engine
Advanced fusion of multiple detection engines for ultimate accuracy
Developer: Saleh Asaad Abughabra
Email: saleh87alally@gmail.com
"""

import asyncio
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import numpy as np
from dataclasses import dataclass
import secrets
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import aiofiles
from contextlib import asynccontextmanager
import requests
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import uvicorn
import unittest
import tempfile
import platform
import psutil
import asyncpg
from asyncpg.pool import Pool
import redis.asyncio as redis
import docker
from docker.models.containers import Container

# =============================================================================
# تعريف FastAPI app أولاً لتجنب الأخطاء
# =============================================================================

app = FastAPI(
    title="AI Model Sentinel Fusion Engine API",
    description="Enhanced Fusion Intelligence Engine for AI Model Security",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# =============================================================================
# Enhanced Fusion Engine Core Components
# =============================================================================

class FusionStrategy(Enum):
    """Fusion strategies for combining engine results"""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    MAXIMUM_CONFIDENCE = "maximum_confidence"
    BAYESIAN_FUSION = "bayesian_fusion"
    ADAPTIVE_FUSION = "adaptive_fusion"

class EngineWeight(Enum):
    """Weight assignments for different engines based on reliability"""
    QUANTUM_ENGINE = 0.35
    ML_ENGINE = 0.30
    BEHAVIORAL_ENGINE = 0.20
    SIGNATURE_ENGINE = 0.15

@dataclass
class EngineResult:
    """Standardized result from individual detection engine"""
    engine_name: str
    threat_score: float
    confidence: float
    threat_level: str
    details: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class FusionDecision:
    """Final decision after fusion analysis"""
    final_threat_score: float
    final_threat_level: str
    confidence: float
    consensus_level: str
    contributing_engines: List[str]
    fusion_strategy: str
    details: Dict[str, Any]

class SecurityError(Exception):
    """Security-related exception"""
    pass

class SystemSpecs:
    """System specifications detector"""
    
    @staticmethod
    def detect() -> Dict[str, Any]:
        """Detect system specifications"""
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_cores": os.cpu_count(),
            "storage_gb": psutil.disk_usage('/').total / (1024**3),
            "is_ssd": SystemSpecs._is_ssd()
        }
    
    @staticmethod
    def _is_ssd() -> bool:
        """Check if storage is SSD"""
        try:
            if platform.system() == "Linux":
                return any('ssd' in line.lower() 
                          for line in open('/sys/block/sda/queue/rotational'))
            return True  # Assume SSD for other platforms
        except:
            return True

class UniversalConfigManager:
    """Universal configuration manager for all platforms"""
    
    @staticmethod
    def get_optimized_config(system_specs: Dict) -> Dict[str, Any]:
        """Get optimized configuration based on system specs"""
        ram_gb = system_specs['ram_gb']
        cpu_cores = system_specs['cpu_cores']
        is_ssd = system_specs['is_ssd']
        
        # Database configuration
        db_config = {
            "postgresql": {
                "shared_buffers": f"{max(1, int(ram_gb * 0.25))}GB",
                "work_mem": f"{max(64, int(ram_gb * 8))}MB",
                "maintenance_work_mem": f"{max(512, int(ram_gb * 64))}MB",
                "max_connections": min(1000, max(100, cpu_cores * 50)),
                "effective_cache_size": f"{int(ram_gb * 0.6)}GB",
                "random_page_cost": "1.1" if is_ssd else "4.0",
                "checkpoint_completion_target": "0.9",
                "wal_buffers": f"{min(16, max(1, int(ram_gb * 0.03)))}MB"
            },
            "performance": {
                "max_scans_per_hour": UniversalConfigManager._calculate_capacity(ram_gb, cpu_cores),
                "recommended_scans_day": UniversalConfigManager._calculate_daily_capacity(ram_gb, cpu_cores)
            },
            "partitioning": UniversalConfigManager._get_partitioning_strategy(ram_gb)
        }
        
        return db_config
    
    @staticmethod
    def _calculate_capacity(ram_gb: float, cpu_cores: int) -> str:
        """Calculate hourly scan capacity"""
        base_capacity = ram_gb * 1000 + cpu_cores * 500
        if base_capacity <= 10000:
            return "10,000"
        elif base_capacity <= 50000:
            return "50,000"
        elif base_capacity <= 200000:
            return "200,000"
        else:
            return "500,000+"
    
    @staticmethod
    def _calculate_daily_capacity(ram_gb: float, cpu_cores: int) -> str:
        """Calculate daily scan capacity"""
        hourly = UniversalConfigManager._calculate_capacity(ram_gb, cpu_cores)
        hourly_num = int(hourly.replace(',', '').replace('+', ''))
        return f"{hourly_num * 24:,}"
    
    @staticmethod
    def _get_partitioning_strategy(ram_gb: float) -> Dict[str, str]:
        """Get partitioning strategy based on RAM"""
        if ram_gb <= 8:
            return {"fusion_history": "BY MONTH", "scan_logs": "BY WEEK"}
        elif ram_gb <= 32:
            return {"fusion_history": "BY WEEK", "scan_logs": "BY DAY"}
        else:
            return {"fusion_history": "BY DAY", "scan_logs": "BY HOUR"}

class PostgreSQLManager:
    """PostgreSQL database manager with connection pooling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[Pool] = None
        self.logger = logging.getLogger("PostgreSQLManager")
    
    async def initialize(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                database=self.config.get('database', 'ai_sentinel'),
                user=self.config.get('username', 'admin'),
                password=self.config.get('password', 'password'),
                min_size=10,
                max_size=50
            )
            await self._create_tables()
            self.logger.info("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize PostgreSQL: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create necessary tables"""
        async with self.pool.acquire() as conn:
            # Fusion history table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS fusion_history (
                    id SERIAL PRIMARY KEY,
                    fusion_id TEXT UNIQUE NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    final_threat_score DECIMAL(5,4) CHECK (final_threat_score >= 0 AND final_threat_score <= 1),
                    final_threat_level TEXT NOT NULL,
                    confidence DECIMAL(5,4) CHECK (confidence >= 0 AND confidence <= 1),
                    consensus_level TEXT NOT NULL,
                    fusion_strategy TEXT NOT NULL,
                    contributing_engines JSONB NOT NULL,
                    details JSONB NOT NULL,
                    ground_truth TEXT,
                    performance_metrics JSONB,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Engine weights table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS engine_weights (
                    engine_name TEXT PRIMARY KEY,
                    weight DECIMAL(5,4) CHECK (weight >= 0 AND weight <= 1),
                    accuracy DECIMAL(5,4) CHECK (accuracy >= 0 AND accuracy <= 1),
                    last_updated TIMESTAMPTZ NOT NULL,
                    is_active BOOLEAN DEFAULT true
                )
            ''')
            
            # API audit log table
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS api_audit_log (
                    id SERIAL PRIMARY KEY,
                    api_key_hash TEXT,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status_code INTEGER,
                    user_agent TEXT,
                    ip_address TEXT,
                    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    processing_time_ms INTEGER
                )
            ''')
            
            # Create indexes
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_fusion_history_timestamp 
                ON fusion_history(timestamp)
            ''')
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_fusion_history_threat_level 
                ON fusion_history(final_threat_level)
            ''')
    
    async def store_fusion_record(self, decision: FusionDecision, original_results: List[EngineResult], 
                                ground_truth: str = None):
        """Store fusion record in PostgreSQL"""
        async with self.pool.acquire() as conn:
            await conn.execute('''
                INSERT INTO fusion_history 
                (fusion_id, timestamp, final_threat_score, final_threat_level, 
                 confidence, consensus_level, fusion_strategy, contributing_engines, 
                 details, ground_truth)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ''', (
                decision.details.get('fusion_id', 'unknown'),
                datetime.now(),
                decision.final_threat_score,
                decision.final_threat_level,
                decision.confidence,
                decision.consensus_level,
                decision.fusion_strategy,
                json.dumps(decision.contributing_engines),
                json.dumps(self._sanitize_details(decision.details)),
                ground_truth
            ))
    
    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize details to remove sensitive information"""
        sanitized = details.copy()
        sensitive_keys = ['api_key', 'password', 'token', 'secret', 'key']
        for key in list(sanitized.keys()):
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = '***REDACTED***'
        return sanitized
    
    async def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get performance statistics from database"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow('''
                SELECT 
                    COUNT(*) as total_records,
                    AVG(CASE WHEN ground_truth = final_threat_level THEN 1.0 ELSE 0.0 END) as accuracy,
                    AVG(final_threat_score) as avg_threat_score,
                    AVG(confidence) as avg_confidence
                FROM fusion_history 
                WHERE ground_truth IS NOT NULL 
                AND timestamp >= NOW() - $1 * INTERVAL '1 day'
            ''', days)
            
            return {
                "total_records": row['total_records'] if row else 0,
                "accuracy": float(row['accuracy']) if row and row['accuracy'] is not None else 0.0,
                "avg_threat_score": float(row['avg_threat_score']) if row and row['avg_threat_score'] is not None else 0.0,
                "avg_confidence": float(row['avg_confidence']) if row and row['avg_confidence'] is not None else 0.0
            }

class DockerSandboxManager:
    """Docker sandbox manager for safe file analysis"""
    
    def __init__(self):
        try:
            self.client = docker.from_env()
            self.logger = logging.getLogger("DockerSandboxManager")
        except Exception as e:
            self.logger = logging.getLogger("DockerSandboxManager")
            self.client = None
            self.logger.warning(f"Docker not available: {str(e)}")
    
    async def create_sandbox(self, image: str = "ubuntu:20.04") -> Optional[Container]:
        """Create a Docker sandbox container"""
        if not self.client:
            return None
            
        try:
            container = self.client.containers.create(
                image=image,
                command=["sleep", "infinity"],
                network_disabled=True,
                mem_limit="512m",
                cpu_period=100000,
                cpu_quota=50000,  # 50% CPU
                read_only=True,
                security_opt=["no-new-privileges:true"],
                volumes={
                    '/tmp': {'bind': '/tmp', 'mode': 'ro'}
                }
            )
            container.start()
            self.logger.info(f"Sandbox container created: {container.id}")
            return container
        except Exception as e:
            self.logger.error(f"Failed to create sandbox: {str(e)}")
            return None
    
    async def analyze_file_safely(self, file_path: str, container: Container) -> Dict[str, Any]:
        """Analyze file safely within sandbox"""
        if not container:
            return {"error": "No container available"}
            
        try:
            # Copy file to sandbox
            with open(file_path, 'rb') as f:
                container.put_archive('/tmp', f.read())
            
            # Run analysis commands
            analysis_commands = [
                "file /tmp/*",
                "stat /tmp/*",
                "strings /tmp/* | head -100"
            ]
            
            results = {}
            for cmd in analysis_commands:
                exit_code, output = container.exec_run(cmd)
                results[cmd] = {
                    "exit_code": exit_code,
                    "output": output.decode('utf-8', errors='ignore')[:1000]
                }
            
            return results
        except Exception as e:
            self.logger.error(f"Sandbox analysis failed: {str(e)}")
            return {"error": str(e)}
    
    async def cleanup_sandbox(self, container: Container):
        """Clean up sandbox container"""
        if not container:
            return
            
        try:
            container.stop()
            container.remove()
            self.logger.info(f"Sandbox container cleaned up: {container.id}")
        except Exception as e:
            self.logger.warning(f"Failed to clean up sandbox: {str(e)}")

class ThreatCloudSync:
    """Threat intelligence cloud synchronization"""
    
    def __init__(self, virustotal_api_key: str = None, misp_url: str = None, misp_key: str = None):
        self.virustotal_api_key = virustotal_api_key
        self.misp_url = misp_url
        self.misp_key = misp_key
        self.redis_client = None
        self.logger = logging.getLogger("ThreatCloudSync")
    
    async def initialize_redis(self, host: str = "localhost", port: int = 6379):
        """Initialize Redis client for caching"""
        try:
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            await self.redis_client.ping()
            self.logger.info("Redis client initialized successfully")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {str(e)}")
            self.redis_client = None
    
    async def sync_with_virustotal(self, file_hash: str) -> Dict[str, Any]:
        """Sync threat data with VirusTotal"""
        if not self.virustotal_api_key:
            return {}
        
        # Check cache first
        cache_key = f"virustotal:{file_hash}"
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass
        
        try:
            url = f"https://www.virustotal.com/api/v3/files/{file_hash}"
            headers = {"x-apikey": self.virustotal_api_key}
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.get(url, headers=headers, timeout=10)
            )
            
            if response.status_code == 200:
                result = response.json()
                # Cache result for 1 hour
                if self.redis_client:
                    try:
                        await self.redis_client.setex(cache_key, 3600, json.dumps(result))
                    except Exception:
                        pass
                return result
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"VirusTotal sync failed: {str(e)}")
            return {}
    
    async def sync_with_misp(self, indicators: List[str]) -> Dict[str, Any]:
        """Sync threat data with MISP"""
        if not self.misp_url or not self.misp_key:
            return {}
        
        try:
            headers = {
                "Authorization": self.misp_key,
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            payload = {
                "returnFormat": "json",
                "type": [{"OR": indicators}]
            }
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.post(
                    f"{self.misp_url}/events/restSearch",
                    headers=headers,
                    json=payload,
                    timeout=15
                )
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"MISP sync failed: {str(e)}")
            return {}

class AlertManager:
    """Alert manager for notifications"""
    
    def __init__(self):
        self.logger = logging.getLogger("AlertManager")
    
    async def send_alert(self, alert_type: str, message: str, severity: str = "medium", 
                        metadata: Dict[str, Any] = None):
        """Send alert through configured channels"""
        alert_data = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Log all alerts
        self.logger.warning(f"ALERT: {alert_type} - {message}")
        
        # Here you can integrate with:
        # - Email (SMTP)
        # - Slack webhooks
        # - Discord webhooks
        # - Custom webhooks
        # - SMS services
        
        return alert_data
    
    async def send_critical_alert(self, decision: FusionDecision, file_path: str):
        """Send critical alert for high-threat detections"""
        if decision.final_threat_level in ["QUANTUM_HIGH", "QUANTUM_CRITICAL"]:
            message = f"CRITICAL THREAT DETECTED: {decision.final_threat_level} in {file_path}"
            return await self.send_alert(
                "critical_threat",
                message,
                "critical",
                {
                    "file_path": file_path,
                    "threat_score": decision.final_threat_score,
                    "threat_level": decision.final_threat_level,
                    "fusion_id": decision.details.get('fusion_id')
                }
            )

class OptimizedBayesianCalculator:
    """Optimized Bayesian calculations with log-space to prevent underflow"""
    
    @staticmethod
    def calculate_posterior(prior_malicious: float, engine_results: List[EngineResult], 
                          engine_weights: Dict[str, float]) -> float:
        """
        Calculate posterior probability using log-space to prevent underflow
        """
        prior_clean = 1.0 - prior_malicious
        
        # Use logarithms to prevent numerical underflow
        log_likelihood_malicious = 0.0
        log_likelihood_clean = 0.0
        
        for result in engine_results:
            weight = engine_weights.get(result.engine_name, 0.1)
            score = max(0.01, min(0.99, result.threat_score))  # Avoid extremes
            
            # Log likelihoods with weight consideration
            log_likelihood_malicious += np.log(score * weight + 1e-10)
            log_likelihood_clean += np.log((1 - score) * weight + 1e-10)
        
        # Add prior log probabilities
        log_posterior_malicious = np.log(prior_malicious) + log_likelihood_malicious
        log_posterior_clean = np.log(prior_clean) + log_likelihood_clean
        
        # Use log-sum-exp trick for numerical stability
        max_log = max(log_posterior_malicious, log_posterior_clean)
        log_sum_exp = max_log + np.log(np.exp(log_posterior_malicious - max_log) + 
                                     np.exp(log_posterior_clean - max_log))
        
        posterior_malicious = np.exp(log_posterior_malicious - log_sum_exp)
        
        return min(posterior_malicious, 1.0)

class SelfEvaluationModule:
    """Self-evaluation module for automatic weight adjustment"""
    
    def __init__(self, db_manager: PostgreSQLManager):
        self.db_manager = db_manager
        self.learning_rate = 0.1
        self.min_weight = 0.05  # Minimum weight to prevent zero influence
    
    async def evaluate_and_adjust_weights(self, ground_truth_data: List[Dict]) -> Dict[str, float]:
        """Evaluate performance and adjust engine weights automatically"""
        correct_predictions = {engine: 0 for engine in ["quantum_engine", "ml_engine", "behavioral_engine", "signature_engine"]}
        total_predictions = {engine: 0 for engine in correct_predictions.keys()}
        
        for record in ground_truth_data:
            actual_threat = record.get('actual_threat')
            engine_predictions = record.get('engine_predictions', {})
            
            for engine_name, prediction in engine_predictions.items():
                if engine_name in correct_predictions:
                    total_predictions[engine_name] += 1
                    if self._is_correct_prediction(prediction, actual_threat):
                        correct_predictions[engine_name] += 1
        
        # Calculate new weights based on accuracy with smoothing
        new_weights = {}
        total_accuracy = 0
        
        for engine in correct_predictions:
            accuracy = correct_predictions[engine] / total_predictions[engine] if total_predictions[engine] > 0 else 0.5
            # Apply smoothing to avoid extreme values
            smoothed_accuracy = 0.7 * accuracy + 0.3 * 0.5  # Bias toward neutral
            total_accuracy += smoothed_accuracy
            new_weights[engine] = max(self.min_weight, smoothed_accuracy)
        
        # Normalize weights to sum to 1
        if total_accuracy > 0:
            weight_sum = sum(new_weights.values())
            for engine in new_weights:
                new_weights[engine] = new_weights[engine] / weight_sum
        
        return new_weights
    
    def _is_correct_prediction(self, prediction: str, actual: str) -> bool:
        """Check if prediction matches actual threat level with tolerance"""
        threat_mapping = {
            "QUANTUM_CLEAN": "clean", "CLEAN": "clean",
            "QUANTUM_LOW": "low", "LOW": "low", 
            "QUANTUM_MEDIUM": "medium", "MEDIUM": "medium",
            "QUANTUM_HIGH": "high", "HIGH": "high",
            "QUANTUM_CRITICAL": "critical", "CRITICAL": "critical"
        }
        
        pred_category = threat_mapping.get(prediction, "unknown")
        actual_category = threat_mapping.get(actual, "unknown")
        
        return pred_category == actual_category

class EnhancedFusionIntelligenceEngine:
    """
    Enhanced fusion engine with universal compatibility
    """
    
    def __init__(self, db_config: Dict[str, Any] = None):
        # Auto-detect system specs and configure
        system_specs = SystemSpecs.detect()
        self.optimal_config = UniversalConfigManager.get_optimized_config(system_specs)
        
        self.engine_weights = {
            "quantum_engine": EngineWeight.QUANTUM_ENGINE.value,
            "ml_engine": EngineWeight.ML_ENGINE.value,
            "behavioral_engine": EngineWeight.BEHAVIORAL_ENGINE.value,
            "signature_engine": EngineWeight.SIGNATURE_ENGINE.value
        }
        
        self.threat_level_mapping = {
            "QUANTUM_CLEAN": 0.0,
            "QUANTUM_LOW": 0.2,
            "QUANTUM_MEDIUM": 0.4,
            "QUANTUM_HIGH": 0.6,
            "QUANTUM_CRITICAL": 0.8,
            "CLEAN": 0.0,
            "LOW": 0.2,
            "MEDIUM": 0.4,
            "HIGH": 0.6,
            "CRITICAL": 0.8
        }
        
        self.consensus_thresholds = {
            "HIGH_CONSENSUS": 0.8,
            "MEDIUM_CONSENSUS": 0.6,
            "LOW_CONSENSUS": 0.4,
            "CONFLICT": 0.2
        }
        
        # Initialize components
        self.db_manager = PostgreSQLManager(db_config or {})
        self.sandbox_manager = DockerSandboxManager()
        self.threat_cloud = ThreatCloudSync()
        self.alert_manager = AlertManager()
        self.self_evaluation = SelfEvaluationModule(self.db_manager)
        self.bayesian_calculator = OptimizedBayesianCalculator()
        
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),
            thread_name_prefix="fusion_worker"
        )
        
        self.logger = self._setup_logger()
        self.parallel_executions = 0
        self.fusion_history = []
        
    def _setup_logger(self):
        """Setup fusion engine logger"""
        logger = logging.getLogger("EnhancedFusionIntelligenceEngine")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        # Prevent log injection
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            # Sanitize log messages
            if hasattr(record, 'msg'):
                record.msg = str(record.msg).replace('\n', ' ').replace('\r', ' ')
            return record
        
        logging.setLogRecordFactory(record_factory)
        return logger

    async def initialize(self):
        """Initialize all components"""
        await self.db_manager.initialize()
        await self.threat_cloud.initialize_redis()
        self.logger.info("Enhanced Fusion Engine initialized successfully")
        self.logger.info(f"Optimal configuration: {self.optimal_config['performance']}")

    async def universal_scan_analysis(self, file_path: str, use_sandbox: bool = False,
                                    sync_threat_intel: bool = False) -> Dict[str, Any]:
        """
        Universal scan analysis with all advanced features
        """
        fusion_id = secrets.token_hex(8)
        self.logger.info(f"Starting universal scan analysis [ID: {fusion_id}]")
        
        start_time = time.time()
        
        try:
            # Security validation
            if not await self._validate_file_path(file_path):
                raise SecurityError(f"Invalid file path: {file_path}")
            
            # Parallel engine processing
            engine_results = await self.parallel_engine_processing(
                file_path, list(self.engine_weights.keys())
            )
            
            if not engine_results:
                raise SecurityError("No engine results available")
            
            # Sandbox analysis (optional)
            sandbox_results = {}
            if use_sandbox:
                sandbox_results = await self._run_sandbox_analysis(file_path)
            
            # Threat intelligence sync (optional)
            threat_intel = {}
            if sync_threat_intel:
                file_hash = await self._calculate_file_hash(file_path)
                threat_intel = {
                    "virustotal": await self.threat_cloud.sync_with_virustotal(file_hash),
                    "misp": await self.threat_cloud.sync_with_misp([file_hash])
                }
            
            # Fusion analysis
            decision = await self.fuse_engine_results(engine_results, FusionStrategy.ADAPTIVE_FUSION)
            
            # Store results
            await self.db_manager.store_fusion_record(decision, engine_results)
            
            # Send alerts for critical threats
            await self.alert_manager.send_critical_alert(decision, file_path)
            
            processing_time = time.time() - start_time
            
            # Comprehensive report
            report = {
                "scan_id": fusion_id,
                "file_path": file_path,
                "processing_time": processing_time,
                "final_decision": {
                    "threat_level": decision.final_threat_level,
                    "threat_score": decision.final_threat_score,
                    "confidence": decision.confidence,
                    "consensus": decision.consensus_level
                },
                "engine_results": [
                    {
                        "engine": result.engine_name,
                        "threat_score": result.threat_score,
                        "threat_level": result.threat_level,
                        "confidence": result.confidence
                    } for result in engine_results
                ],
                "sandbox_analysis": sandbox_results,
                "threat_intelligence": threat_intel,
                "system_info": {
                    "optimal_config": self.optimal_config['performance'],
                    "total_engines_used": len(engine_results)
                }
            }
            
            self.logger.info(f"Universal scan completed [ID: {fusion_id}] - Threat: {decision.final_threat_level}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Universal scan failed [ID: {fusion_id}]: {str(e)}")
            await self.alert_manager.send_alert(
                "scan_failed",
                f"Scan failed for {file_path}: {str(e)}",
                "high",
                {"file_path": file_path, "error": str(e)}
            )
            raise

    async def _run_sandbox_analysis(self, file_path: str) -> Dict[str, Any]:
        """Run sandbox analysis for file"""
        container = None
        try:
            container = await self.sandbox_manager.create_sandbox()
            if container:
                results = await self.sandbox_manager.analyze_file_safely(file_path, container)
                return results
            else:
                return {"error": "Docker sandbox not available"}
        except Exception as e:
            self.logger.warning(f"Sandbox analysis failed: {str(e)}")
            return {"error": str(e)}
        finally:
            if container:
                await self.sandbox_manager.cleanup_sandbox(container)

    async def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for threat intelligence"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def fuse_engine_results(self, engine_results: List[EngineResult], 
                                strategy: FusionStrategy = FusionStrategy.ADAPTIVE_FUSION,
                                ground_truth: str = None) -> FusionDecision:
        """
        Intelligently fuse results from multiple detection engines with security fixes
        """
        fusion_id = secrets.token_hex(8)
        self.logger.info(f"Starting secure fusion analysis [ID: {fusion_id}]")
        
        try:
            # Validate and sanitize input
            if not engine_results:
                return self._create_empty_decision(fusion_id)
            
            # Validate all engine results
            validated_results = await self._validate_engine_results(engine_results)
            
            if not validated_results:
                return self._create_error_decision("No valid engine results", fusion_id)
            
            # Normalize all threat scores to ensure consistent scale
            normalized_results = await self._normalize_engine_results(validated_results)
            
            # Apply selected fusion strategy
            if strategy == FusionStrategy.WEIGHTED_AVERAGE:
                decision = await self._weighted_average_fusion(normalized_results, fusion_id)
            elif strategy == FusionStrategy.MAJORITY_VOTE:
                decision = await self._majority_vote_fusion(normalized_results, fusion_id)
            elif strategy == FusionStrategy.MAXIMUM_CONFIDENCE:
                decision = await self._maximum_confidence_fusion(normalized_results, fusion_id)
            elif strategy == FusionStrategy.BAYESIAN_FUSION:
                decision = await self._optimized_bayesian_fusion(normalized_results, fusion_id)
            else:  # ADAPTIVE_FUSION
                decision = await self._adaptive_fusion(normalized_results, fusion_id)
            
            # Store fusion history in database
            await self.db_manager.store_fusion_record(decision, engine_results, ground_truth)
            
            self.logger.info(f"Secure fusion analysis completed [ID: {fusion_id}] - Final Threat: {decision.final_threat_level}")
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Secure fusion analysis failed [ID: {fusion_id}]: {str(e)}")
            return self._create_error_decision(str(e), fusion_id)

    async def _validate_engine_results(self, results: List[EngineResult]) -> List[EngineResult]:
        """Validate and sanitize engine results"""
        validated_results = []
        
        for result in results:
            # Validate threat score range
            if not (0 <= result.threat_score <= 1):
                self.logger.warning(f"Invalid threat score from {result.engine_name}: {result.threat_score}")
                continue
            
            # Validate confidence range
            if not (0 <= result.confidence <= 1):
                self.logger.warning(f"Invalid confidence from {result.engine_name}: {result.confidence}")
                continue
            
            # Sanitize details to prevent injection
            sanitized_details = [
                str(detail).replace('\n', ' ').replace('\r', ' ')[:1000]  # Limit length
                for detail in result.details
            ]
            
            validated_result = EngineResult(
                engine_name=result.engine_name,
                threat_score=result.threat_score,
                confidence=result.confidence,
                threat_level=result.threat_level,
                details=sanitized_details,
                metadata=result.metadata,
                timestamp=result.timestamp
            )
            validated_results.append(validated_result)
        
        return validated_results

    async def _normalize_engine_results(self, results: List[EngineResult]) -> List[EngineResult]:
        """Normalize engine results to ensure consistent threat score ranges"""
        normalized_results = []
        
        for result in results:
            # Ensure threat score is in 0-1 range using safe normalization
            normalized_score = max(0.0, min(1.0, result.threat_score))
            
            # Create normalized result
            normalized_result = EngineResult(
                engine_name=result.engine_name,
                threat_score=normalized_score,
                confidence=result.confidence,
                threat_level=result.threat_level,
                details=result.details,
                metadata=result.metadata,
                timestamp=result.timestamp
            )
            normalized_results.append(normalized_result)
        
        return normalized_results

    async def _optimized_bayesian_fusion(self, results: List[EngineResult], fusion_id: str) -> FusionDecision:
        """Optimized Bayesian fusion using log-space calculations - FIXED VERSION"""
        self.logger.info(f"Applying optimized Bayesian fusion [ID: {fusion_id}]")
        
        # Prior probability (assuming 10% of files are malicious)
        prior_malicious = 0.1
        
        # FIXED: Use the corrected Bayesian calculator
        final_score = self.bayesian_calculator.calculate_posterior(
            prior_malicious, results, self.engine_weights
        )
        
        return FusionDecision(
            final_threat_score=final_score,
            final_threat_level=self._score_to_threat_level(final_score),
            confidence=final_score,
            consensus_level=self._calculate_bayesian_consensus(results),
            contributing_engines=[r.engine_name for r in results],
            fusion_strategy="optimized_bayesian_fusion",
            details={
                "posterior_probability": final_score,
                "prior_malicious": prior_malicious,
                "log_space_calculation": True,  # Indicate fixed method
                "fusion_id": fusion_id
            }
        )

    async def _weighted_average_fusion(self, results: List[EngineResult], fusion_id: str) -> FusionDecision:
        """Weighted average fusion with security improvements"""
        self.logger.info(f"Applying weighted average fusion [ID: {fusion_id}]")
        
        total_weight = 0.0
        weighted_score = 0.0
        confidences = []
        
        for result in results:
            weight = self.engine_weights.get(result.engine_name, 0.1)
            # Apply confidence-based weighting
            confidence_weight = weight * result.confidence
            weighted_score += result.threat_score * confidence_weight
            total_weight += confidence_weight
            confidences.append(result.confidence)
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            # Fallback to simple average
            final_score = sum(r.threat_score for r in results) / len(results)
        
        avg_confidence = np.mean(confidences) if confidences else 0.5
        consensus = await self._calculate_consensus(results, final_score)
        
        return FusionDecision(
            final_threat_score=final_score,
            final_threat_level=self._score_to_threat_level(final_score),
            confidence=avg_confidence,
            consensus_level=consensus,
            contributing_engines=[r.engine_name for r in results],
            fusion_strategy="weighted_average",
            details={
                "weighted_scores": {r.engine_name: r.threat_score for r in results},
                "weights_applied": {r.engine_name: self.engine_weights.get(r.engine_name, 0.1) for r in results},
                "total_weight": total_weight,
                "fusion_id": fusion_id
            }
        )

    async def _majority_vote_fusion(self, results: List[EngineResult], fusion_id: str) -> FusionDecision:
        """Majority vote fusion based on threat levels"""
        self.logger.info(f"Applying majority vote fusion [ID: {fusion_id}]")
        
        # Count votes for each threat level
        threat_votes = {}
        for result in results:
            level = result.threat_level
            threat_votes[level] = threat_votes.get(level, 0) + 1
        
        # Find majority threat level
        if threat_votes:
            majority_level = max(threat_votes.items(), key=lambda x: x[1])[0]
            vote_ratio = threat_votes[majority_level] / len(results)
        else:
            majority_level = "QUANTUM_CLEAN"
            vote_ratio = 0.0
        
        # Convert threat level to score
        final_score = self.threat_level_mapping.get(majority_level, 0.5)
        
        return FusionDecision(
            final_threat_score=final_score,
            final_threat_level=majority_level,
            confidence=vote_ratio,
            consensus_level=self._vote_ratio_to_consensus(vote_ratio),
            contributing_engines=[r.engine_name for r in results],
            fusion_strategy="majority_vote",
            details={
                "vote_distribution": threat_votes,
                "majority_ratio": vote_ratio,
                "fusion_id": fusion_id
            }
        )

    async def _maximum_confidence_fusion(self, results: List[EngineResult], fusion_id: str) -> FusionDecision:
        """Fusion based on the most confident engine"""
        self.logger.info(f"Applying maximum confidence fusion [ID: {fusion_id}]")
        
        if not results:
            return self._create_empty_decision(fusion_id)
        
        # Find engine with highest confidence
        best_engine = max(results, key=lambda x: x.confidence)
        
        return FusionDecision(
            final_threat_score=best_engine.threat_score,
            final_threat_level=best_engine.threat_level,
            confidence=best_engine.confidence,
            consensus_level="HIGH_CONSENSUS",  # Single decision maker
            contributing_engines=[best_engine.engine_name],
            fusion_strategy="maximum_confidence",
            details={
                "selected_engine": best_engine.engine_name,
                "engine_confidence": best_engine.confidence,
                "fusion_id": fusion_id
            }
        )

    async def _adaptive_fusion(self, results: List[EngineResult], fusion_id: str) -> FusionDecision:
        """Adaptive fusion that selects best strategy based on context"""
        self.logger.info(f"Applying adaptive fusion [ID: {fusion_id}]")
        
        # Analyze context to choose best strategy
        strategy_confidence = {}
        
        # Check for high confidence consensus
        high_confidence_engines = [r for r in results if r.confidence > 0.8]
        if len(high_confidence_engines) >= 2:
            strategy_confidence["maximum_confidence"] = 0.9
        
        # Check for agreement levels
        scores = [r.threat_score for r in results]
        avg_score = np.mean(scores) if scores else 0.5
        score_std = np.std(scores) if len(scores) > 1 else 0.0
        
        if score_std < 0.2:  # Low variance - engines agree
            strategy_confidence["weighted_average"] = 0.8
        else:  # High variance - engines disagree
            strategy_confidence["bayesian_fusion"] = 0.7
        
        # Check for clear majority
        threat_levels = [r.threat_level for r in results]
        if threat_levels:
            most_common = max(set(threat_levels), key=threat_levels.count)
            majority_ratio = threat_levels.count(most_common) / len(threat_levels)
            
            if majority_ratio > 0.6:
                strategy_confidence["majority_vote"] = 0.85
        
        # Select best strategy
        if not strategy_confidence:
            best_strategy = FusionStrategy.WEIGHTED_AVERAGE
        else:
            best_strategy_name = max(strategy_confidence.items(), key=lambda x: x[1])[0]
            best_strategy = FusionStrategy[best_strategy_name.upper()]
        
        # Apply selected strategy
        if best_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return await self._weighted_average_fusion(results, fusion_id)
        elif best_strategy == FusionStrategy.MAJORITY_VOTE:
            return await self._majority_vote_fusion(results, fusion_id)
        elif best_strategy == FusionStrategy.MAXIMUM_CONFIDENCE:
            return await self._maximum_confidence_fusion(results, fusion_id)
        else:  # bayesian_fusion
            return await self._optimized_bayesian_fusion(results, fusion_id)

    async def _calculate_consensus(self, results: List[EngineResult], final_score: float) -> str:
        """Calculate consensus level among engines"""
        if not results:
            return "NO_CONSENSUS"
        
        scores = [r.threat_score for r in results]
        avg_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0.0
        
        # Normalize consensus score (0-1)
        consensus_score = 1 - min(std_score / 0.5, 1.0)  # Cap at 1.0
        
        if consensus_score >= self.consensus_thresholds["HIGH_CONSENSUS"]:
            return "HIGH_CONSENSUS"
        elif consensus_score >= self.consensus_thresholds["MEDIUM_CONSENSUS"]:
            return "MEDIUM_CONSENSUS"
        elif consensus_score >= self.consensus_thresholds["LOW_CONSENSUS"]:
            return "LOW_CONSENSUS"
        else:
            return "CONFLICT"

    def _calculate_bayesian_consensus(self, results: List[EngineResult]) -> str:
        """Calculate consensus for Bayesian fusion"""
        scores = [r.threat_score for r in results]
        if not scores:
            return "NO_CONSENSUS"
            
        entropy = -sum(p * np.log(p + 1e-10) for p in scores)  # Add small value to avoid log(0)
        max_entropy = np.log(len(scores)) if scores else 1
        
        consensus_level = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
        
        if consensus_level >= 0.8:
            return "HIGH_CONSENSUS"
        elif consensus_level >= 0.6:
            return "MEDIUM_CONSENSUS"
        elif consensus_level >= 0.4:
            return "LOW_CONSENSUS"
        else:
            return "CONFLICT"

    def _vote_ratio_to_consensus(self, vote_ratio: float) -> str:
        """Convert vote ratio to consensus level"""
        if vote_ratio >= 0.8:
            return "HIGH_CONSENSUS"
        elif vote_ratio >= 0.6:
            return "MEDIUM_CONSENSUS"
        elif vote_ratio >= 0.4:
            return "LOW_CONSENSUS"
        else:
            return "CONFLICT"

    def _score_to_threat_level(self, score: float) -> str:
        """Convert numerical score to threat level"""
        if score >= 0.8:
            return "QUANTUM_CRITICAL"
        elif score >= 0.6:
            return "QUANTUM_HIGH"
        elif score >= 0.4:
            return "QUANTUM_MEDIUM"
        elif score >= 0.2:
            return "QUANTUM_LOW"
        else:
            return "QUANTUM_CLEAN"

    async def parallel_engine_processing(self, file_path: str, engines: List[str]) -> List[EngineResult]:
        """Process multiple engines in parallel with security checks"""
        self.logger.info(f"Starting secure parallel engine processing for: {file_path}")
        
        # Security check: validate file path
        if not await self._validate_file_path(file_path):
            raise SecurityError(f"Invalid or unsafe file path: {file_path}")
        
        # Create tasks for each engine
        tasks = []
        for engine_name in engines:
            if engine_name == "quantum_engine":
                task = self._run_quantum_engine(file_path)
            elif engine_name == "ml_engine":
                task = self._run_ml_engine(file_path)
            elif engine_name == "behavioral_engine":
                task = self._run_behavioral_engine(file_path)
            elif engine_name == "signature_engine":
                task = self._run_signature_engine(file_path)
            else:
                self.logger.warning(f"Unknown engine: {engine_name}")
                continue
            tasks.append(task)
        
        # Execute all engines in parallel with timeout
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self.parallel_executions += 1
        except asyncio.TimeoutError:
            self.logger.error("Parallel engine processing timeout")
            return []
        
        # Filter out exceptions and return valid results
        valid_results = []
        for result in results:
            if isinstance(result, EngineResult):
                valid_results.append(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Engine processing failed: {str(result)}")
        
        return valid_results

    async def _validate_file_path(self, file_path: str) -> bool:
        """Validate file path for security"""
        try:
            # Convert to absolute path and check for path traversal
            abs_path = os.path.abspath(file_path)
            canonical_path = os.path.realpath(abs_path)
            
            # Check for path traversal attempts
            if '..' in file_path or (file_path.startswith('/') and not file_path.startswith('/tmp')):
                return False
            
            # Check if path is within allowed directories
            allowed_dirs = [os.getcwd(), "/tmp", "/var/tmp"]
            if not any(canonical_path.startswith(dir) for dir in allowed_dirs):
                return False
            
            # Check file exists and is readable
            if not os.path.exists(canonical_path) or not os.path.isfile(canonical_path):
                return False
            
            # Check file size limit (100MB)
            if os.path.getsize(canonical_path) > 100 * 1024 * 1024:
                return False
                
            return True
        except Exception:
            return False

    async def _run_quantum_engine(self, file_path: str) -> EngineResult:
        """Simulate quantum engine processing"""
        # This would integrate with your actual quantum engine
        await asyncio.sleep(0.1)  # Simulate processing time
        
        return EngineResult(
            engine_name="quantum_engine",
            threat_score=0.75,
            confidence=0.9,
            threat_level="QUANTUM_HIGH",
            details=["Quantum pattern detection: Suspicious entropy"],
            metadata={"quantum_bits": 512, "processing_time": 0.15},
            timestamp=datetime.now()
        )

    async def _run_ml_engine(self, file_path: str) -> EngineResult:
        """Simulate ML engine processing"""
        await asyncio.sleep(0.08)
        
        return EngineResult(
            engine_name="ml_engine",
            threat_score=0.68,
            confidence=0.85,
            threat_level="QUANTUM_MEDIUM",
            details=["Anomaly detection: Unusual feature weights"],
            metadata={"model": "IsolationForest", "features_analyzed": 150},
            timestamp=datetime.now()
        )

    async def _run_behavioral_engine(self, file_path: str) -> EngineResult:
        """Simulate behavioral engine processing"""
        await asyncio.sleep(0.12)
        
        return EngineResult(
            engine_name="behavioral_engine",
            threat_score=0.55,
            confidence=0.75,
            threat_level="QUANTUM_MEDIUM",
            details=["Behavioral analysis: Suspicious system calls"],
            metadata={"api_calls_monitored": 45, "risk_factors": 2},
            timestamp=datetime.now()
        )

    async def _run_signature_engine(self, file_path: str) -> EngineResult:
        """Simulate signature engine processing"""
        await asyncio.sleep(0.05)
        
        return EngineResult(
            engine_name="signature_engine",
            threat_score=0.82,
            confidence=0.95,
            threat_level="QUANTUM_HIGH",
            details=["Signature match: Known malware pattern"],
            metadata={"signature_database": "updated", "matches_found": 1},
            timestamp=datetime.now()
        )

    def _create_empty_decision(self, fusion_id: str) -> FusionDecision:
        """Create empty decision when no engine results available"""
        return FusionDecision(
            final_threat_score=0.0,
            final_threat_level="QUANTUM_CLEAN",
            confidence=0.0,
            consensus_level="NO_CONSENSUS",
            contributing_engines=[],
            fusion_strategy="no_engines",
            details={"error": "No engine results available", "fusion_id": fusion_id}
        )

    def _create_error_decision(self, error_msg: str, fusion_id: str) -> FusionDecision:
        """Create error decision when fusion fails"""
        return FusionDecision(
            final_threat_score=0.0,
            final_threat_level="ERROR",
            confidence=0.0,
            consensus_level="ERROR",
            contributing_engines=[],
            fusion_strategy="error",
            details={"error": error_msg, "fusion_id": fusion_id}
        )

    def get_fusion_statistics(self) -> Dict[str, Any]:
        """Get fusion engine statistics and performance metrics"""
        return {
            "total_parallel_executions": self.parallel_executions,
            "engine_weights": self.engine_weights,
            "active_engines": list(self.engine_weights.keys())
        }

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including security metrics"""
        fusion_stats = self.get_fusion_statistics()
        db_stats = asyncio.run(self.db_manager.get_performance_stats())
        
        return {
            **fusion_stats,
            "database_performance": db_stats,
            "system_specs": SystemSpecs.detect(),
            "optimal_configuration": self.optimal_config,
            "security_metrics": {
                "input_validation": "enabled",
                "path_traversal_protection": "enabled",
                "sandbox_analysis": "available",
                "threat_intel_sync": "available"
            }
        }

    def update_engine_weights(self, new_weights: Dict[str, float]):
        """Update engine weights with validation"""
        for engine, weight in new_weights.items():
            if engine in self.engine_weights and 0 <= weight <= 1:
                self.engine_weights[engine] = weight
                self.logger.info(f"Updated {engine} weight to {weight:.3f}")
            else:
                self.logger.warning(f"Invalid weight for {engine}: {weight}")

# =============================================================================
# FastAPI REST API Integration
# =============================================================================

# Pydantic models for API
class ScanRequest(BaseModel):
    file_path: str
    engines: List[str] = ["quantum_engine", "ml_engine", "behavioral_engine", "signature_engine"]
    strategy: str = "adaptive_fusion"
    use_sandbox: bool = False
    sync_threat_intel: bool = False

class ScanResponse(BaseModel):
    scan_id: str
    threat_level: str
    threat_score: float
    confidence: float
    consensus: str
    processing_time: float
    engines_used: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    database_status: str
    system_specs: Dict[str, Any]

# Global engine instance
fusion_engine = EnhancedFusionIntelligenceEngine()

@app.post("/scan", response_model=ScanResponse)
async def api_scan_file(request: ScanRequest):
    """API endpoint for file scanning"""
    start_time = time.time()
    
    try:
        report = await fusion_engine.universal_scan_analysis(
            request.file_path,
            use_sandbox=request.use_sandbox,
            sync_threat_intel=request.sync_threat_intel
        )
        
        processing_time = time.time() - start_time
        
        return ScanResponse(
            scan_id=report["scan_id"],
            threat_level=report["final_decision"]["threat_level"],
            threat_score=report["final_decision"]["threat_score"],
            confidence=report["final_decision"]["confidence"],
            consensus=report["final_decision"]["consensus"],
            processing_time=processing_time,
            engines_used=report["system_info"]["total_engines_used"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def api_get_statistics():
    """API endpoint for engine statistics"""
    return fusion_engine.get_comprehensive_statistics()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        await fusion_engine.db_manager.get_performance_stats()
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        database_status=db_status,
        system_specs=SystemSpecs.detect()
    )

# =============================================================================
# Demo and Execution
# =============================================================================

async def comprehensive_demo():
    """Comprehensive demonstration of all enhanced features"""
    print("🚀 AI Model Sentinel v2.0.0 - Enhanced Fusion Engine")
    print("=" * 70)
    
    # Create enhanced fusion engine
    engine = EnhancedFusionIntelligenceEngine()
    await engine.initialize()
    
    print("\n1. Testing Universal Scan Analysis...")
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"demo file content for testing universal features")
        temp_file = f.name
    
    try:
        # Test universal scan with all features
        report = await engine.universal_scan_analysis(
            temp_file, 
            use_sandbox=True,
            sync_threat_intel=True
        )
        
        print(f"   ✓ Universal scan completed: {report['final_decision']['threat_level']}")
        print(f"   ✓ Processing time: {report['processing_time']:.2f}s")
        print(f"   ✓ Engines used: {report['system_info']['total_engines_used']}")
        
        print("\n2. Testing System Statistics...")
        stats = engine.get_comprehensive_statistics()
        print(f"   ✓ System specs: {stats['system_specs']['os']} with {stats['system_specs']['ram_gb']:.1f}GB RAM")
        print(f"   ✓ Optimal config: {stats['optimal_configuration']['performance']}")
        
        print("\n3. Testing Database Performance...")
        db_stats = await engine.db_manager.get_performance_stats()
        print(f"   ✓ Database records: {db_stats['total_records']}")
        print(f"   ✓ Average accuracy: {db_stats['accuracy']:.3f}")
        
        print("\n🎯 Enhanced Fusion Engine Demo Completed Successfully!")
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

if __name__ == "__main__":
    # Run comprehensive demo
    print("Starting AI Model Sentinel v2.0.0 - Universal Fusion Engine")
    asyncio.run(comprehensive_demo())