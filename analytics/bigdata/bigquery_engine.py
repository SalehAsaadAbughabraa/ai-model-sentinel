"""
🎯 Local Analytics Engine
📦 High-performance local analytics with PostgreSQL & DuckDB
👨‍💻 Author: Saleh Abughabraa
🚀 Version: 2.0.0
💡 Business Logic: 
   - Provides high-performance analytics on security datasets using local databases
   - Supports advanced analytics with PostgreSQL window functions and DuckDB for large datasets
   - Enables real-time analytics and predictive modeling with local ML libraries
   - Supports geospatial analysis and anomaly detection
"""

import logging
import pandas as pd
import numpy as np
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone, timedelta
from functools import lru_cache
import os
import json

from config.settings import settings

logger = logging.getLogger("LocalAnalyticsEngine")


class DatabaseEngine:
    """Database abstraction layer supporting PostgreSQL and DuckDB"""
    
    def __init__(self):
        self.postgres_engine = None
        self.duckdb_connection = None
        self.active_engine = None
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize database engines"""
        try:
            # Try PostgreSQL first
            from sqlalchemy import create_engine, text
            self.postgres_engine = create_engine(settings.database.url)
            
            # Test PostgreSQL connection
            with self.postgres_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.active_engine = 'postgresql'
            logger.info("✅ PostgreSQL engine connected successfully")
            
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL connection failed: {e}")
            # Fallback to DuckDB
            try:
                import duckdb
                db_path = "analytics_data.duckdb"
                self.duckdb_connection = duckdb.connect(db_path)
                self.active_engine = 'duckdb'
                logger.info("✅ DuckDB engine connected successfully")
                self._initialize_duckdb_schema()
            except ImportError:
                logger.error("❌ Neither PostgreSQL nor DuckDB available")
                self.active_engine = None
    
    def _initialize_duckdb_schema(self):
        """Initialize DuckDB schema and tables"""
        if self.active_engine != 'duckdb':
            return
        
        try:
            # Create tables if they don't exist
            self.duckdb_connection.execute("""
                CREATE TABLE IF NOT EXISTS scan_results (
                    id BIGINT,
                    tenant_id VARCHAR,
                    repository VARCHAR,
                    threat_level VARCHAR,
                    threat_score FLOAT,
                    scan_duration FLOAT,
                    models_scanned INTEGER,
                    status VARCHAR,
                    geographic_region VARCHAR,
                    created_at TIMESTAMP,
                    user_id VARCHAR
                )
            """)
            
            self.duckdb_connection.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id BIGINT,
                    tenant_id VARCHAR,
                    user_id VARCHAR,
                    action VARCHAR,
                    resource_type VARCHAR,
                    resource_id VARCHAR,
                    success BOOLEAN,
                    created_at TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            self.duckdb_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_results_tenant_created 
                ON scan_results(tenant_id, created_at)
            """)
            
            self.duckdb_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_results_threat 
                ON scan_results(threat_level, threat_score)
            """)
            
            logger.info("✅ DuckDB schema initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ DuckDB schema initialization failed: {e}")
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute query on active database engine"""
        if self.active_engine == 'postgresql':
            return self._execute_postgres_query(query, params)
        elif self.active_engine == 'duckdb':
            return self._execute_duckdb_query(query, params)
        else:
            raise RuntimeError("No database engine available")
    
    def _execute_postgres_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute PostgreSQL query"""
        from sqlalchemy import text
        
        try:
            with self.postgres_engine.connect() as conn:
                # Convert named parameters to PostgreSQL format
                if params:
                    formatted_query = query
                    for key, value in params.items():
                        if isinstance(value, datetime):
                            formatted_value = f"'{value.isoformat()}'"
                        elif isinstance(value, str):
                            formatted_value = f"'{value}'"
                        else:
                            formatted_value = str(value)
                        formatted_query = formatted_query.replace(f":{key}", formatted_value)
                    result = conn.execute(text(formatted_query))
                else:
                    result = conn.execute(text(query))
                
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
                
        except Exception as e:
            logger.error(f"❌ PostgreSQL query failed: {e}")
            raise
    
    def _execute_duckdb_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute DuckDB query"""
        try:
            if params:
                # Convert named parameters to positional parameters for DuckDB
                positional_params = []
                formatted_query = query
                for key, value in params.items():
                    formatted_query = formatted_query.replace(f":{key}", "?")
                    positional_params.append(value)
                
                result = self.duckdb_connection.execute(formatted_query, positional_params)
            else:
                result = self.duckdb_connection.execute(query)
            
            df = result.df()
            return df
            
        except Exception as e:
            logger.error(f"❌ DuckDB query failed: {e}")
            raise


class LocalAnalyticsEngine:
    """
    💻 Local Analytics Engine for AI Model Sentinel
    💡 Provides high-performance analytics using local databases
    """
    
    def __init__(self):
        self.db_engine = DatabaseEngine()
        self.is_connected = self.db_engine.active_engine is not None
        self.ml_models = {}
        self._initialize_ml_models()
    
    def _initialize_ml_models(self):
        """Initialize local ML models for predictive analytics"""
        try:
            # Lazy imports for ML libraries
            self.ml_imports_available = False
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.preprocessing import StandardScaler
                import statsmodels.api as sm
                self.ml_imports_available = True
                logger.info("✅ ML libraries imported successfully")
            except ImportError as e:
                logger.warning(f"⚠️ ML libraries not available: {e}")
                
        except Exception as e:
            logger.error(f"❌ ML models initialization failed: {e}")
    
    async def get_realtime_threat_analytics(self, tenant_id: str, hours: int = 24) -> Dict[str, Any]:
        """
        ⚡ Get real-time threat analytics for the last N hours
        💡 Provides near real-time insights for security monitoring
        """
        if not self.is_connected:
            return {"error": "Local analytics engine not available"}
        
        try:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=hours)
            
            # Real-time threat detection query
            realtime_query = """
                SELECT 
                    -- Time-based aggregations
                    DATE_TRUNC('hour', created_at) as time_window,
                    
                    -- Threat metrics
                    COUNT(*) as total_scans,
                    SUM(CASE WHEN threat_level = 'HIGH' THEN 1 ELSE 0 END) as high_threats,
                    SUM(CASE WHEN threat_level = 'CRITICAL' THEN 1 ELSE 0 END) as critical_threats,
                    AVG(threat_score) as avg_threat_score,
                    
                    -- Performance metrics
                    AVG(scan_duration) as avg_scan_duration,
                    AVG(models_scanned) as avg_models_scanned
                    
                FROM scan_results
                WHERE tenant_id = :tenant_id
                AND created_at BETWEEN :start_time AND :end_time
                GROUP BY time_window
                ORDER BY time_window
            """
            
            realtime_data = self.db_engine.execute_query(realtime_query, {
                'tenant_id': tenant_id,
                'start_time': start_time,
                'end_time': end_time
            })
            
            # Top threats in the period
            top_threats_query = """
                SELECT 
                    repository,
                    threat_level,
                    threat_score,
                    scan_duration,
                    created_at
                FROM scan_results
                WHERE tenant_id = :tenant_id
                AND created_at BETWEEN :start_time AND :end_time
                AND threat_score >= 0.7
                ORDER BY threat_score DESC, created_at DESC
                LIMIT 10
            """
            
            top_threats = self.db_engine.execute_query(top_threats_query, {
                'tenant_id': tenant_id,
                'start_time': start_time,
                'end_time': end_time
            })
            
            return {
                'realtime_trends': realtime_data.to_dict('records'),
                'top_threats': top_threats.to_dict('records'),
                'timeframe': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'hours': hours
                },
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Real-time threat analytics failed: {e}")
            return {"error": str(e)}
    
    async def get_anomaly_detection_insights(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """
        🔍 Detect anomalies in security scanning patterns
        💡 Uses local ML for advanced anomaly detection
        """
        if not self.is_connected:
            return {"error": "Local analytics engine not available"}
        
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Get daily metrics for anomaly detection
            daily_metrics_query = """
                SELECT 
                    DATE(created_at) as scan_date,
                    COUNT(*) as scan_count,
                    AVG(threat_score) as avg_threat_score,
                    AVG(scan_duration) as avg_duration,
                    SUM(CASE WHEN threat_level = 'HIGH' THEN 1 ELSE 0 END) as high_threat_count
                FROM scan_results
                WHERE tenant_id = :tenant_id
                AND created_at BETWEEN :start_date AND :end_date
                GROUP BY scan_date
                ORDER BY scan_date
            """
            
            daily_data = self.db_engine.execute_query(daily_metrics_query, {
                'tenant_id': tenant_id,
                'start_date': start_date,
                'end_date': end_date
            })
            
            if daily_data.empty:
                return {"error": "No data available for anomaly detection"}
            
            # Perform statistical analysis for anomalies
            anomaly_data = self._perform_statistical_anomaly_detection(daily_data)
            
            return {
                'anomaly_detection_results': anomaly_data.to_dict('records'),
                'anomalies_found': len(anomaly_data[anomaly_data['anomaly_status'] == 'ANOMALY']),
                'anomaly_details': anomaly_data[anomaly_data['anomaly_status'] == 'ANOMALY'].to_dict('records'),
                'timeframe': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days
                },
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection failed: {e}")
            return {"error": str(e)}
    
    def _perform_statistical_anomaly_detection(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """Perform statistical anomaly detection using Z-scores"""
        try:
            df = daily_data.copy()
            
            # Calculate Z-scores for scan_count
            scan_mean = df['scan_count'].mean()
            scan_std = df['scan_count'].std()
            df['scan_count_zscore'] = (df['scan_count'] - scan_mean) / (scan_std if scan_std > 0 else 1)
            
            # Calculate Z-scores for threat_score
            threat_mean = df['avg_threat_score'].mean()
            threat_std = df['avg_threat_score'].std()
            df['threat_score_zscore'] = (df['avg_threat_score'] - threat_mean) / (threat_std if threat_std > 0 else 1)
            
            # Mark anomalies (Z-score > 2 or < -2)
            df['anomaly_status'] = 'NORMAL'
            df.loc[
                (df['scan_count_zscore'].abs() > 2) | (df['threat_score_zscore'].abs() > 2), 
                'anomaly_status'
            ] = 'ANOMALY'
            
            return df.round(4)
            
        except Exception as e:
            logger.error(f"❌ Statistical anomaly detection failed: {e}")
            return daily_data
    
    async def get_geospatial_analysis(self, tenant_id: str, days: int = 30) -> Dict[str, Any]:
        """
        🌍 Perform geospatial analysis on threat data
        💡 Analyzes geographic patterns in security threats
        """
        if not self.is_connected:
            return {"error": "Local analytics engine not available"}
        
        try:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Geospatial threat distribution
            geospatial_query = """
                SELECT 
                    geographic_region,
                    COUNT(*) as total_scans,
                    AVG(threat_score) as avg_threat_score,
                    SUM(CASE WHEN threat_level = 'HIGH' THEN 1 ELSE 0 END) as high_threats,
                    SUM(CASE WHEN threat_level = 'CRITICAL' THEN 1 ELSE 0 END) as critical_threats
                FROM scan_results
                WHERE tenant_id = :tenant_id
                AND created_at BETWEEN :start_date AND :end_date
                AND geographic_region != 'global'
                GROUP BY geographic_region
                ORDER BY total_scans DESC
            """
            
            geospatial_data = self.db_engine.execute_query(geospatial_query, {
                'tenant_id': tenant_id,
                'start_date': start_date,
                'end_date': end_date
            })
            
            # Regional threat hotspots
            hotspot_query = """
                SELECT 
                    geographic_region,
                    repository,
                    COUNT(*) as scan_count,
                    AVG(threat_score) as avg_threat_score
                FROM scan_results
                WHERE tenant_id = :tenant_id
                AND created_at BETWEEN :start_date AND :end_date
                AND threat_score >= 0.7
                GROUP BY geographic_region, repository
                ORDER BY avg_threat_score DESC
                LIMIT 20
            """
            
            hotspot_data = self.db_engine.execute_query(hotspot_query, {
                'tenant_id': tenant_id,
                'start_date': start_date,
                'end_date': end_date
            })
            
            return {
                'regional_distribution': geospatial_data.to_dict('records'),
                'threat_hotspots': hotspot_data.to_dict('records'),
                'timeframe': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days
                },
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Geospatial analysis failed: {e}")
            return {"error": str(e)}
    
    async def get_ml_predictive_insights(self, tenant_id: str) -> Dict[str, Any]:
        """
        🤖 Generate predictive insights using local ML
        💡 Uses machine learning to predict future threats and trends
        """
        if not self.is_connected:
            return {"error": "Local analytics engine not available"}
        
        try:
            # Get historical data for forecasting
            historical_query = """
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as daily_scans,
                    AVG(threat_score) as avg_threat_score
                FROM scan_results
                WHERE tenant_id = :tenant_id
                AND created_at >= CURRENT_DATE - INTERVAL '90 days'
                GROUP BY date
                ORDER BY date
            """
            
            historical_data = self.db_engine.execute_query(historical_query, {
                'tenant_id': tenant_id
            })
            
            if historical_data.empty:
                return {"error": "No historical data available for prediction"}
            
            # Generate forecasts using local methods
            forecast_data = self._generate_local_forecasts(historical_data)
            
            # Threat trend analysis
            trend_query = """
                SELECT 
                    threat_level,
                    COUNT(*) as occurrence_count,
                    AVG(threat_score) as avg_score
                FROM scan_results
                WHERE tenant_id = :tenant_id
                AND created_at >= CURRENT_DATE - INTERVAL '60 days'
                GROUP BY threat_level
                ORDER BY occurrence_count DESC
            """
            
            trend_data = self.db_engine.execute_query(trend_query, {
                'tenant_id': tenant_id
            })
            
            return {
                'activity_forecast': forecast_data.to_dict('records'),
                'threat_trends': trend_data.to_dict('records'),
                'predictive_insights': {
                    'next_week_forecast': self._predict_next_week_activity(historical_data),
                    'high_risk_periods': self._identify_high_risk_periods(historical_data),
                    'recommended_actions': [
                        "Monitor repositories with recent high threat scores",
                        "Review security configurations for critical assets",
                        "Consider increasing scan frequency during predicted high-activity periods"
                    ]
                },
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Predictive insights failed: {e}")
            return {"error": str(e)}
    
    def _generate_local_forecasts(self, historical_data: pd.DataFrame) -> pd.DataFrame:
        """Generate forecasts using local statistical methods"""
        try:
            df = historical_data.copy()
            
            # Simple moving average forecast
            df['predicted_scans'] = df['daily_scans'].rolling(
                window=7, min_periods=1
            ).mean().shift(1)
            
            # Activity level classification
            avg_scans = df['daily_scans'].mean()
            df['activity_level'] = 'NORMAL_ACTIVITY'
            df.loc[df['predicted_scans'] > avg_scans * 1.2, 'activity_level'] = 'HIGH_ACTIVITY'
            df.loc[df['predicted_scans'] < avg_scans * 0.8, 'activity_level'] = 'LOW_ACTIVITY'
            
            return df.tail(30).round(2)
            
        except Exception as e:
            logger.error(f"❌ Local forecast generation failed: {e}")
            return historical_data.tail(30)
    
    def _predict_next_week_activity(self, historical_data: pd.DataFrame) -> str:
        """Predict next week's activity level"""
        try:
            recent_avg = historical_data['daily_scans'].tail(7).mean()
            overall_avg = historical_data['daily_scans'].mean()
            
            if recent_avg > overall_avg * 1.2:
                return "HIGH_ACTIVITY"
            elif recent_avg < overall_avg * 0.8:
                return "LOW_ACTIVITY"
            else:
                return "NORMAL_ACTIVITY"
                
        except Exception:
            return "NORMAL_ACTIVITY"
    
    def _identify_high_risk_periods(self, historical_data: pd.DataFrame) -> List[str]:
        """Identify high-risk periods based on historical data"""
        try:
            high_risk_days = historical_data[
                historical_data['avg_threat_score'] > historical_data['avg_threat_score'].quantile(0.75)
            ]['date'].dt.strftime('%Y-%m-%d').tolist()
            
            return high_risk_days[-5:]  # Return last 5 high-risk days
            
        except Exception:
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """
        ❤️ Perform local analytics engine health check
        💡 Verifies connectivity and basic functionality
        """
        if not self.is_connected:
            return {
                "status": "disconnected",
                "message": "Local analytics engine not available",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        try:
            start_time = time.time()
            
            # Test connection with simple query
            test_query = "SELECT COUNT(*) as table_count FROM scan_results LIMIT 1"
            result = self.db_engine.execute_query(test_query)
            
            response_time = time.time() - start_time
            
            health_status = {
                "status": "healthy",
                "connected": True,
                "database_engine": self.db_engine.active_engine,
                "response_time_seconds": round(response_time, 3),
                "test_query_successful": not result.empty,
                "ml_capabilities": self.ml_imports_available,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            if response_time > 1.0:
                health_status["status"] = "degraded"
                health_status["warning"] = "High response time detected"
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Global local analytics engine instance
local_analytics_engine = LocalAnalyticsEngine()


async def initialize_local_analytics() -> bool:
    """
    🚀 Initialize local analytics engine
    💡 Main entry point for local analytics setup
    """
    try:
        # Engine is initialized in constructor
        if local_analytics_engine.is_connected:
            logger.info("✅ Local analytics engine initialized successfully")
        else:
            logger.warning("⚠️ Local analytics engine initialization failed")
        
        return local_analytics_engine.is_connected
        
    except Exception as e:
        logger.error(f"❌ Local analytics initialization failed: {e}")
        return False