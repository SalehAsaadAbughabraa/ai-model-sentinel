"""
Enterprise AI Sentinel - Global Production System v2.0.0
World's Most Advanced AI Model Monitoring & Anomaly Detection System
Developer: Saleh Asaad Abughabra  
Email: saleh87alally@gmail.com
License: MIT - Global Enterprise
"""

import numpy as np
import logging
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from diagnostic_numpy_fixes import NumPyStabilityFixer

# ==================== ENTERPRISE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sentinel_production.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Enterprise_AI_Sentinel')

class DataQualityEngine:
    """Production Data Quality Analysis Engine - STRICT MODE"""
    
    def __init__(self):
        self.stability_fixer = NumPyStabilityFixer()
    
    def analyze_data_quality(self, data: np.ndarray) -> Tuple[float, List[str], Dict[str, Any]]:
        """STRICT Data Quality Assessment - Zero Tolerance for Bad Data"""
        issues = []
        metrics = {}
        
        try:
            # 🚨 CRITICAL VALIDATION - IMMEDIATE FAILURES
            if data.size == 0:
                return 0.0, ["CRITICAL: Empty data array"], metrics
            
            # Check for complete data corruption
            if np.all(np.isnan(data)):
                return 0.0, ["CRITICAL: All values are NaN"], metrics
            
            if np.all(np.isinf(data)):
                return 0.0, ["CRITICAL: All values are Infinite"], metrics
            
            if np.all(data == 0):
                return 0.1, ["CRITICAL: All values are Zero"], metrics
            
            # Statistical analysis with error handling
            with np.errstate(all='ignore'):
                valid_data = data[~np.isnan(data) & ~np.isinf(data)]
                
                if valid_data.size == 0:
                    return 0.0, ["CRITICAL: No valid data points after cleaning"], metrics                
                
                metrics.update({
                    'data_points': data.size,
                    'valid_data_points': valid_data.size,
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'median': float(np.median(valid_data))
                })
            
            # 🎯 STRICT QUALITY SCORE CALCULATION
            quality_score = 1.0
            
            # Heavy penalties for invalid data
            nan_count = np.sum(np.isnan(data))
            inf_count = np.sum(np.isinf(data))
            total_invalid = nan_count + inf_count
            invalid_ratio = total_invalid / data.size
            
            if invalid_ratio > 0:
                # Severe penalty for any invalid data
                penalty = min(invalid_ratio * 2.0, 0.8)  # Up to 80% penalty
                quality_score -= penalty
                
                if nan_count > 0:
                    issues.append(f"DATA_CORRUPTION: {nan_count} NaN values ({nan_count/data.size:.1%})")
                if inf_count > 0:
                    issues.append(f"DATA_CORRUPTION: {inf_count} Infinite values ({inf_count/data.size:.1%})")
            
            # Variability check - critical for meaningful analysis
            if metrics['std'] == 0:
                quality_score = 0.3  # Severe penalty for constant data
                issues.append("CRITICAL: Zero standard deviation - constant data")
            
            # Data range sanity check
            data_range = metrics['max'] - metrics['min']
            if data_range > 1e15:  # Unreasonably large range
                quality_score *= 0.5
                issues.append("SUSPICIOUS: Extremely large data range detected")
            
            quality_score = max(0.0, min(1.0, quality_score))
            return quality_score, issues, metrics
            
        except Exception as e:
            logger.error(f"Data quality analysis failed: {str(e)}")
            return 0.0, [f"ANALYSIS_ERROR: {str(e)}"], {}

class AdvancedAnomalyDetector:
    """BALANCED Anomaly Detection System - Optimal Sensitivity"""
    
    def __init__(self):
        self.anomaly_threshold = 3.0  # Standard threshold for normal distributions
    
    def detect_anomalies(self, data: np.ndarray, metrics: Dict) -> Tuple[float, List[str], Dict[str, Any]]:
        """BALANCED Multi-dimensional anomaly detection"""
        anomalies = []
        anomaly_metrics = {}
        
        try:
            if data.size < 10:
                return 1.0, ["INFO: Insufficient data for anomaly detection"], anomaly_metrics
            
            # Clean data - remove NaN and Inf
            clean_data = data[~np.isnan(data) & ~np.isinf(data)]
            
            if clean_data.size < 10:
                return 0.5, ["WARNING: Limited valid data points"], anomaly_metrics
            
            # Calculate robust statistics
            with np.errstate(all='ignore'):
                mean = np.mean(clean_data)
                std = np.std(clean_data)
                median = np.median(clean_data)
                mad = np.median(np.abs(clean_data - median))  # Median Absolute Deviation
            
            # 🎯 BALANCED ANOMALY DETECTION
            outlier_count = 0  # FIX: Initialize variables
            outlier_ratio = 0.0
            
            # Use MAD for more robust outlier detection
            if mad > 0:
                modified_z_scores = 0.6745 * np.abs(clean_data - median) / mad
                outliers = modified_z_scores > self.anomaly_threshold
                outlier_count = np.sum(outliers)
                outlier_ratio = outlier_count / clean_data.size
            else:
                # Fallback to standard Z-score if MAD is zero
                if std > 0:
                    z_scores = np.abs((clean_data - mean) / std)
                    outliers = z_scores > self.anomaly_threshold
                    outlier_count = np.sum(outliers)
                    outlier_ratio = outlier_count / clean_data.size
                else:
                    outlier_count = 0
                    outlier_ratio = 0.0
            
            anomaly_metrics.update({
                'outlier_count': int(outlier_count),
                'outlier_ratio': float(outlier_ratio),
                'std_dev': float(std),
                'mad': float(mad),
                'detection_method': 'MAD' if mad > 0 else 'Z-score'
            })
            
            # 🎯 BALANCED ANOMALY CLASSIFICATION
            if outlier_ratio > 0.10:  # 10% outliers = CRITICAL
                anomalies.append(f"CRITICAL_ANOMALIES: {outlier_count} outliers ({outlier_ratio:.1%})")
            elif outlier_ratio > 0.05:  # 5% outliers = HIGH
                anomalies.append(f"HIGH_ANOMALIES: {outlier_count} outliers ({outlier_ratio:.1%})")
            elif outlier_ratio > 0.02:  # 2% outliers = MODERATE
                anomalies.append(f"MODERATE_ANOMALIES: {outlier_count} outliers ({outlier_ratio:.1%})")
            elif outlier_ratio > 0.005:  # 0.5% outliers = LOW
                anomalies.append(f"LOW_ANOMALIES: {outlier_count} outliers ({outlier_ratio:.1%})")
            
            # 🎯 BALANCED ANOMALY SCORE CALCULATION
            anomaly_score = 1.0
            
            # Reasonable penalties based on outlier ratio
            if outlier_ratio > 0.10:
                anomaly_score = 0.3  # Critical
            elif outlier_ratio > 0.05:
                anomaly_score = 0.6  # High
            elif outlier_ratio > 0.02:
                anomaly_score = 0.8  # Moderate
            elif outlier_ratio > 0.005:
                anomaly_score = 0.95  # Low
                
            anomaly_score = max(0.0, min(1.0, anomaly_score))
            return anomaly_score, anomalies, anomaly_metrics
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return 0.5, [f"DETECTION_WARNING: {str(e)}"], {}

class Enterprise_AI_Sentinel:
    """
    ENTERPRISE AI SENTINEL v2.0.0 - BALANCED PRODUCTION SYSTEM
    Advanced AI Model Monitoring & Security Platform
    """
    
    def __init__(self, config: Optional[Dict] = None):
        # Initialize core engines
        self.data_quality_engine = DataQualityEngine()
        self.anomaly_detector = AdvancedAnomalyDetector()
        self.stability_fixer = NumPyStabilityFixer()
        
        # BALANCED Enterprise configuration
        self.config = {
            'min_data_points': 10,
            'health_threshold_green': 0.8,
            'health_threshold_yellow': 0.5,
            'version': '2.0.0',
            'balanced_mode': True
        }
        
        if config:
            self.config.update(config)
        
        logger.info(f"Enterprise_AI_Sentinel v{self.config['version']} initialized - BALANCED MODE")
    
    def _calculate_model_signature(self, data: np.ndarray, model_name: str) -> str:
        """Generate cryptographic model signature"""
        data_hash = hashlib.sha256(data.tobytes()).hexdigest()
        timestamp = datetime.now().isoformat()
        signature_string = f"{model_name}_{data_hash}_{timestamp}"
        return hashlib.sha256(signature_string.encode()).hexdigest()[:16]
    
    def _determine_risk_level(self, health_score: float) -> str:
        """BALANCED Enterprise risk assessment"""
        if health_score >= self.config['health_threshold_green']:
            return "LOW_RISK"
        elif health_score >= self.config['health_threshold_yellow']:
            return "MEDIUM_RISK"
        else:
            return "HIGH_RISK"
    
    def analyze_model_enterprise(self, model_data, model_name: str) -> Dict[str, Any]:
        """
        BALANCED ENTERPRISE-GRADE MODEL ANALYSIS
        Optimal Balance Between Sensitivity and Specificity
        """
        start_time = datetime.now()
        model_signature = self._calculate_model_signature(np.array(model_data), model_name)
        
        logger.info(f"Starting analysis for model: {model_name}")
        
        try:
            # Convert and validate input data
            data = np.array(model_data)
            
            if data.size < self.config['min_data_points']:
                return {
                    'health_score': 0.0,
                    'analysis_status': 'REJECTED',
                    'risk_level': 'HIGH_RISK',
                    'timestamp': datetime.now().isoformat(),
                    'error': f"Insufficient data: {data.size} points"
                }
            
            # Execute analysis pipeline
            quality_score, quality_issues, quality_metrics = self.data_quality_engine.analyze_data_quality(data)
            anomaly_score, anomaly_issues, anomaly_metrics = self.anomaly_detector.detect_anomalies(data, quality_metrics)
            
            # 🎯 BALANCED HEALTH SCORE CALCULATION
            # Equal weighting for balanced approach
            overall_health = (quality_score * 0.5) + (anomaly_score * 0.5)
            
            # Moderate penalties for critical issues
            all_issues = quality_issues + anomaly_issues
            critical_issues = [issue for issue in all_issues if 'CRITICAL' in issue]
            
            if critical_issues:
                overall_health *= 0.7  # Moderate penalty for critical issues
            
            # Risk assessment
            risk_level = self._determine_risk_level(overall_health)
            confidence = min(overall_health * 1.1, 1.0)  # Conservative confidence
            
            # 🎯 BALANCED STATUS DETERMINATION
            if overall_health >= 0.8:
                status = "OPTIMAL"
            elif overall_health >= 0.6:
                status = "HEALTHY"
            elif overall_health >= 0.4:
                status = "DEGRADED"
            else:
                status = "CRITICAL"
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            # Prepare comprehensive result
            result = {
                'health_score': round(overall_health, 4),
                'analysis_status': status,
                'risk_level': risk_level,
                'confidence': round(confidence, 4),
                'metrics': {
                    'data_quality_score': quality_score,
                    'anomaly_score': anomaly_score,
                    'quality_issues': quality_issues,
                    'anomaly_issues': anomaly_issues,
                    'total_issues': len(all_issues),
                    'critical_issues': len(critical_issues),
                    'statistical_metrics': {**quality_metrics, **anomaly_metrics}
                },
                'timestamp': datetime.now().isoformat(),
                'model_signature': model_signature,
                'analysis_time_seconds': analysis_time,
                'version': self.config['version'],
                'balanced_mode': self.config['balanced_mode']
            }
            
            logger.info(f"Analysis completed - Health: {overall_health:.3f}, Status: {status}, Issues: {len(all_issues)}")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed - Model: {model_name}, Error: {str(e)}")
            return {
                'health_score': 0.0,
                'analysis_status': 'SYSTEM_ERROR',
                'risk_level': 'CRITICAL_RISK',
                'timestamp': datetime.now().isoformat(),
                'error': f"Analysis failed: {str(e)}",
                'version': self.config['version']
            }

# Global initialization
logger.info("ENTERPRISE AI SENTINEL v2.0.0 - BALANCED PRODUCTION SYSTEM ACTIVATED")