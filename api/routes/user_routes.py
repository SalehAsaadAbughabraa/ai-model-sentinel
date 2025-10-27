# analytics/drift_detector.py
"""
🏭 Production Drift Detection System v2.0.0
Enterprise-ready drift detection for AI models
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import logging
import time
import warnings
from scipy import stats
from scipy.spatial.distance import jensenshannon
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from datetime import datetime

try:
    from alibi_detect.cd import KSDrift, MMDDrift, CVMDrift
    ALIBI_AVAILABLE = True
except ImportError:
    ALIBI_AVAILABLE = False

class DriftType(Enum):
    """Supported Drift Types"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    MODEL_DRIFT = "model_drift"
    FEATURE_DRIFT = "feature_drift"

class DetectionMethod(Enum):
    """Supported Detection Methods"""
    KOLMOGOROV_SMIRNOV = "ks"
    MAXIMUM_MEAN_DISCREPANCY = "mmd"
    CRAMER_VON_MISES = "cvm"
    STATISTICAL = "statistical"

@dataclass
class DriftConfig:
    """Drift Detection Configuration"""
    reference_data: Optional[np.ndarray] = None
    drift_types: List[DriftType] = field(default_factory=lambda: [DriftType.DATA_DRIFT])
    detection_methods: List[DetectionMethod] = field(default_factory=lambda: [DetectionMethod.KOLMOGOROV_SMIRNOV])
    significance_level: float = 0.05
    window_size: int = 1000
    enable_adaptive_threshold: bool = True
    max_workers: int = 4

class StatisticalAnalyzer:
    """Statistical Analyzer for Drift Detection"""
    
    def __init__(self):
        self.logger = logging.getLogger('StatisticalAnalyzer')
    
    def analyze_distribution_shift(self, reference_data: np.ndarray, current_data: np.ndarray) -> Dict[str, Any]:
        """Analyze distribution shift between reference and current data"""
        try:
            results = {}
            
            # KS Test
            ks_statistic, ks_pvalue = stats.ks_2samp(reference_data.flatten(), current_data.flatten())
            results['ks_test'] = {
                'statistic': float(ks_statistic),
                'p_value': float(ks_pvalue),
                'drift_detected': ks_pvalue < 0.05
            }
            
            # Jensen-Shannon Distance
            js_distance = self._calculate_js_distance(reference_data, current_data)
            results['js_distance'] = {
                'distance': js_distance,
                'drift_detected': js_distance > 0.1
            }
            
            # Statistical Comparison
            stats_comparison = self._compare_statistics(reference_data, current_data)
            results['statistical_comparison'] = stats_comparison
            
            return results
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {e}")
            return {'error': str(e)}
    
    def _calculate_js_distance(self, ref_data: np.ndarray, curr_data: np.ndarray) -> float:
        """Calculate Jensen-Shannon distance between distributions"""
        try:
            hist_ref, bin_edges = np.histogram(ref_data.flatten(), bins=50, density=True)
            hist_curr, _ = np.histogram(curr_data.flatten(), bins=bin_edges, density=True)
            
            hist_ref = np.maximum(hist_ref, 1e-10)
            hist_curr = np.maximum(hist_curr, 1e-10)
            
            return jensenshannon(hist_ref, hist_curr)
            
        except Exception as e:
            self.logger.warning(f"JS distance calculation failed: {e}")
            return 1.0
    
    def _compare_statistics(self, ref_data: np.ndarray, curr_data: np.ndarray) -> Dict[str, Any]:
        """Compare statistics between reference and current data"""
        ref_flat = ref_data.flatten()
        curr_flat = curr_data.flatten()
        
        return {
            'means': {
                'reference': float(np.mean(ref_flat)),
                'current': float(np.mean(curr_flat)),
                'difference': float(np.abs(np.mean(ref_flat) - np.mean(curr_flat))),
                'ratio': float(np.mean(curr_flat) / np.mean(ref_flat)) if np.mean(ref_flat) != 0 else float('inf')
            },
            'std_devs': {
                'reference': float(np.std(ref_flat)),
                'current': float(np.std(curr_flat)),
                'difference': float(np.abs(np.std(ref_flat) - np.std(curr_flat))),
                'ratio': float(np.std(curr_flat) / np.std(ref_flat)) if np.std(ref_flat) != 0 else float('inf')
            }
        }

class FeatureDriftAnalyzer:
    """Feature Drift Analyzer"""
    
    def __init__(self):
        self.logger = logging.getLogger('FeatureDriftAnalyzer')
    
    def analyze_feature_drift(self, reference_data: np.ndarray, 
                            current_data: np.ndarray,
                            feature_names: List[str] = None) -> Dict[str, Any]:
        """Analyze drift for each feature individually"""
        try:
            if reference_data.shape[1] != current_data.shape[1]:
                raise ValueError("Reference and current data must have same number of features")
            
            n_features = reference_data.shape[1]
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(n_features)]
            
            results = {}
            
            for i in range(n_features):
                feature_name = feature_names[i]
                results[feature_name] = self._analyze_single_feature(
                    reference_data[:, i], current_data[:, i], feature_name
                )
            
            overall_analysis = self._analyze_overall_feature_drift(results)
            results['overall'] = overall_analysis
            
            return results
            
        except Exception as e:
            self.logger.error(f"Feature drift analysis failed: {e}")
            return {'error': str(e)}
    
    def _analyze_single_feature(self, ref_feature: np.ndarray, 
                              curr_feature: np.ndarray, 
                              feature_name: str) -> Dict[str, Any]:
        """Analyze drift for single feature"""
        analyzer = StatisticalAnalyzer()
        statistical_results = analyzer.analyze_distribution_shift(
            ref_feature.reshape(-1, 1), curr_feature.reshape(-1, 1)
        )
        
        drift_score = self._calculate_feature_drift_score(statistical_results)
        
        return {
            'statistical_analysis': statistical_results,
            'drift_score': drift_score,
            'drift_detected': drift_score > 0.5,
            'feature_type': self._infer_feature_type(ref_feature)
        }
    
    def _calculate_feature_drift_score(self, statistical_results: Dict[str, Any]) -> float:
        """Calculate feature drift score"""
        scores = []
        
        ks_pvalue = statistical_results.get('ks_test', {}).get('p_value', 1.0)
        scores.append(1.0 - ks_pvalue)
        
        js_distance = statistical_results.get('js_distance', {}).get('distance', 0.0)
        scores.append(min(js_distance * 5, 1.0))
        
        return float(np.mean(scores))
    
    def _infer_feature_type(self, feature_data: np.ndarray) -> str:
        """Infer feature type"""
        unique_values = np.unique(feature_data)
        
        if len(unique_values) <= 10:
            return "categorical"
        elif np.issubdtype(feature_data.dtype, np.number):
            return "numerical"
        else:
            return "unknown"
    
    def _analyze_overall_feature_drift(self, feature_results: Dict[str, Any]) -> Dict[str, Any]:
        """Overall feature drift analysis"""
        drift_scores = []
        drifted_features = []
        
        for feature_name, result in feature_results.items():
            if feature_name != 'overall':
                drift_score = result.get('drift_score', 0.0)
                drift_scores.append(drift_score)
                
                if result.get('drift_detected', False):
                    drifted_features.append(feature_name)
        
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        
        return {
            'overall_drift_score': float(overall_drift_score),
            'drifted_features': drifted_features,
            'drift_ratio': len(drifted_features) / len(drift_scores) if drift_scores else 0.0,
            'severity': 'high' if overall_drift_score > 0.7 else 
                       'medium' if overall_drift_score > 0.3 else 'low'
        }

class MLDriftDetector:
    """Machine Learning Drift Detector"""
    
    def __init__(self):
        self.logger = logging.getLogger('MLDriftDetector')
        self.detectors = {}
    
    def create_detectors(self, reference_data: np.ndarray, methods: List[DetectionMethod]) -> Dict[str, Any]:
        """Create drift detectors based on reference data"""
        detectors = {}
        
        try:
            if not ALIBI_AVAILABLE:
                return detectors
            
            for method in methods:
                try:
                    if method == DetectionMethod.KOLMOGOROV_SMIRNOV:
                        detectors['ks'] = KSDrift(reference_data, p_val=0.05)
                    
                    elif method == DetectionMethod.MAXIMUM_MEAN_DISCREPANCY:
                        detectors['mmd'] = MMDDrift(reference_data, p_val=0.05)
                    
                    elif method == DetectionMethod.CRAMER_VON_MISES:
                        detectors['cvm'] = CVMDrift(reference_data, p_val=0.05)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to create detector {method.value}: {e}")
            
            self.detectors = detectors
            return detectors
            
        except Exception as e:
            self.logger.error(f"Detector creation failed: {e}")
            return {}
    
    def detect_drift_ml(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect drift using ML detectors"""
        results = {}
        
        try:
            if not self.detectors:
                return {'error': 'Detectors not initialized'}
            
            for detector_name, detector in self.detectors.items():
                try:
                    detection_result = detector.predict(current_data)
                    
                    results[detector_name] = {
                        'drift_detected': detection_result['data']['is_drift'] == 1,
                        'p_value': detection_result['data'].get('p_val', None),
                        'distance': detection_result['data'].get('distance', None),
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Detection failed with {detector_name}: {e}")
                    results[detector_name] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            self.logger.error(f"ML drift detection failed: {e}")
            return {'error': str(e)}

class DriftDetector:
    """Main Drift Detector Class - Production Ready"""
    
    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.statistical_analyzer = StatisticalAnalyzer()
        self.feature_analyzer = FeatureDriftAnalyzer()
        self.ml_detector = MLDriftDetector()
        
        # Initialize ML detectors if reference data provided
        if self.config.reference_data is not None:
            self.ml_detector.create_detectors(
                self.config.reference_data, 
                self.config.detection_methods
            )
        
        self.logger.info("✅ Production Drift Detector v2.0.0 Initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging system"""
        logger = logging.getLogger('DriftDetector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def detect_drift(self, current_data: np.ndarray, 
                    feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Main drift detection method - Production Ready"""
        start_time = time.time()
        
        try:
            results = {
                'timestamp': time.time(),
                'drift_types_analyzed': [],
                'system_version': '2.0.0'
            }
            
            # Data Drift Analysis
            if DriftType.DATA_DRIFT in self.config.drift_types:
                data_drift_results = self._analyze_data_drift(current_data)
                results['data_drift'] = data_drift_results
                results['drift_types_analyzed'].append('data_drift')
            
            # Feature Drift Analysis
            if DriftType.FEATURE_DRIFT in self.config.drift_types:
                feature_drift_results = self.feature_analyzer.analyze_feature_drift(
                    self.config.reference_data, current_data, feature_names
                )
                results['feature_drift'] = feature_drift_results
                results['drift_types_analyzed'].append('feature_drift')
            
            # ML-based Detection
            ml_results = self.ml_detector.detect_drift_ml(current_data)
            results['ml_detection'] = ml_results
            
            results['processing_time'] = time.time() - start_time
            results['status'] = 'completed'
            
            return results
            
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            return {
                'error': str(e), 
                'timestamp': time.time(),
                'status': 'failed',
                'system_version': '2.0.0'
            }
    
    def _analyze_data_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Analyze data drift between reference and current data"""
        if self.config.reference_data is None:
            return {'error': 'Reference data not provided'}
        
        statistical_results = self.statistical_analyzer.analyze_distribution_shift(
            self.config.reference_data, current_data
        )
        
        ml_results = self.ml_detector.detect_drift_ml(current_data)
        
        return {
            'statistical_analysis': statistical_results,
            'ml_detection': ml_results,
            'overall_drift_detected': statistical_results.get('ks_test', {}).get('drift_detected', False)
        }

# Export main classes
__all__ = [
    'DriftDetector',
    'DriftConfig',
    'DriftType',
    'DetectionMethod'
]

# Simple test
if __name__ == "__main__":
    print("🧪 Testing Production Drift Detector...")
    
    # Sample data
    np.random.seed(42)
    reference_data = np.random.normal(0, 1, (1000, 5))
    current_data = np.random.normal(0.2, 1.2, (800, 5))
    
    # Configure detector
    config = DriftConfig(
        reference_data=reference_data,
        drift_types=[DriftType.DATA_DRIFT, DriftType.FEATURE_DRIFT]
    )
    
    # Initialize detector
    detector = DriftDetector(config)
    
    # Test detection
    results = detector.detect_drift(
        current_data=current_data,
        feature_names=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    )
    
    print(f"✅ System Version: {results.get('system_version')}")
    print(f"✅ Status: {results.get('status')}")
    print(f"✅ Processing Time: {results.get('processing_time', 0):.2f}s")
    print("🎉 Production test completed!")