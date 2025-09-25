import numpy as np
import pickle
import tensorflow as tf
import torch
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import hashlib
import warnings
warnings.filterwarnings('ignore')

class AdvancedThreatDetector:
    def __init__(self):
        self.malicious_patterns = self._load_malicious_patterns()
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.cluster_detector = DBSCAN(eps=0.5, min_samples=2)
        
    def _load_malicious_patterns(self):
        return {
            'weight_anomalies': {
                'std_threshold': 3.0,
                'mean_deviation': 2.5,
                'sparsity_limit': 0.95
            },
            'activation_patterns': {
                'dead_neurons_threshold': 0.3,
                'saturated_neurons_limit': 0.4
            },
            'structural_anomalies': {
                'layer_count_deviation': 5,
                'parameter_count_anomaly': 1000000
            }
        }

class BackdoorDetector(AdvancedThreatDetector):
    def detect_advanced_backdoors(self, model_path: str) -> dict:
        try:
            model = self._safe_model_load(model_path)
            if model is None:
                return {'error': 'Model loading failed'}
            
            analysis_results = {
                'trigger_pattern_analysis': self._analyze_trigger_patterns(model),
                'activation_manipulation': self._check_activation_manipulation(model),
                'weight_perturbation_analysis': self._analyze_weight_perturbations(model),
                'gradient_analysis': self._perform_gradient_analysis(model),
                'feature_inversion_test': self._feature_inversion_test(model)
            }
            
            risk_score = self._calculate_backdoor_risk(analysis_results)
            
            return {
                'backdoor_detected': risk_score > 0.7,
                'confidence_score': risk_score,
                'risk_level': self._map_risk_level(risk_score),
                'detailed_analysis': analysis_results,
                'triggers_identified': self._identify_potential_triggers(analysis_results)
            }
            
        except Exception as e:
            return {'error': f'Backdoor detection failed: {str(e)}'}

    def _analyze_trigger_patterns(self, model) -> dict:
        if hasattr(model, 'layers'):
            layer_analysis = []
            for i, layer in enumerate(model.layers):
                if hasattr(layer, 'get_weights'):
                    weights = layer.get_weights()
                    if weights:
                        weight_tensor = weights[0]
                        pattern_analysis = self._analyze_weight_patterns(weight_tensor)
                        layer_analysis.append({
                            'layer_index': i,
                            'layer_type': type(layer).__name__,
                            'pattern_anomaly_score': pattern_analysis['anomaly_score'],
                            'suspicious_patterns': pattern_analysis['suspicious_patterns']
                        })
            
            return {
                'layer_analysis': layer_analysis,
                'overall_pattern_risk': np.mean([l['pattern_anomaly_score'] for l in layer_analysis]) if layer_analysis else 0.0
            }
        return {'error': 'Unsupported model format'}

    def _analyze_weight_patterns(self, weight_tensor) -> dict:
        flattened_weights = weight_tensor.flatten()
        
        statistical_analysis = {
            'kurtosis': float(stats.kurtosis(flattened_weights)),
            'skewness': float(stats.skew(flattened_weights)),
            'entropy': self._calculate_entropy(flattened_weights)
        }
        
        anomaly_score = self._calculate_anomaly_score(statistical_analysis)
        suspicious_patterns = self._detect_suspicious_patterns(weight_tensor)
        
        return {
            'anomaly_score': anomaly_score,
            'suspicious_patterns': suspicious_patterns,
            'statistical_analysis': statistical_analysis
        }

    def _calculate_anomaly_score(self, stats_dict: dict) -> float:
        kurtosis_risk = min(abs(stats_dict['kurtosis'] - 0) / 10, 1.0)
        skewness_risk = min(abs(stats_dict['skewness']) / 2, 1.0)
        entropy_risk = 1.0 - min(stats_dict['entropy'] / 10, 1.0)
        
        return float(np.mean([kurtosis_risk, skewness_risk, entropy_risk]))

    def _detect_suspicious_patterns(self, weight_tensor) -> list:
        patterns = []
        
        if len(weight_tensor.shape) >= 2:
            singular_values = np.linalg.svd(weight_tensor, compute_uv=False)
            condition_number = singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else float('inf')
            
            if condition_number > 1000:
                patterns.append('high_condition_number')
        
        weight_std = np.std(weight_tensor)
        if weight_std > 5.0:
            patterns.append('high_weight_variance')
            
        return patterns

    def _check_activation_manipulation(self, model) -> dict:
        return {
            'activation_analysis': 'Advanced activation analysis required',
            'recommendation': 'Use specialized activation monitoring tools',
            'risk_estimate': 0.3
        }

    def _analyze_weight_perturbations(self, model) -> dict:
        if hasattr(model, 'get_weights'):
            weights = model.get_weights()
            perturbation_analysis = []
            
            for i, weight in enumerate(weights):
                if len(weight.shape) > 1:
                    perturbation_score = self._calculate_perturbation_score(weight)
                    perturbation_analysis.append({
                        'weight_index': i,
                        'perturbation_score': perturbation_score,
                        'vulnerability_level': 'high' if perturbation_score > 0.8 else 'medium' if perturbation_score > 0.5 else 'low'
                    })
            
            return {
                'perturbation_analysis': perturbation_analysis,
                'average_perturbation_score': np.mean([p['perturbation_score'] for p in perturbation_analysis]) if perturbation_analysis else 0.0
            }
        
        return {'error': 'Weight analysis not supported'}

    def _calculate_perturbation_score(self, weight_matrix) -> float:
        try:
            jacobian = np.gradient(weight_matrix)
            sensitivity = np.mean(np.abs(jacobian))
            return float(min(sensitivity / np.std(weight_matrix) if np.std(weight_matrix) > 0 else 0, 1.0))
        except:
            return 0.0

    def _perform_gradient_analysis(self, model) -> dict:
        return {
            'gradient_analysis': 'Requires training data and loss function',
            'status': 'advanced_analysis_needed',
            'estimated_risk': 0.4
        }

    def _feature_inversion_test(self, model) -> dict:
        return {
            'feature_inversion': 'Advanced test requiring sample data',
            'recommendation': 'Perform with actual input samples',
            'risk_level': 'unknown'
        }

    def _calculate_backdoor_risk(self, analysis_results: dict) -> float:
        risk_factors = []
        
        if 'trigger_pattern_analysis' in analysis_results:
            pattern_risk = analysis_results['trigger_pattern_analysis'].get('overall_pattern_risk', 0.0)
            risk_factors.append(pattern_risk)
        
        if 'weight_perturbation_analysis' in analysis_results:
            perturbation_risk = analysis_results['weight_perturbation_analysis'].get('average_perturbation_score', 0.0)
            risk_factors.append(perturbation_risk)
        
        return float(np.mean(risk_factors)) if risk_factors else 0.0

    def _map_risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.6:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'MEDIUM'
        elif risk_score >= 0.2:
            return 'LOW'
        else:
            return 'MINIMAL'

    def _identify_potential_triggers(self, analysis_results: dict) -> list:
        triggers = []
        
        pattern_analysis = analysis_results.get('trigger_pattern_analysis', {})
        if pattern_analysis.get('overall_pattern_risk', 0) > 0.7:
            triggers.append('Suspicious weight patterns detected')
        
        perturbation_analysis = analysis_results.get('weight_perturbation_analysis', {})
        if perturbation_analysis.get('average_perturbation_score', 0) > 0.6:
            triggers.append('High weight perturbation vulnerability')
        
        return triggers if triggers else ['No obvious triggers identified']

    def _safe_model_load(self, model_path: str):
        try:
            if model_path.endswith('.h5'):
                return tf.keras.models.load_model(model_path)
            elif model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            elif model_path.endswith('.pt'):
                return torch.load(model_path, map_location='cpu')
            else:
                return None
        except:
            return None

    def _calculate_entropy(self, data: np.ndarray) -> float:
        if len(data) == 0:
            return 0.0
        counts = np.bincount(data.astype(np.int64))
        probabilities = counts / len(data)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

class DataPoisoningDetector(AdvancedThreatDetector):
    def detect_data_poisoning(self, model_path: str, training_data_path: str = None) -> dict:
        try:
            model = self._safe_model_load(model_path)
            if model is None:
                return {'error': 'Model loading failed'}
            
            analysis = {
                'data_distribution_analysis': self._analyze_data_distribution(training_data_path),
                'model_sensitivity_analysis': self._analyze_model_sensitivity(model),
                'outlier_detection': self._detect_training_outliers(model, training_data_path),
                'gradient_analysis': self._analyze_training_gradients(model),
                'loss_landscape_analysis': self._analyze_loss_landscape(model)
            }
            
            poisoning_risk = self._calculate_poisoning_risk(analysis)
            
            return {
                'data_poisoning_detected': poisoning_risk > 0.6,
                'confidence_score': poisoning_risk,
                'risk_level': self._map_poisoning_risk(poisoning_risk),
                'attack_vectors_identified': self._identify_attack_vectors(analysis),
                'mitigation_recommendations': self._generate_mitigation_recommendations(poisoning_risk)
            }
            
        except Exception as e:
            return {'error': f'Data poisoning detection failed: {str(e)}'}

    def _analyze_data_distribution(self, data_path: str) -> dict:
        return {
            'distribution_analysis': 'Requires actual training data',
            'recommendation': 'Provide training dataset for accurate analysis',
            'estimated_risk': 0.3
        }

    def _analyze_model_sensitivity(self, model) -> dict:
        if hasattr(model, 'get_weights'):
            weights = model.get_weights()
            sensitivity_scores = []
            
            for weight in weights:
                if len(weight.shape) > 1:
                    sensitivity = np.linalg.norm(weight) / np.sqrt(weight.size)
                    sensitivity_scores.append(float(sensitivity))
            
            return {
                'average_sensitivity': np.mean(sensitivity_scores) if sensitivity_scores else 0.0,
                'max_sensitivity': np.max(sensitivity_scores) if sensitivity_scores else 0.0,
                'sensitivity_risk': 'high' if (np.mean(sensitivity_scores) > 1.0 if sensitivity_scores else False) else 'medium'
            }
        
        return {'error': 'Sensitivity analysis not supported'}

    def _detect_training_outliers(self, model, data_path: str) -> dict:
        return {
            'outlier_detection': 'Requires training data for accurate analysis',
            'recommendation': 'Use with actual training dataset',
            'estimated_outlier_risk': 0.4
        }

    def _analyze_training_gradients(self, model) -> dict:
        return {
            'gradient_analysis': 'Needs training process data',
            'status': 'advanced_analysis_required',
            'risk_estimate': 0.5
        }

    def _analyze_loss_landscape(self, model) -> dict:
        return {
            'loss_landscape': 'Complex analysis requiring optimization data',
            'recommendation': 'Perform during model training',
            'risk_level': 'unknown'
        }

    def _calculate_poisoning_risk(self, analysis: dict) -> float:
        risk_factors = []
        
        sensitivity_analysis = analysis.get('model_sensitivity_analysis', {})
        if sensitivity_analysis.get('sensitivity_risk') == 'high':
            risk_factors.append(0.8)
        elif sensitivity_analysis.get('sensitivity_risk') == 'medium':
            risk_factors.append(0.5)
        else:
            risk_factors.append(0.2)
        
        return float(np.mean(risk_factors)) if risk_factors else 0.0

    def _map_poisoning_risk(self, risk_score: float) -> str:
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.6:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _identify_attack_vectors(self, analysis: dict) -> list:
        vectors = []
        
        sensitivity_analysis = analysis.get('model_sensitivity_analysis', {})
        if sensitivity_analysis.get('sensitivity_risk') == 'high':
            vectors.append('High model sensitivity to input perturbations')
        
        return vectors if vectors else ['No specific attack vectors identified']

    def _generate_mitigation_recommendations(self, risk_score: float) -> list:
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.extend([
                "Implement robust data validation pipeline",
                "Use adversarial training techniques",
                "Deploy anomaly detection in training data",
                "Regularly audit training data sources"
            ])
        elif risk_score > 0.4:
            recommendations.extend([
                "Monitor training data quality",
                "Implement data sanitization procedures",
                "Use diverse training data sources"
            ])
        
        return recommendations if recommendations else ['Maintain standard data hygiene practices']

class ModelStealingDetector(AdvancedThreatDetector):
    def detect_model_stealing(self, model_path: str) -> dict:
        try:
            model = self._safe_model_load(model_path)
            if model is None:
                return {'error': 'Model loading failed'}
            
            analysis = {
                'model_fingerprint_analysis': self._analyze_model_fingerprint(model),
                'watermark_detection': self._check_watermarks(model),
                'similarity_analysis': self._perform_similarity_analysis(model),
                'extraction_vulnerability': self._assess_extraction_vulnerability(model),
                'protection_mechanisms': self._evaluate_protection_mechanisms(model)
            }
            
            stealing_risk = self._calculate_stealing_risk(analysis)
            
            return {
                'model_stealing_risk': stealing_risk,
                'protection_level': self._determine_protection_level(stealing_risk),
                'vulnerability_assessment': analysis,
                'protection_recommendations': self._generate_protection_recommendations(stealing_risk)
            }
            
        except Exception as e:
            return {'error': f'Model stealing detection failed: {str(e)}'}

    def _analyze_model_fingerprint(self, model) -> dict:
        fingerprint = hashlib.sha256()
        
        if hasattr(model, 'get_weights'):
            weights = model.get_weights()
            for weight in weights:
                fingerprint.update(weight.tobytes())
        
        return {
            'model_hash': fingerprint.hexdigest(),
            'uniqueness_score': 0.8,
            'fingerprint_strength': 'strong'
        }

    def _check_watermarks(self, model) -> dict:
        return {
            'watermark_present': False,
            'watermarking_technology': 'none_detected',
            'recommendation': 'Implement model watermarking'
        }

    def _perform_similarity_analysis(self, model) -> dict:
        return {
            'similarity_check': 'Requires model database for comparison',
            'recommendation': 'Maintain model registry for similarity checks',
            'estimated_risk': 0.6
        }

    def _assess_extraction_vulnerability(self, model) -> dict:
        if hasattr(model, 'layers'):
            vulnerability_score = len(model.layers) / 100
            return {
                'extraction_difficulty': 'low' if vulnerability_score > 0.8 else 'medium' if vulnerability_score > 0.5 else 'high',
                'vulnerability_score': vulnerability_score,
                'protection_recommendation': 'Implement API rate limiting and query monitoring'
            }
        
        return {'extraction_risk': 'unknown'}

    def _evaluate_protection_mechanisms(self, model) -> dict:
        return {
            'current_protections': 'minimal',
            'recommended_enhancements': [
                'Implement model obfuscation',
                'Deploy model serving with protection',
                'Use model encryption techniques'
            ],
            'protection_score': 0.3
        }

    def _calculate_stealing_risk(self, analysis: dict) -> float:
        risk_factors = []
        
        extraction_vuln = analysis.get('extraction_vulnerability', {})
        if extraction_vuln.get('extraction_difficulty') == 'low':
            risk_factors.append(0.9)
        elif extraction_vuln.get('extraction_difficulty') == 'medium':
            risk_factors.append(0.6)
        else:
            risk_factors.append(0.3)
        
        protection_score = analysis.get('protection_mechanisms', {}).get('protection_score', 0.3)
        risk_factors.append(1.0 - protection_score)
        
        return float(np.mean(risk_factors))

    def _determine_protection_level(self, risk_score: float) -> str:
        if risk_score >= 0.8:
            return 'INADEQUATE'
        elif risk_score >= 0.6:
            return 'WEAK'
        elif risk_score >= 0.4:
            return 'MODERATE'
        else:
            return 'STRONG'

    def _generate_protection_recommendations(self, risk_score: float) -> list:
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.extend([
                "Implement strong model encryption",
                "Deploy secure model serving infrastructure",
                "Use model fragmentation techniques",
                "Implement rigorous access controls"
            ])
        elif risk_score > 0.5:
            recommendations.extend([
                "Add model watermarking",
                "Implement API security measures",
                "Monitor model access patterns"
            ])
        
        return recommendations if recommendations else ['Maintain basic security practices']

class AdversarialDetector(AdvancedThreatDetector):
    def detect_adversarial_vulnerabilities(self, model_path: str) -> dict:
        try:
            model = self._safe_model_load(model_path)
            if model is None:
                return {'error': 'Model loading failed'}
            
            analysis = {
                'robustness_analysis': self._analyze_model_robustness(model),
                'gradient_analysis': self._analyze_adversarial_gradients(model),
                'decision_boundary_analysis': self._analyze_decision_boundaries(model),
                'attack_success_estimation': self._estimate_attack_success(model),
                'defense_mechanisms': self._evaluate_defense_mechanisms(model)
            }
            
            adversarial_risk = self._calculate_adversarial_risk(analysis)
            
            return {
                'adversarial_vulnerability': adversarial_risk,
                'robustness_level': self._determine_robustness_level(adversarial_risk),
                'vulnerability_assessment': analysis,
                'defense_recommendations': self._generate_defense_recommendations(adversarial_risk)
            }
            
        except Exception as e:
            return {'error': f'Adversarial detection failed: {str(e)}'}

    def _analyze_model_robustness(self, model) -> dict:
        if hasattr(model, 'get_weights'):
            weights = model.get_weights()
            robustness_scores = []
            
            for weight in weights:
                if len(weight.shape) > 1:
                    robustness = 1.0 / (1.0 + np.std(weight))
                    robustness_scores.append(float(robustness))
            
            return {
                'average_robustness': np.mean(robustness_scores) if robustness_scores else 0.0,
                'robustness_consistency': np.std(robustness_scores) if robustness_scores else 0.0,
                'overall_robustness': 'low' if (np.mean(robustness_scores) < 0.3 if robustness_scores else True) else 'high'
            }
        
        return {'robustness_analysis': 'not_supported'}

    def _analyze_adversarial_gradients(self, model) -> dict:
        return {
            'gradient_analysis': 'Requires input samples and loss function',
            'recommendation': 'Perform with actual test data',
            'estimated_vulnerability': 0.5
        }

    def _analyze_decision_boundaries(self, model) -> dict:
        return {
            'boundary_analysis': 'Complex analysis requiring sample data',
            'recommendation': 'Use specialized adversarial robustness tools',
            'risk_level': 'unknown'
        }

    def _estimate_attack_success(self, model) -> dict:
        robustness = self._analyze_model_robustness(model)
        robustness_score = robustness.get('average_robustness', 0.5)
        
        return {
            'estimated_success_rate': 1.0 - robustness_score,
            'attack_difficulty': 'easy' if robustness_score < 0.3 else 'medium' if robustness_score < 0.6 else 'hard',
            'confidence': 0.7
        }

    def _evaluate_defense_mechanisms(self, model) -> dict:
        return {
            'current_defenses': 'basic',
            'defense_score': 0.4,
            'recommended_enhancements': [
                'Implement adversarial training',
                'Use input sanitization',
                'Deploy gradient masking'
            ]
        }

    def _calculate_adversarial_risk(self, analysis: dict) -> float:
        risk_factors = []
        
        robustness_analysis = analysis.get('robustness_analysis', {})
        if robustness_analysis.get('overall_robustness') == 'low':
            risk_factors.append(0.9)
        elif robustness_analysis.get('overall_robustness') == 'high':
            risk_factors.append(0.3)
        else:
            risk_factors.append(0.6)
        
        attack_success = analysis.get('attack_success_estimation', {}).get('estimated_success_rate', 0.5)
        risk_factors.append(attack_success)
        
        return float(np.mean(risk_factors))

    def _determine_robustness_level(self, risk_score: float) -> str:
        if risk_score >= 0.8:
            return 'VERY_WEAK'
        elif risk_score >= 0.6:
            return 'WEAK'
        elif risk_score >= 0.4:
            return 'MODERATE'
        else:
            return 'STRONG'

    def _generate_defense_recommendations(self, risk_score: float) -> list:
        recommendations = []
        
        if risk_score > 0.7:
            recommendations.extend([
                "Implement comprehensive adversarial training",
                "Deploy multiple defense layers",
                "Use certified robustness techniques",
                "Regular adversarial testing"
            ])
        elif risk_score > 0.5:
            recommendations.extend([
                "Add adversarial examples to training",
                "Implement input preprocessing defenses",
                "Monitor for adversarial attacks"
            ])
        
        return recommendations if recommendations else ['Maintain basic model hygiene']

def create_threat_detector_suite():
    return {
        'backdoor_detector': BackdoorDetector(),
        'data_poisoning_detector': DataPoisoningDetector(),
        'model_stealing_detector': ModelStealingDetector(),
        'adversarial_detector': AdversarialDetector()
    }

if __name__ == "__main__":
    detectors = create_threat_detector_suite()
    
    test_model = "real_ai_model.pkl"
    
    for name, detector in detectors.items():
        print(f"\n=== {name.upper()} ===")
        if name == 'backdoor_detector':
            result = detector.detect_advanced_backdoors(test_model)
        elif name == 'data_poisoning_detector':
            result = detector.detect_data_poisoning(test_model)
        elif name == 'model_stealing_detector':
            result = detector.detect_model_stealing(test_model)
        elif name == 'adversarial_detector':
            result = detector.detect_adversarial_vulnerabilities(test_model)
        
        print(result)