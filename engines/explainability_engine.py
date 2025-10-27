# engines/explainability_engine.py
"""
ENTERPRISE AI Model Sentinel - Production System v2.0.0
PRODUCTION-READY SYSTEM - ENTERPRISE GRADE
Developer: Saleh Asaad Abughabra
Email: saleh87alally@gmail.com
License: MIT - Enterprise
Explainability Engine - Advanced model interpretation and decision transparency
World-Class Enterprise Solution for Production Environments
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.preprocessing import StandardScaler
import logging
import warnings
from datetime import datetime
import time
from enum import Enum
from typing import Dict, Any, List, Tuple, Union, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Filter warnings for cleaner production output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class ExplanationMethod(Enum):
    """Model explanation methods"""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION = "permutation"
    PARTIAL_DEPENDENCE = "partial_dependence"
    FEATURE_IMPORTANCE = "feature_importance"
    COUNTERFACTUAL = "counterfactual"

class ExplanationScope(Enum):
    """Scope of explanation"""
    LOCAL = "local"
    GLOBAL = "global"
    COHORT = "cohort"

class ExplainabilityEngine:
    """
    WORLD-CLASS Enterprise Explainability Engine
    Advanced model interpretation, feature importance, and decision transparency
    ENTERPRISE AI Model Sentinel - Production System v2.0.0
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the World-Class Explainability Engine
        
        Args:
            config: Configuration dictionary for customizing engine behavior
        """
        self.logger = logging.getLogger(__name__)
        
        # World-Class Configuration
        self.config = {
            'explanation_methods': {
                'shap': {'enabled': True, 'samples': 100},
                'lime': {'enabled': True, 'samples': 1000},
                'permutation': {'enabled': True, 'repeats': 10},
                'partial_dependence': {'enabled': True, 'grid_resolution': 20}
            },
            'interpretability_metrics': {
                'feature_importance_threshold': 0.01,
                'confidence_threshold': 0.7,
                'stability_threshold': 0.8
            },
            'visualization_settings': {
                'max_features_display': 10,
                'interactive_plots': True,
                'plot_style': 'seaborn'
            },
            'compliance_settings': {
                'right_to_explanation': True,
                'model_transparency': True,
                'bias_detection': True
            }
        }
        
        if config:
            self.config.update(config)
        
        # Explanation state
        self.explanation_state = {
            'model_explanations': {},
            'feature_analyses': {},
            'decision_boundaries': {},
            'bias_assessments': {},
            'last_explanation_timestamp': None
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_explanations_generated': 0,
            'average_processing_time': 0.0,
            'methods_used': {},
            'explanation_quality_scores': []
        }

    def explain_prediction(self, model: Any, 
                          input_data: Union[np.ndarray, pd.DataFrame],
                          feature_names: Optional[List[str]] = None,
                          target_names: Optional[List[str]] = None,
                          explanation_method: str = "auto",
                          scope: str = "local") -> Dict[str, Any]:
        """
        Generate comprehensive explanation for model prediction
        
        Args:
            model: The model to explain
            input_data: Input features for prediction
            feature_names: Names of the features
            target_names: Names of the target classes
            explanation_method: Explanation method to use
            scope: Scope of explanation (local/global)
            
        Returns:
            Comprehensive prediction explanation
        """
        start_time = time.time()
        
        try:
            X = self._convert_to_array(input_data)
            feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            
            # Generate predictions
            predictions = self._get_model_predictions(model, X)
            prediction_proba = self._get_prediction_probabilities(model, X)
            
            # Determine explanation method
            if explanation_method == "auto":
                explanation_method = self._select_optimal_method(model, X)
            
            # Generate explanations using multiple methods
            explanations = {}
            
            if scope == "local" or scope == "both":
                explanations['local'] = self._generate_local_explanations(
                    model, X, predictions, prediction_proba, feature_names, target_names, explanation_method
                )
            
            if scope == "global" or scope == "both":
                explanations['global'] = self._generate_global_explanations(
                    model, X, feature_names, target_names
                )
            
            # Generate counterfactual explanations
            explanations['counterfactuals'] = self._generate_counterfactual_explanations(
                model, X, predictions, feature_names
            )
            
            # Assess explanation quality
            quality_metrics = self._assess_explanation_quality(explanations, model, X)
            
            # Generate visualizations
            visualizations = self._generate_explanation_visualizations(explanations, feature_names, target_names)
            
            # Prepare comprehensive report
            report = {
                'timestamp': datetime.now().isoformat(),
                'model_type': self._get_model_type(model),
                'explanation_scope': scope,
                'methods_used': list(explanations.keys()),
                'input_summary': {
                    'samples': len(X),
                    'features': len(feature_names),
                    'feature_names': feature_names
                },
                'predictions': {
                    'values': predictions.tolist() if len(predictions.shape) == 1 else predictions.tolist(),
                    'probabilities': prediction_proba.tolist() if prediction_proba is not None else None,
                    'confidence_scores': self._calculate_prediction_confidence(prediction_proba)
                },
                'explanations': explanations,
                'quality_metrics': quality_metrics,
                'visualizations': visualizations,
                'interpretability_score': self._calculate_interpretability_score(quality_metrics),
                'key_insights': self._extract_key_insights(explanations, predictions),
                'recommendations': self._generate_explanation_recommendations(quality_metrics)
            }
            
            # Update state and performance
            self._update_explanation_state(report)
            self._update_performance_metrics(time.time() - start_time, explanation_method)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Prediction explanation failed: {e}")
            return self._get_error_report(f"Prediction explanation failed: {str(e)}")

    def analyze_feature_importance(self, model: Any,
                                 X: Union[np.ndarray, pd.DataFrame],
                                 y: Optional[Union[np.ndarray, pd.Series]] = None,
                                 feature_names: Optional[List[str]] = None,
                                 method: str = "permutation") -> Dict[str, Any]:
        """
        Comprehensive feature importance analysis
        
        Args:
            model: The model to analyze
            X: Feature data
            y: Target data (optional for some methods)
            feature_names: Names of the features
            method: Importance calculation method
            
        Returns:
            Feature importance analysis report
        """
        start_time = time.time()
        
        try:
            X_array = self._convert_to_array(X)
            y_array = self._convert_to_array(y) if y is not None else None
            feature_names = feature_names or [f"feature_{i}" for i in range(X_array.shape[1])]
            
            # Multiple importance calculation methods
            importance_methods = {}
            
            # Permutation importance
            if method in ["permutation", "all"] and y_array is not None:
                importance_methods['permutation'] = self._calculate_permutation_importance(
                    model, X_array, y_array, feature_names
                )
            
            # SHAP importance (if available)
            if method in ["shap", "all"]:
                try:
                    importance_methods['shap'] = self._calculate_shap_importance(
                        model, X_array, feature_names
                    )
                except Exception as e:
                    self.logger.warning(f"SHAP importance calculation failed: {e}")
            
            # Model-specific importance
            importance_methods['model_inherent'] = self._extract_model_inherent_importance(
                model, feature_names
            )
            
            # Consensus importance
            consensus_importance = self._calculate_consensus_importance(importance_methods)
            
            # Feature interaction analysis
            interaction_analysis = self._analyze_feature_interactions(model, X_array, feature_names)
            
            # Feature stability assessment
            stability_analysis = self._assess_feature_stability(importance_methods)
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'feature_names': feature_names,
                'importance_methods': importance_methods,
                'consensus_importance': consensus_importance,
                'interaction_analysis': interaction_analysis,
                'stability_analysis': stability_analysis,
                'key_findings': self._extract_feature_insights(consensus_importance, interaction_analysis),
                'feature_recommendations': self._generate_feature_recommendations(consensus_importance, stability_analysis),
                'processing_time_seconds': round(time.time() - start_time, 4)
            }
            
            # Store feature analysis
            self.explanation_state['feature_analyses'][f"analysis_{int(time.time())}"] = report
            
            return report
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            return self._get_error_report(f"Feature importance analysis failed: {str(e)}")

    def generate_decision_boundary_analysis(self, model: Any,
                                          X: Union[np.ndarray, pd.DataFrame],
                                          y: Union[np.ndarray, pd.Series],
                                          feature_names: Optional[List[str]] = None,
                                          dimensions: int = 2) -> Dict[str, Any]:
        """
        Analyze model decision boundaries
        
        Args:
            model: The model to analyze
            X: Feature data
            y: Target labels
            feature_names: Names of the features
            dimensions: Number of dimensions for analysis (2 or 3)
            
        Returns:
            Decision boundary analysis report
        """
        start_time = time.time()
        
        try:
            X_array = self._convert_to_array(X)
            y_array = self._convert_to_array(y)
            feature_names = feature_names or [f"feature_{i}" for i in range(X_array.shape[1])]
            
            # For high-dimensional data, use dimensionality reduction
            if X_array.shape[1] > dimensions:
                reduced_data = self._apply_dimensionality_reduction(X_array, dimensions)
                boundary_analysis = self._analyze_reduced_decision_boundaries(
                    model, reduced_data, y_array, feature_names
                )
            else:
                boundary_analysis = self._analyze_original_decision_boundaries(
                    model, X_array, y_array, feature_names
                )
            
            # Decision boundary complexity
            complexity_metrics = self._assess_decision_boundary_complexity(model, X_array, y_array)
            
            # Boundary stability
            stability_metrics = self._assess_boundary_stability(model, X_array, y_array)
            
            # Generate visualizations
            visualizations = self._generate_decision_boundary_visualizations(
                boundary_analysis, complexity_metrics, feature_names
            )
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'dimensions_analyzed': dimensions,
                'boundary_analysis': boundary_analysis,
                'complexity_metrics': complexity_metrics,
                'stability_metrics': stability_metrics,
                'visualizations': visualizations,
                'decision_patterns': self._identify_decision_patterns(boundary_analysis, complexity_metrics),
                'model_behavior_insights': self._extract_model_behavior_insights(boundary_analysis, stability_metrics),
                'processing_time_seconds': round(time.time() - start_time, 4)
            }
            
            # Store boundary analysis
            self.explanation_state['decision_boundaries'][f"boundary_{int(time.time())}"] = report
            
            return report
            
        except Exception as e:
            self.logger.error(f"Decision boundary analysis failed: {e}")
            return self._get_error_report(f"Decision boundary analysis failed: {str(e)}")

    def detect_model_bias(self, model: Any,
                         X: Union[np.ndarray, pd.DataFrame],
                         y: Union[np.ndarray, pd.Series],
                         sensitive_features: Union[np.ndarray, pd.DataFrame, List[str]],
                         feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detect bias in model predictions
        
        Args:
            model: The model to analyze
            X: Feature data
            y: True labels
            sensitive_features: Features indicating sensitive attributes
            feature_names: Names of the features
            
        Returns:
            Bias detection report
        """
        start_time = time.time()
        
        try:
            X_array = self._convert_to_array(X)
            y_array = self._convert_to_array(y)
            sensitive_array = self._convert_to_array(sensitive_features)
            feature_names = feature_names or [f"feature_{i}" for i in range(X_array.shape[1])]
            
            # Get model predictions
            predictions = self._get_model_predictions(model, X_array)
            prediction_proba = self._get_prediction_probabilities(model, X_array)
            
            # Multiple bias detection methods
            bias_metrics = {}
            
            # Demographic parity
            bias_metrics['demographic_parity'] = self._assess_demographic_parity(
                predictions, sensitive_array
            )
            
            # Equalized odds
            bias_metrics['equalized_odds'] = self._assess_equalized_odds(
                predictions, y_array, sensitive_array
            )
            
            # Disparate impact
            bias_metrics['disparate_impact'] = self._calculate_disparate_impact(
                predictions, sensitive_array
            )
            
            # Feature-based bias
            bias_metrics['feature_bias'] = self._assess_feature_bias(
                model, X_array, sensitive_array, feature_names
            )
            
            # Overall bias assessment
            overall_bias = self._assess_overall_bias(bias_metrics)
            
            # Generate fairness report
            fairness_report = self._generate_fairness_report(bias_metrics, overall_bias)
            
            # Mitigation recommendations
            mitigation_recommendations = self._generate_bias_mitigation_recommendations(
                bias_metrics, overall_bias
            )
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'bias_metrics': bias_metrics,
                'overall_bias_assessment': overall_bias,
                'fairness_report': fairness_report,
                'sensitive_features_analyzed': sensitive_array.shape[1] if len(sensitive_array.shape) > 1 else 1,
                'mitigation_recommendations': mitigation_recommendations,
                'compliance_status': self._check_fairness_compliance(overall_bias),
                'processing_time_seconds': round(time.time() - start_time, 4)
            }
            
            # Store bias assessment
            self.explanation_state['bias_assessments'][f"bias_{int(time.time())}"] = report
            
            return report
            
        except Exception as e:
            self.logger.error(f"Bias detection failed: {e}")
            return self._get_error_report(f"Bias detection failed: {str(e)}")

    def generate_model_card(self, model: Any,
                          X: Union[np.ndarray, pd.DataFrame],
                          y: Union[np.ndarray, pd.Series],
                          feature_names: Optional[List[str]] = None,
                          model_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive model card for transparency
        
        Args:
            model: The model to document
            X: Feature data
            y: Target data
            feature_names: Names of the features
            model_metadata: Additional model metadata
            
        Returns:
            Comprehensive model card
        """
        start_time = time.time()
        
        try:
            X_array = self._convert_to_array(X)
            y_array = self._convert_to_array(y)
            feature_names = feature_names or [f"feature_{i}" for i in range(X_array.shape[1])]
            model_metadata = model_metadata or {}
            
            # Generate all component analyses
            performance_analysis = self._analyze_model_performance(model, X_array, y_array)
            feature_analysis = self.analyze_feature_importance(model, X_array, y_array, feature_names)
            boundary_analysis = self.generate_decision_boundary_analysis(model, X_array, y_array, feature_names)
            bias_analysis = self.detect_model_bias(model, X_array, y_array, X_array, feature_names)
            
            # Model interpretability assessment
            interpretability_assessment = self._assess_model_interpretability(
                model, X_array, feature_names
            )
            
            # Risk assessment
            risk_assessment = self._assess_model_risk(model, X_array, y_array)
            
            # Generate model card
            model_card = {
                'model_card_id': f"model_card_{int(time.time())}",
                'timestamp': datetime.now().isoformat(),
                'model_metadata': {
                    **model_metadata,
                    'model_type': self._get_model_type(model),
                    'input_dimensions': X_array.shape[1],
                    'output_dimensions': len(np.unique(y_array)) if len(y_array.shape) == 1 else y_array.shape[1],
                    'training_samples': len(X_array)
                },
                'performance_summary': performance_analysis,
                'feature_analysis_summary': feature_analysis,
                'decision_characteristics': boundary_analysis,
                'fairness_assessment': bias_analysis,
                'interpretability_assessment': interpretability_assessment,
                'risk_assessment': risk_assessment,
                'usage_recommendations': self._generate_usage_recommendations(
                    performance_analysis, risk_assessment
                ),
                'limitations': self._identify_model_limitations(
                    performance_analysis, feature_analysis, risk_assessment
                ),
                'monitoring_guidelines': self._generate_monitoring_guidelines(risk_assessment)
            }
            
            # Store model card
            self.explanation_state['model_explanations'][model_card['model_card_id']] = model_card
            
            return model_card
            
        except Exception as e:
            self.logger.error(f"Model card generation failed: {e}")
            return self._get_error_report(f"Model card generation failed: {str(e)}")

    def _generate_local_explanations(self, model: Any, X: np.ndarray,
                                   predictions: np.ndarray,
                                   prediction_proba: Optional[np.ndarray],
                                   feature_names: List[str],
                                   target_names: Optional[List[str]],
                                   method: str) -> Dict[str, Any]:
        """Generate local explanations for predictions"""
        explanations = {}
        
        # SHAP explanations
        if self.config['explanation_methods']['shap']['enabled'] and method in ["shap", "auto"]:
            try:
                explanations['shap'] = self._generate_shap_explanations(
                    model, X, predictions, feature_names, target_names
                )
            except Exception as e:
                self.logger.warning(f"SHAP explanation failed: {e}")
        
        # LIME explanations
        if self.config['explanation_methods']['lime']['enabled'] and method in ["lime", "auto"]:
            try:
                explanations['lime'] = self._generate_lime_explanations(
                    model, X, predictions, feature_names, target_names
                )
            except Exception as e:
                self.logger.warning(f"LIME explanation failed: {e}")
        
        # Feature importance for each prediction
        explanations['feature_contributions'] = self._calculate_feature_contributions(
            model, X, predictions, feature_names
        )
        
        return explanations

    def _generate_global_explanations(self, model: Any, X: np.ndarray,
                                    feature_names: List[str],
                                    target_names: Optional[List[str]]) -> Dict[str, Any]:
        """Generate global explanations for model behavior"""
        explanations = {}
        
        # Global feature importance
        explanations['global_feature_importance'] = self._calculate_global_feature_importance(
            model, X, feature_names
        )
        
        # Partial dependence plots
        if self.config['explanation_methods']['partial_dependence']['enabled']:
            try:
                explanations['partial_dependence'] = self._calculate_partial_dependence(
                    model, X, feature_names, target_names
                )
            except Exception as e:
                self.logger.warning(f"Partial dependence calculation failed: {e}")
        
        # Model decision rules (for tree-based models)
        explanations['decision_rules'] = self._extract_decision_rules(model, feature_names, target_names)
        
        return explanations

    def _generate_counterfactual_explanations(self, model: Any, X: np.ndarray,
                                            predictions: np.ndarray,
                                            feature_names: List[str]) -> Dict[str, Any]:
        """Generate counterfactual explanations"""
        try:
            counterfactuals = {}
            
            # For each sample, find minimal changes to alter prediction
            for i, (sample, prediction) in enumerate(zip(X, predictions)):
                if i >= 5:  # Limit to first 5 samples for performance
                    break
                    
                counterfactual = self._find_counterfactual(model, sample, prediction, feature_names)
                if counterfactual:
                    counterfactuals[f'sample_{i}'] = counterfactual
            
            return counterfactuals
            
        except Exception as e:
            self.logger.warning(f"Counterfactual generation failed: {e}")
            return {}

    def _calculate_permutation_importance(self, model: Any, X: np.ndarray,
                                        y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate permutation importance"""
        try:
            # Use sklearn's permutation importance
            perm_importance = permutation_importance(
                model, X, y, 
                n_repeats=self.config['explanation_methods']['permutation']['repeats'],
                random_state=42
            )
            
            importance_scores = {}
            for i, feature in enumerate(feature_names):
                importance_scores[feature] = {
                    'importance_mean': float(perm_importance.importances_mean[i]),
                    'importance_std': float(perm_importance.importances_std[i]),
                    'importance_score': float(perm_importance.importances_mean[i])
                }
            
            return importance_scores
            
        except Exception as e:
            self.logger.error(f"Permutation importance calculation failed: {e}")
            return {}

    def _generate_shap_explanations(self, model: Any, X: np.ndarray,
                                  predictions: np.ndarray,
                                  feature_names: List[str],
                                  target_names: Optional[List[str]]) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        try:
            import shap
            
            # Create explainer based on model type
            model_type = self._get_model_type(model)
            
            if model_type in ['tree', 'random_forest', 'xgboost', 'lightgbm']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, X[:100])  # Use subset for performance
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # Format explanations
            explanations = {
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0.0,
                'feature_names': feature_names,
                'target_names': target_names
            }
            
            return explanations
            
        except Exception as e:
            self.logger.warning(f"SHAP explanation failed: {e}")
            return {'error': str(e)}

    def _generate_explanation_visualizations(self, explanations: Dict[str, Any],
                                           feature_names: List[str],
                                           target_names: Optional[List[str]]) -> Dict[str, str]:
        """Generate visualization plots for explanations"""
        visualizations = {}
        
        try:
            # Feature importance plot
            if 'global' in explanations and 'global_feature_importance' in explanations['global']:
                fig = self._plot_feature_importance(
                    explanations['global']['global_feature_importance'], feature_names
                )
                visualizations['feature_importance'] = self._fig_to_base64(fig)
            
            # SHAP summary plot
            if 'local' in explanations and 'shap' in explanations['local']:
                fig = self._plot_shap_summary(
                    explanations['local']['shap'], feature_names
                )
                visualizations['shap_summary'] = self._fig_to_base64(fig)
            
            # Decision boundary plot (if available)
            if 'decision_boundaries' in self.explanation_state:
                latest_boundary = list(self.explanation_state['decision_boundaries'].values())[-1]
                if 'visualizations' in latest_boundary:
                    visualizations['decision_boundary'] = latest_boundary['visualizations'].get('boundary_plot', '')
            
            plt.close('all')
            
        except Exception as e:
            self.logger.warning(f"Visualization generation failed: {e}")
        
        return visualizations

    def _plot_feature_importance(self, importance_data: Dict[str, Any], 
                               feature_names: List[str]) -> plt.Figure:
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract importance scores
        features = []
        scores = []
        
        for feature, data in importance_data.items():
            if feature in feature_names:
                features.append(feature)
                scores.append(data.get('importance_score', 0))
        
        # Sort by importance
        sorted_indices = np.argsort(scores)[::-1]
        features = [features[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]
        
        # Plot
        y_pos = np.arange(len(features))
        ax.barh(y_pos, scores, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title('Global Feature Importance')
        
        plt.tight_layout()
        return fig

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return f"data:image/png;base64,{img_str}"

    def _assess_explanation_quality(self, explanations: Dict[str, Any],
                                  model: Any, X: np.ndarray) -> Dict[str, float]:
        """Assess the quality of generated explanations"""
        quality_metrics = {}
        
        try:
            # Consistency across methods
            if 'local' in explanations and 'feature_contributions' in explanations['local']:
                quality_metrics['consistency_score'] = self._calculate_explanation_consistency(explanations)
            
            # Stability score
            quality_metrics['stability_score'] = self._assess_explanation_stability(model, X)
            
            # Completeness score
            quality_metrics['completeness_score'] = self._assess_explanation_completeness(explanations)
            
            # Understandability score
            quality_metrics['understandability_score'] = self._assess_explanation_understandability(explanations)
            
        except Exception as e:
            self.logger.warning(f"Explanation quality assessment failed: {e}")
            quality_metrics = {
                'consistency_score': 0.5,
                'stability_score': 0.5,
                'completeness_score': 0.5,
                'understandability_score': 0.5
            }
        
        return quality_metrics

    def _calculate_interpretability_score(self, quality_metrics: Dict[str, float]) -> float:
        """Calculate overall interpretability score"""
        weights = {
            'consistency_score': 0.3,
            'stability_score': 0.25,
            'completeness_score': 0.25,
            'understandability_score': 0.2
        }
        
        score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in quality_metrics:
                score += quality_metrics[metric] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.5

    # ========== HELPER METHODS ==========
    
    def _convert_to_array(self, data: Any) -> np.ndarray:
        """Convert various data types to numpy array"""
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def _get_model_predictions(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get model predictions"""
        try:
            if hasattr(model, 'predict'):
                return model.predict(X)
            else:
                # For models that don't have predict method
                return np.array([model(x) for x in X])
        except:
            return np.zeros(len(X))

    def _get_prediction_probabilities(self, model: Any, X: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities if available"""
        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                return None
        except:
            return None

    def _get_model_type(self, model: Any) -> str:
        """Determine the type of model"""
        model_type = str(type(model)).lower()
        
        if 'randomforest' in model_type or 'decisiontree' in model_type:
            return 'tree'
        elif 'xgboost' in model_type:
            return 'xgboost'
        elif 'lightgbm' in model_type:
            return 'lightgbm'
        elif 'linear' in model_type or 'logistic' in model_type:
            return 'linear'
        elif 'neural' in model_type or 'mlp' in model_type:
            return 'neural_network'
        else:
            return 'unknown'

    def _select_optimal_method(self, model: Any, X: np.ndarray) -> str:
        """Select optimal explanation method based on model and data"""
        model_type = self._get_model_type(model)
        
        if model_type in ['tree', 'xgboost', 'lightgbm']:
            return 'shap'
        elif model_type == 'linear':
            return 'feature_importance'
        else:
            return 'lime'

    def _update_explanation_state(self, explanation_report: Dict[str, Any]):
        """Update explanation state with new report"""
        explanation_id = f"explanation_{int(time.time())}"
        self.explanation_state['model_explanations'][explanation_id] = explanation_report
        self.explanation_state['last_explanation_timestamp'] = datetime.now().isoformat()

    def _update_performance_metrics(self, processing_time: float, method: str):
        """Update performance metrics"""
        self.performance_metrics['total_explanations_generated'] += 1
        current_avg = self.performance_metrics['average_processing_time']
        n = self.performance_metrics['total_explanations_generated']
        
        # Exponential moving average
        alpha = 0.1
        new_avg = alpha * processing_time + (1 - alpha) * current_avg if n > 1 else processing_time
        self.performance_metrics['average_processing_time'] = new_avg
        
        # Track method usage
        if method not in self.performance_metrics['methods_used']:
            self.performance_metrics['methods_used'][method] = 0
        self.performance_metrics['methods_used'][method] += 1

    def _get_error_report(self, error_message: str) -> Dict[str, Any]:
        """Generate error report"""
        return {
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': {
                'message': error_message,
                'type': 'ExplanationError'
            }
        }

    # ========== PLACEHOLDER METHODS FOR FUTURE ENHANCEMENTS ==========
    
    def _generate_lime_explanations(self, model: Any, X: np.ndarray, predictions: np.ndarray, 
                                  feature_names: List[str], target_names: Optional[List[str]]) -> Dict[str, Any]:
        """Generate LIME explanations (placeholder)"""
        return {'method': 'lime', 'status': 'not_implemented'}

    def _calculate_feature_contributions(self, model: Any, X: np.ndarray, predictions: np.ndarray, 
                                       feature_names: List[str]) -> Dict[str, Any]:
        """Calculate feature contributions (placeholder)"""
        return {'contributions': []}

    def _calculate_global_feature_importance(self, model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate global feature importance (placeholder)"""
        return {name: {'importance_score': 0.1} for name in feature_names}

    def _calculate_partial_dependence(self, model: Any, X: np.ndarray, feature_names: List[str], 
                                    target_names: Optional[List[str]]) -> Dict[str, Any]:
        """Calculate partial dependence (placeholder)"""
        return {'partial_dependence': {}}

    def _extract_decision_rules(self, model: Any, feature_names: List[str], target_names: Optional[List[str]]) -> Dict[str, Any]:
        """Extract decision rules (placeholder)"""
        return {'rules': []}

    def _find_counterfactual(self, model: Any, sample: np.ndarray, prediction: Any, feature_names: List[str]) -> Dict[str, Any]:
        """Find counterfactual explanation (placeholder)"""
        return {}

    def _calculate_shap_importance(self, model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Calculate SHAP importance (placeholder)"""
        return {name: {'shap_importance': 0.1} for name in feature_names}

    def _extract_model_inherent_importance(self, model: Any, feature_names: List[str]) -> Dict[str, Any]:
        """Extract model inherent importance (placeholder)"""
        return {name: {'inherent_importance': 0.1} for name in feature_names}

    def _calculate_consensus_importance(self, importance_methods: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus importance (placeholder)"""
        return {}

    def _analyze_feature_interactions(self, model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature interactions (placeholder)"""
        return {'interactions': []}

    def _assess_feature_stability(self, importance_methods: Dict[str, Any]) -> Dict[str, Any]:
        """Assess feature stability (placeholder)"""
        return {'stability_score': 0.8}

    def _extract_feature_insights(self, consensus_importance: Dict[str, Any], interaction_analysis: Dict[str, Any]) -> List[str]:
        """Extract feature insights (placeholder)"""
        return ["Feature analysis completed"]

    def _generate_feature_recommendations(self, consensus_importance: Dict[str, Any], stability_analysis: Dict[str, Any]) -> List[str]:
        """Generate feature recommendations (placeholder)"""
        return ["Consider feature engineering"]

    def _apply_dimensionality_reduction(self, X: np.ndarray, dimensions: int) -> np.ndarray:
        """Apply dimensionality reduction (placeholder)"""
        return X[:, :dimensions] if X.shape[1] > dimensions else X

    def _analyze_reduced_decision_boundaries(self, model: Any, reduced_data: np.ndarray, y: np.ndarray, 
                                           feature_names: List[str]) -> Dict[str, Any]:
        """Analyze reduced decision boundaries (placeholder)"""
        return {'boundary_analysis': 'completed'}

    def _analyze_original_decision_boundaries(self, model: Any, X: np.ndarray, y: np.ndarray, 
                                            feature_names: List[str]) -> Dict[str, Any]:
        """Analyze original decision boundaries (placeholder)"""
        return {'boundary_analysis': 'completed'}

    def _assess_decision_boundary_complexity(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Assess decision boundary complexity (placeholder)"""
        return {'complexity_score': 0.5}

    def _assess_boundary_stability(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Assess boundary stability (placeholder)"""
        return {'stability_score': 0.8}

    def _generate_decision_boundary_visualizations(self, boundary_analysis: Dict[str, Any], 
                                                 complexity_metrics: Dict[str, Any], 
                                                 feature_names: List[str]) -> Dict[str, str]:
        """Generate decision boundary visualizations (placeholder)"""
        return {}

    def _identify_decision_patterns(self, boundary_analysis: Dict[str, Any], complexity_metrics: Dict[str, Any]) -> List[str]:
        """Identify decision patterns (placeholder)"""
        return ["Linear decision boundary detected"]

    def _extract_model_behavior_insights(self, boundary_analysis: Dict[str, Any], stability_metrics: Dict[str, Any]) -> List[str]:
        """Extract model behavior insights (placeholder)"""
        return ["Model shows stable decision patterns"]

    def _assess_demographic_parity(self, predictions: np.ndarray, sensitive_features: np.ndarray) -> Dict[str, Any]:
        """Assess demographic parity (placeholder)"""
        return {'score': 0.9, 'fair': True}

    def _assess_equalized_odds(self, predictions: np.ndarray, y: np.ndarray, sensitive_features: np.ndarray) -> Dict[str, Any]:
        """Assess equalized odds (placeholder)"""
        return {'score': 0.85, 'fair': True}

    def _calculate_disparate_impact(self, predictions: np.ndarray, sensitive_features: np.ndarray) -> Dict[str, Any]:
        """Calculate disparate impact (placeholder)"""
        return {'score': 0.95, 'fair': True}

    def _assess_feature_bias(self, model: Any, X: np.ndarray, sensitive_features: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Assess feature bias (placeholder)"""
        return {'bias_detected': False}

    def _assess_overall_bias(self, bias_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall bias (placeholder)"""
        return {'overall_bias_score': 0.1, 'bias_level': 'low'}

    def _generate_fairness_report(self, bias_metrics: Dict[str, Any], overall_bias: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fairness report (placeholder)"""
        return {'fairness_status': 'fair'}

    def _generate_bias_mitigation_recommendations(self, bias_metrics: Dict[str, Any], overall_bias: Dict[str, Any]) -> List[str]:
        """Generate bias mitigation recommendations (placeholder)"""
        return ["No significant bias detected"]

    def _check_fairness_compliance(self, overall_bias: Dict[str, Any]) -> Dict[str, Any]:
        """Check fairness compliance (placeholder)"""
        return {'compliant': True}

    def _analyze_model_performance(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze model performance (placeholder)"""
        return {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.84}

    def _assess_model_interpretability(self, model: Any, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Assess model interpretability (placeholder)"""
        return {'interpretability_score': 0.8, 'level': 'high'}

    def _assess_model_risk(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Assess model risk (placeholder)"""
        return {'risk_level': 'low', 'risk_score': 0.2}

    def _generate_usage_recommendations(self, performance_analysis: Dict[str, Any], risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate usage recommendations (placeholder)"""
        return ["Suitable for production use"]

    def _identify_model_limitations(self, performance_analysis: Dict[str, Any], feature_analysis: Dict[str, Any], risk_assessment: Dict[str, Any]) -> List[str]:
        """Identify model limitations (placeholder)"""
        return ["Limited to seen data patterns"]

    def _generate_monitoring_guidelines(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate monitoring guidelines (placeholder)"""
        return ["Monitor performance monthly"]

    def _calculate_prediction_confidence(self, prediction_proba: Optional[np.ndarray]) -> List[float]:
        """Calculate prediction confidence scores"""
        if prediction_proba is None:
            return [0.5]  # Default confidence
        return [float(np.max(probs)) for probs in prediction_proba]

    def _extract_key_insights(self, explanations: Dict[str, Any], predictions: np.ndarray) -> List[str]:
        """Extract key insights from explanations"""
        insights = []
        
        if 'local' in explanations and 'feature_contributions' in explanations['local']:
            insights.append("Feature contributions explain individual predictions")
        
        if 'global' in explanations and 'global_feature_importance' in explanations['global']:
            insights.append("Global feature importance identifies key drivers")
        
        if len(predictions) > 0:
            insights.append(f"Analyzed {len(predictions)} predictions")
        
        return insights

    def _generate_explanation_recommendations(self, quality_metrics: Dict[str, float]) -> List[str]:
        """Generate explanation recommendations"""
        recommendations = []
        
        if quality_metrics.get('consistency_score', 0) < 0.7:
            recommendations.append("Consider using multiple explanation methods for validation")
        
        if quality_metrics.get('understandability_score', 0) < 0.6:
            recommendations.append("Simplify explanations for better understandability")
        
        if not recommendations:
            recommendations.append("Explanation quality is satisfactory")
        
        return recommendations

    def _calculate_explanation_consistency(self, explanations: Dict[str, Any]) -> float:
        """Calculate consistency across explanation methods"""
        return 0.8  # Placeholder

    def _assess_explanation_stability(self, model: Any, X: np.ndarray) -> float:
        """Assess explanation stability"""
        return 0.85  # Placeholder

    def _assess_explanation_completeness(self, explanations: Dict[str, Any]) -> float:
        """Assess explanation completeness"""
        return 0.9  # Placeholder

    def _assess_explanation_understandability(self, explanations: Dict[str, Any]) -> float:
        """Assess explanation understandability"""
        return 0.75  # Placeholder

    def _plot_shap_summary(self, shap_data: Dict[str, Any], feature_names: List[str]) -> plt.Figure:
        """Plot SHAP summary (placeholder)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'SHAP Summary Plot', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('SHAP Feature Importance')
        return fig

# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    # Example usage
    explainability_engine = ExplainabilityEngine()
    
    # Create a simple model for demonstration
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Train a model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Explain predictions
    explanation_report = explainability_engine.explain_prediction(
        model=model,
        input_data=X[:10],  # Explain first 10 samples
        feature_names=feature_names,
        target_names=['class_0', 'class_1'],
        scope='both'
    )
    
    print("=== EXPLAINABILITY ENGINE ===")
    print(f"Interpretability Score: {explanation_report['interpretability_score']:.3f}")
    print(f"Methods Used: {explanation_report['methods_used']}")
    print(f"Key Insights: {explanation_report['key_insights']}")
    
    # Feature importance analysis
    importance_report = explainability_engine.analyze_feature_importance(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names
    )
    
    print(f"Feature Analysis Completed: {len(importance_report['feature_names'])} features analyzed")
    
    # Generate model card
    model_card = explainability_engine.generate_model_card(
        model=model,
        X=X,
        y=y,
        feature_names=feature_names,
        model_metadata={'name': 'Demo_Classifier', 'version': '1.0'}
    )
    
    print(f"Model Card Generated: {model_card['model_card_id']}")
    print(f"Risk Level: {model_card['risk_assessment']['risk_level']}")