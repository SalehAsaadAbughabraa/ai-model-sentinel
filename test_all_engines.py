"""
ENTERPRISE AI Model Sentinel - Comprehensive Engine Integration Test
PRODUCTION-READY SYSTEM - ENTERPRISE GRADE
Developer: Saleh Asaad Abughabra
Email: saleh87alally@gmail.com
License: MIT - Enterprise
Integrated Test for All System Engines
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json

# Add engines directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'engines'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('engine_integration_test.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class EngineIntegrationTester:
    """
    Comprehensive integration tester for all AI Model Sentinel engines
    """
    
    def __init__(self):
        self.results = {}
        self.engines = {}
        self.test_data = self._generate_test_data()
    
    def _generate_test_data(self):
        """Generate comprehensive test data for all engines"""
        np.random.seed(42)
        
        # Generate sample dataset
        n_samples = 1000
        n_features = 10
        
        X = np.random.normal(0, 1, (n_samples, n_features))
        y = np.random.randint(0, 2, n_samples)
        
        # Create some realistic patterns
        X[:, 0] = X[:, 0] * 2 + y * 1.5  # Feature 0 correlated with target
        X[:, 1] = X[:, 1] * 0.5  # Low variance feature
        X[:, 2] = np.where(X[:, 2] > 1, np.nan, X[:, 2])  # Some missing values
        
        # Create DataFrame with feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        # Create test model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'model': model,
            'dataframe': df,
            'feature_names': feature_names,
            'target_names': ['class_0', 'class_1']
        }
    
    def load_engines(self):
        """Load all available engines"""
        logger.info("🔄 Loading all engines...")
        
        engines_to_load = [
            ('data_quality_engine', 'DataQualityEngine'),
            ('model_monitoring_engine', 'ModelMonitoringEngine'), 
            ('security_engine', 'SecurityEngine'),
            ('explainability_engine', 'ExplainabilityEngine')
        ]
        
        for module_name, class_name in engines_to_load:
            try:
                module = __import__(module_name)
                engine_class = getattr(module, class_name)
                self.engines[module_name] = engine_class()
                logger.info(f"✅ Loaded {module_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load {module_name}: {e}")
                self.engines[module_name] = None
    
    def test_data_quality_engine(self):
        """Test Data Quality Engine"""
        logger.info("🧪 Testing Data Quality Engine...")
        
        if not self.engines.get('data_quality_engine'):
            return {'status': 'error', 'message': 'Engine not loaded'}
        
        try:
            engine = self.engines['data_quality_engine']
            results = {}
            
            # Test 1: Basic data integrity
            integrity_report = engine.analyze_data_integrity(self.test_data['X_train'])
            results['integrity_analysis'] = {
                'status': 'success',
                'metrics_found': list(integrity_report.keys()),
                'sample_metrics': {k: v for k, v in list(integrity_report.items())[:3]}
            }
            
            # Test 2: Noise sensitivity
            noise_report = engine.analyze_noise_sensitivity(self.test_data['X_train'])
            results['noise_analysis'] = {
                'status': 'success', 
                'metrics_found': list(noise_report.keys())
            }
            
            # Test 3: DataFrame analysis
            df_report = engine.analyze_data_integrity(self.test_data['dataframe'])
            results['dataframe_analysis'] = {
                'status': 'success',
                'dataframe_shape': self.test_data['dataframe'].shape
            }
            
            # Test 4: Quality report
            quality_report = engine.generate_quality_report(self.test_data['X_train'])
            results['quality_report'] = {
                'status': 'success',
                'overall_score': quality_report.get('overall_quality_score', 0),
                'report_keys': list(quality_report.keys())
            }
            
            logger.info("✅ Data Quality Engine tests passed")
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            logger.error(f"❌ Data Quality Engine test failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def test_model_monitoring_engine(self):
        """Test Model Monitoring Engine"""
        logger.info("🧪 Testing Model Monitoring Engine...")
        
        if not self.engines.get('model_monitoring_engine'):
            return {'status': 'error', 'message': 'Engine not loaded'}
        
        try:
            engine = self.engines['model_monitoring_engine']
            results = {}
            
            # Test 1: Model registration
            registration = engine.register_model(
                model_id="test_model_v1",
                model_type="classification",
                model_object=self.test_data['model'],
                baseline_data={
                    'features': self.test_data['X_train'],
                    'labels': self.test_data['y_train']
                },
                metadata={
                    'version': '1.0',
                    'description': 'Test classification model'
                }
            )
            results['model_registration'] = {
                'status': 'success',
                'registration_id': registration.get('model_id'),
                'message': registration.get('message', '')
            }
            
            # Test 2: Performance monitoring
            y_pred = self.test_data['model'].predict(self.test_data['X_test'][:50])
            performance_report = engine.monitor_model_performance(
                model_id="test_model_v1",
                X=self.test_data['X_test'][:50],
                y_true=self.test_data['y_test'][:50],
                y_pred=y_pred
            )
            results['performance_monitoring'] = {
                'status': 'success',
                'health_status': performance_report.get('monitoring_summary', {}).get('overall_health_status', 'unknown'),
                'alerts_generated': len(performance_report.get('alerts', []))
            }
            
            # Test 3: Data drift detection
            drift_report = engine.detect_data_drift(
                model_id="test_model_v1",
                current_data=self.test_data['X_test'][:100],
                reference_data=self.test_data['X_train'][:100]
            )
            results['drift_detection'] = {
                'status': 'success',
                'drift_detected': drift_report.get('drift_detected', False),
                'confidence': drift_report.get('confidence', 0)
            }
            
            # Test 4: Health report
            health_report = engine.get_model_health_report("test_model_v1")
            results['health_report'] = {
                'status': 'success',
                'health_score': health_report.get('health_summary', {}).get('overall_health_score', 0),
                'health_status': health_report.get('health_summary', {}).get('health_status', 'unknown')
            }
            
            logger.info("✅ Model Monitoring Engine tests passed")
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            logger.error(f"❌ Model Monitoring Engine test failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def test_security_engine(self):
        """Test Security Engine"""
        logger.info("🧪 Testing Security Engine...")
        
        if not self.engines.get('security_engine'):
            return {'status': 'error', 'message': 'Engine not loaded'}
        
        try:
            engine = self.engines['security_engine']
            results = {}
            
            # Test 1: Model protection
            protection_report = engine.protect_model(
                model_id="secure_test_model",
                model_object=self.test_data['model'],
                protection_level="high",
                metadata={
                    'sensitivity': 'high',
                    'owner': 'test_suite'
                }
            )
            results['model_protection'] = {
                'status': 'success',
                'protection_level': protection_report.get('protection_level'),
                'security_features': protection_report.get('security_features', [])
            }
            
            # Test 2: Adversarial detection
            adversarial_report = engine.detect_adversarial_attacks(
                model_id="secure_test_model",
                input_data=self.test_data['X_test'][:20]
            )
            results['adversarial_detection'] = {
                'status': 'success',
                'threat_detected': adversarial_report.get('threat_detected', False),
                'threat_level': adversarial_report.get('threat_level', 'low')
            }
            
            # Test 3: Data leakage detection
            leakage_report = engine.detect_data_leakage(
                model_id="secure_test_model",
                training_data=self.test_data['X_train'],
                model_predictions=self.test_data['model'].predict(self.test_data['X_train'])
            )
            results['data_leakage'] = {
                'status': 'success',
                'leakage_risk': leakage_report.get('leakage_risk_detected', False),
                'risk_score': leakage_report.get('overall_risk_score', 0)
            }
            
            # Test 4: Security audit
            audit_report = engine.security_audit("secure_test_model")
            results['security_audit'] = {
                'status': 'success',
                'audit_completed': 'overall_security_score' in audit_report,
                'security_score': audit_report.get('overall_security_score', 0)
            }
            
            logger.info("✅ Security Engine tests passed")
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            logger.error(f"❌ Security Engine test failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def test_explainability_engine(self):
        """Test Explainability Engine"""
        logger.info("🧪 Testing Explainability Engine...")
        
        if not self.engines.get('explainability_engine'):
            return {'status': 'error', 'message': 'Engine not loaded'}
        
        try:
            engine = self.engines['explainability_engine']
            results = {}
            
            # Test 1: Prediction explanation
            explanation_report = engine.explain_prediction(
                model=self.test_data['model'],
                input_data=self.test_data['X_test'][:5],
                feature_names=self.test_data['feature_names'],
                target_names=self.test_data['target_names'],
                scope="local"
            )
            results['prediction_explanation'] = {
                'status': 'success',
                'interpretability_score': explanation_report.get('interpretability_score', 0),
                'methods_used': explanation_report.get('methods_used', [])
            }
            
            # Test 2: Feature importance
            importance_report = engine.analyze_feature_importance(
                model=self.test_data['model'],
                X=self.test_data['X_train'],
                y=self.test_data['y_train'],
                feature_names=self.test_data['feature_names']
            )
            results['feature_importance'] = {
                'status': 'success',
                'features_analyzed': len(importance_report.get('feature_names', [])),
                'consensus_found': 'consensus_importance' in importance_report
            }
            
            # Test 3: Bias detection
            bias_report = engine.detect_model_bias(
                model=self.test_data['model'],
                X=self.test_data['X_test'],
                y=self.test_data['y_test'],
                sensitive_features=self.test_data['X_test'][:, :2],  # First 2 features as sensitive
                feature_names=self.test_data['feature_names']
            )
            results['bias_detection'] = {
                'status': 'success',
                'bias_assessment': bias_report.get('overall_bias_assessment', {}),
                'fairness_status': bias_report.get('fairness_report', {}).get('fairness_status', 'unknown')
            }
            
            # Test 4: Model card generation
            model_card = engine.generate_model_card(
                model=self.test_data['model'],
                X=self.test_data['X_train'],
                y=self.test_data['y_train'],
                feature_names=self.test_data['feature_names'],
                model_metadata={
                    'name': 'Test_Classifier',
                    'version': '1.0',
                    'purpose': 'Integration testing'
                }
            )
            results['model_card'] = {
                'status': 'success',
                'card_id': model_card.get('model_card_id'),
                'risk_level': model_card.get('risk_assessment', {}).get('risk_level', 'unknown')
            }
            
            logger.info("✅ Explainability Engine tests passed")
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            logger.error(f"❌ Explainability Engine test failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def test_engine_integration(self):
        """Test integration between all engines"""
        logger.info("🔗 Testing engine integration...")
        
        try:
            integration_results = {}
            
            # Test: Data Quality -> Model Monitoring pipeline
            dq_engine = self.engines['data_quality_engine']
            mm_engine = self.engines['model_monitoring_engine']
            
            if dq_engine and mm_engine:
                # Analyze data quality first
                dq_report = dq_engine.analyze_data_integrity(self.test_data['X_train'])
                data_quality_score = dq_report.get('data_range', 0.5)  # Example metric
                
                # Use quality score in monitoring
                registration = mm_engine.register_model(
                    model_id="integrated_model",
                    model_type="classification", 
                    model_object=self.test_data['model'],
                    metadata={'data_quality_score': data_quality_score}
                )
                
                integration_results['dq_mm_integration'] = {
                    'status': 'success',
                    'data_quality_score': data_quality_score,
                    'model_registered': registration.get('status') == 'success'
                }
            
            # Test: Security -> Explainability pipeline
            sec_engine = self.engines['security_engine']
            exp_engine = self.engines['explainability_engine']
            
            if sec_engine and exp_engine:
                # Protect model first
                protection = sec_engine.protect_model(
                    model_id="secured_for_explanation",
                    model_object=self.test_data['model'],
                    protection_level="medium"
                )
                
                # Then explain the protected model
                explanation = exp_engine.explain_prediction(
                    model=self.test_data['model'],
                    input_data=self.test_data['X_test'][:3],
                    feature_names=self.test_data['feature_names']
                )
                
                integration_results['sec_exp_integration'] = {
                    'status': 'success',
                    'model_protected': protection.get('status') == 'success',
                    'explanation_generated': 'interpretability_score' in explanation
                }
            
            logger.info("✅ Engine integration tests passed")
            return {'status': 'success', 'results': integration_results}
            
        except Exception as e:
            logger.error(f"❌ Engine integration test failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all engines"""
        logger.info("🚀 Starting comprehensive engine integration test...")
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'test_environment': {
                'python_version': sys.version,
                'numpy_version': np.__version__,
                'pandas_version': pd.__version__
            },
            'engines_loaded': [],
            'test_results': {},
            'integration_results': {},
            'overall_status': 'unknown',
            'summary': {}
        }
        
        # Load all engines
        self.load_engines()
        final_report['engines_loaded'] = [name for name, engine in self.engines.items() if engine is not None]
        
        # Test individual engines
        individual_tests = {
            'data_quality': self.test_data_quality_engine,
            'model_monitoring': self.test_model_monitoring_engine,
            'security': self.test_security_engine,
            'explainability': self.test_explainability_engine
        }
        
        for engine_name, test_function in individual_tests.items():
            logger.info(f"🧪 Testing {engine_name} engine...")
            final_report['test_results'][engine_name] = test_function()
        
        # Test integration
        final_report['integration_results'] = self.test_engine_integration()
        
        # Generate summary
        final_report.update(self._generate_test_summary(final_report))
        
        logger.info("🎉 Comprehensive engine test completed!")
        return final_report
    
    def _generate_test_summary(self, final_report):
        """Generate test summary"""
        summary = {
            'overall_status': 'pass',
            'engines_passed': 0,
            'engines_failed': 0,
            'total_tests': len(final_report['test_results']),
            'successful_integration': False,
            'recommendations': []
        }
        
        # Count passed/failed engines
        for engine_name, result in final_report['test_results'].items():
            if result.get('status') == 'success':
                summary['engines_passed'] += 1
            else:
                summary['engines_failed'] += 1
        
        # Check integration
        if final_report['integration_results'].get('status') == 'success':
            summary['successful_integration'] = True
        
        # Determine overall status
        if summary['engines_failed'] > 0:
            summary['overall_status'] = 'fail'
            summary['recommendations'].append("Fix failing engines before deployment")
        elif not summary['successful_integration']:
            summary['overall_status'] = 'warning'
            summary['recommendations'].append("Review engine integration issues")
        else:
            summary['overall_status'] = 'pass'
            summary['recommendations'].append("All systems ready for production")
        
        return summary
    
    def generate_test_report(self, final_report):
        """Generate human-readable test report"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("ENTERPRISE AI MODEL SENTINEL - ENGINE INTEGRATION TEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Timestamp: {final_report['timestamp']}")
        report_lines.append(f"Overall Status: {final_report['overall_status'].upper()}")
        report_lines.append("")
        
        # Test Environment
        report_lines.append("TEST ENVIRONMENT:")
        report_lines.append("-" * 40)
        env = final_report['test_environment']
        report_lines.append(f"Python: {env['python_version']}")
        report_lines.append(f"NumPy: {env['numpy_version']}")
        report_lines.append(f"Pandas: {env['pandas_version']}")
        report_lines.append("")
        
        # Engines Loaded
        report_lines.append("ENGINES LOADED:")
        report_lines.append("-" * 40)
        for engine in final_report['engines_loaded']:
            report_lines.append(f"✅ {engine}")
        report_lines.append("")
        
        # Individual Test Results
        report_lines.append("INDIVIDUAL ENGINE TESTS:")
        report_lines.append("-" * 40)
        for engine_name, result in final_report['test_results'].items():
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            report_lines.append(f"{status_icon} {engine_name}: {result.get('status', 'unknown')}")
            if result.get('status') == 'error':
                report_lines.append(f"   Error: {result.get('message', 'Unknown error')}")
        report_lines.append("")
        
        # Integration Results
        report_lines.append("ENGINE INTEGRATION:")
        report_lines.append("-" * 40)
        integration = final_report['integration_results']
        status_icon = "✅" if integration.get('status') == 'success' else "❌"
        report_lines.append(f"{status_icon} Integration Test: {integration.get('status', 'unknown')}")
        report_lines.append("")
        
        # Summary
        summary = final_report['summary']
        report_lines.append("SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Engines Passed: {summary['engines_passed']}/{summary['total_tests']}")
        report_lines.append(f"Integration Successful: {summary['successful_integration']}")
        report_lines.append(f"Overall Status: {summary['overall_status'].upper()}")
        report_lines.append("")
        
        # Recommendations
        if summary['recommendations']:
            report_lines.append("RECOMMENDATIONS:")
            report_lines.append("-" * 40)
            for rec in summary['recommendations']:
                report_lines.append(f"• {rec}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("ENGINE INTEGRATION TEST COMPLETED")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

def main():
    """Main function to run integration test"""
    print("🚀 ENTERPRISE AI MODEL SENTINEL - ENGINE INTEGRATION TEST")
    print("=" * 60)
    
    tester = EngineIntegrationTester()
    
    try:
        # Run comprehensive test
        final_report = tester.run_comprehensive_test()
        
        # Generate and print report
        report_text = tester.generate_test_report(final_report)
        print(report_text)
        
        # Save reports
        with open('engine_integration_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        with open('engine_integration_report.json', 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        print(f"\n📊 Text report saved to: engine_integration_report.txt")
        print(f"📋 JSON report saved to: engine_integration_report.json")
        print(f"📝 Log file: engine_integration_test.log")
        
        # Exit code based on overall status
        if final_report['overall_status'] == 'pass':
            print("🎉 All engines are integrated and ready for production!")
            sys.exit(0)
        else:
            print("⚠️  Some integration issues found. Please check the reports.")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()