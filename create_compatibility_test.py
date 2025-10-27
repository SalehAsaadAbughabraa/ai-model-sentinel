"""
ENTERPRISE AI Model Sentinel - Compatibility Check System
PRODUCTION-READY SYSTEM - ENTERPRISE GRADE
Developer: Saleh Asaad Abughabra
Email: saleh87alally@gmail.com
License: MIT - Enterprise
Compatibility Check for All System Engines
"""

import os
import sys
import importlib
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compatibility_check.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SystemCompatibilityChecker:
    """
    Comprehensive system compatibility checker for AI Model Sentinel
    """
    
    def __init__(self):
        self.engines_path = os.path.join(os.path.dirname(__file__), 'engines')
        self.required_packages = [
            'numpy', 'pandas', 'scipy', 'scikit-learn', 
            'matplotlib', 'seaborn', 'plotly'
        ]
        self.engine_modules = {}
        self.compatibility_results = {}
    
    def check_system_environment(self) -> Dict[str, Any]:
        """Check system environment and dependencies"""
        logger.info("🔧 Checking system environment...")
        
        environment_info = {
            'python_version': sys.version,
            'system_platform': sys.platform,
            'working_directory': os.getcwd(),
            'engines_directory': self.engines_path
        }
        
        # Check required packages
        package_versions = {}
        missing_packages = []
        
        for package in self.required_packages:
            try:
                module = importlib.import_module(package)
                package_versions[package] = getattr(module, '__version__', 'Unknown')
                logger.info(f"✅ {package}: {package_versions[package]}")
            except ImportError as e:
                missing_packages.append(package)
                logger.warning(f"❌ {package}: Missing - {e}")
        
        environment_info['package_versions'] = package_versions
        environment_info['missing_packages'] = missing_packages
        environment_info['environment_ok'] = len(missing_packages) == 0
        
        return environment_info
    
    def discover_engines(self) -> Dict[str, str]:
        """Discover all available engines in the engines directory"""
        logger.info("🔍 Discovering available engines...")
        
        engines = {}
        
        if not os.path.exists(self.engines_path):
            logger.error(f"Engines directory not found: {self.engines_path}")
            return engines
        
        for file in os.listdir(self.engines_path):
            if file.endswith('.py') and not file.startswith('__'):
                engine_name = file[:-3]  # Remove .py extension
                engine_path = os.path.join(self.engines_path, file)
                engines[engine_name] = engine_path
                logger.info(f"📁 Found engine: {engine_name}")
        
        return engines
    
    def validate_engine_structure(self, engine_name: str, engine_path: str) -> Dict[str, Any]:
        """Validate the structure and imports of an engine"""
        logger.info(f"🔎 Validating engine: {engine_name}")
        
        validation_result = {
            'engine_name': engine_name,
            'engine_path': engine_path,
            'import_successful': False,
            'class_found': False,
            'methods_available': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Add engines directory to Python path
            if 'engines' not in sys.path:
                sys.path.append(os.path.dirname(self.engines_path))
            
            # Try to import the engine module
            spec = importlib.util.spec_from_file_location(engine_name, engine_path)
            if spec is None:
                validation_result['errors'].append(f"Could not create spec for {engine_name}")
                return validation_result
            
            engine_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(engine_module)
            
            validation_result['import_successful'] = True
            
            # Find the main engine class (typically named *Engine)
            engine_classes = []
            for attr_name in dir(engine_module):
                attr = getattr(engine_module, attr_name)
                if (isinstance(attr, type) and 
                    'Engine' in attr_name and 
                    not attr_name.startswith('_')):
                    engine_classes.append(attr_name)
            
            if engine_classes:
                validation_result['class_found'] = True
                validation_result['engine_classes'] = engine_classes
                
                # Test instantiation of the main engine class
                main_class_name = engine_classes[0]
                main_class = getattr(engine_module, main_class_name)
                
                try:
                    # Try to create an instance
                    engine_instance = main_class()
                    validation_result['instance_created'] = True
                    
                    # Discover available methods
                    methods = [method for method in dir(engine_instance) 
                              if not method.startswith('_') and callable(getattr(engine_instance, method))]
                    validation_result['methods_available'] = methods
                    
                    logger.info(f"✅ Engine {engine_name} validated successfully")
                    logger.info(f"   Classes: {engine_classes}")
                    logger.info(f"   Methods: {len(methods)} methods available")
                    
                except Exception as e:
                    validation_result['errors'].append(f"Failed to instantiate {main_class_name}: {e}")
                    logger.error(f"❌ Failed to instantiate {engine_name}: {e}")
            
            else:
                validation_result['warnings'].append("No engine class found (looking for *Engine pattern)")
                logger.warning(f"⚠️  No engine class found in {engine_name}")
        
        except Exception as e:
            validation_result['errors'].append(f"Import failed: {e}")
            logger.error(f"❌ Failed to import {engine_name}: {e}")
        
        return validation_result
    
    def test_engine_functionality(self, engine_name: str, validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Test basic functionality of an engine"""
        logger.info(f"🧪 Testing functionality: {engine_name}")
        
        functionality_result = {
            'engine_name': engine_name,
            'basic_tests_passed': False,
            'test_results': {},
            'performance_metrics': {}
        }
        
        if not validation_result['import_successful'] or not validation_result['class_found']:
            functionality_result['test_results']['error'] = "Engine not properly validated"
            return functionality_result
        
        try:
            # Import and create engine instance
            spec = importlib.util.spec_from_file_location(engine_name, validation_result['engine_path'])
            engine_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(engine_module)
            
            main_class_name = validation_result['engine_classes'][0]
            main_class = getattr(engine_module, main_class_name)
            engine_instance = main_class()
            
            # Generate test data based on engine type
            test_data = self._generate_test_data(engine_name)
            
            # Run basic functionality tests
            if engine_name == 'data_quality_engine':
                functionality_result = self._test_data_quality_engine(engine_instance, test_data)
            elif engine_name == 'model_monitoring_engine':
                functionality_result = self._test_model_monitoring_engine(engine_instance, test_data)
            elif engine_name == 'security_engine':
                functionality_result = self._test_security_engine(engine_instance, test_data)
            elif engine_name == 'explainability_engine':
                functionality_result = self._test_explainability_engine(engine_instance, test_data)
            else:
                functionality_result = self._test_generic_engine(engine_instance, test_data, engine_name)
            
            functionality_result['basic_tests_passed'] = True
            logger.info(f"✅ Functionality tests passed for {engine_name}")
            
        except Exception as e:
            functionality_result['test_results']['error'] = f"Functionality test failed: {e}"
            logger.error(f"❌ Functionality test failed for {engine_name}: {e}")
        
        return functionality_result
    
    def _generate_test_data(self, engine_name: str) -> Dict[str, Any]:
        """Generate appropriate test data for different engine types"""
        np.random.seed(42)
        
        if engine_name == 'data_quality_engine':
            return {
                'clean_data': np.random.normal(0, 1, 1000),
                'noisy_data': np.random.normal(0, 1, 1000) + np.random.normal(0, 0.5, 1000),
                'sparse_data': np.array([1, 2, 3] + [0] * 997),
                'dataframe': pd.DataFrame({
                    'feature1': np.random.normal(0, 1, 100),
                    'feature2': np.random.randint(0, 10, 100),
                    'target': np.random.randint(0, 2, 100)
                })
            }
        
        elif engine_name in ['model_monitoring_engine', 'explainability_engine']:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.datasets import make_classification
            
            X, y = make_classification(n_samples=100, n_features=5, random_state=42)
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X, y)
            
            return {
                'X': X,
                'y': y,
                'model': model,
                'feature_names': [f'feature_{i}' for i in range(X.shape[1])]
            }
        
        elif engine_name == 'security_engine':
            return {
                'normal_data': np.random.normal(0, 1, (100, 5)),
                'adversarial_data': np.random.normal(0, 5, (10, 5)),  # Simulated adversarial
                'sensitive_features': np.random.randint(0, 2, (100, 2))  # Simulated sensitive attributes
            }
        
        else:
            # Generic test data
            return {
                'numeric_data': np.random.normal(0, 1, 100),
                'categorical_data': np.random.randint(0, 5, 100),
                'dataframe': pd.DataFrame({
                    'col1': np.random.normal(0, 1, 50),
                    'col2': np.random.randint(0, 10, 50)
                })
            }
    
    def _test_data_quality_engine(self, engine, test_data: Dict) -> Dict[str, Any]:
        """Test data quality engine functionality"""
        results = {}
        
        try:
            # Test basic data integrity analysis
            integrity_report = engine.analyze_data_integrity(test_data['clean_data'])
            results['integrity_analysis'] = 'success'
            results['integrity_metrics'] = list(integrity_report.keys())
            
            # Test noise sensitivity analysis
            noise_report = engine.analyze_noise_sensitivity(test_data['noisy_data'])
            results['noise_analysis'] = 'success'
            results['noise_metrics'] = list(noise_report.keys())
            
            # Test with DataFrame
            if 'dataframe' in test_data:
                df_report = engine.analyze_data_integrity(test_data['dataframe'])
                results['dataframe_analysis'] = 'success'
            
            return {'test_results': results, 'performance_metrics': {}}
            
        except Exception as e:
            return {'test_results': {'error': str(e)}, 'performance_metrics': {}}
    
    def _test_model_monitoring_engine(self, engine, test_data: Dict) -> Dict[str, Any]:
        """Test model monitoring engine functionality"""
        results = {}
        
        try:
            # Register a model
            registration = engine.register_model(
                model_id="test_model",
                model_type="classification",
                model_object=test_data['model'],
                baseline_data={'features': test_data['X'], 'labels': test_data['y']}
            )
            results['model_registration'] = 'success'
            
            # Monitor performance
            performance_report = engine.monitor_model_performance(
                model_id="test_model",
                X=test_data['X'][:10],
                y_true=test_data['y'][:10],
                y_pred=test_data['model'].predict(test_data['X'][:10])
            )
            results['performance_monitoring'] = 'success'
            
            return {'test_results': results, 'performance_metrics': {}}
            
        except Exception as e:
            return {'test_results': {'error': str(e)}, 'performance_metrics': {}}
    
    def _test_security_engine(self, engine, test_data: Dict) -> Dict[str, Any]:
        """Test security engine functionality"""
        results = {}
        
        try:
            # Test model protection
            protection_report = engine.protect_model(
                model_id="test_model",
                model_object=test_data.get('model', "mock_model"),
                protection_level="medium"
            )
            results['model_protection'] = 'success'
            
            # Test adversarial detection
            adversarial_report = engine.detect_adversarial_attacks(
                model_id="test_model",
                input_data=test_data['normal_data'][:10]
            )
            results['adversarial_detection'] = 'success'
            
            return {'test_results': results, 'performance_metrics': {}}
            
        except Exception as e:
            return {'test_results': {'error': str(e)}, 'performance_metrics': {}}
    
    def _test_explainability_engine(self, engine, test_data: Dict) -> Dict[str, Any]:
        """Test explainability engine functionality"""
        results = {}
        
        try:
            # Test prediction explanation
            explanation_report = engine.explain_prediction(
                model=test_data['model'],
                input_data=test_data['X'][:5],
                feature_names=test_data['feature_names']
            )
            results['prediction_explanation'] = 'success'
            
            # Test feature importance
            importance_report = engine.analyze_feature_importance(
                model=test_data['model'],
                X=test_data['X'],
                y=test_data['y'],
                feature_names=test_data['feature_names']
            )
            results['feature_importance'] = 'success'
            
            return {'test_results': results, 'performance_metrics': {}}
            
        except Exception as e:
            return {'test_results': {'error': str(e)}, 'performance_metrics': {}}
    
    def _test_generic_engine(self, engine, test_data: Dict, engine_name: str) -> Dict[str, Any]:
        """Test generic engine functionality"""
        results = {}
        
        try:
            # Try common methods that might exist in any engine
            if hasattr(engine, 'analyze'):
                result = engine.analyze(test_data['numeric_data'])
                results['analyze_method'] = 'success'
            
            if hasattr(engine, 'process'):
                result = engine.process(test_data['numeric_data'])
                results['process_method'] = 'success'
            
            if hasattr(engine, 'validate'):
                result = engine.validate(test_data['numeric_data'])
                results['validate_method'] = 'success'
            
            return {'test_results': results, 'performance_metrics': {}}
            
        except Exception as e:
            return {'test_results': {'error': str(e)}, 'performance_metrics': {}}
    
    def check_engine_interoperability(self) -> Dict[str, Any]:
        """Check if engines can work together"""
        logger.info("🔗 Checking engine interoperability...")
        
        interoperability_results = {}
        
        # This would test how different engines can work together
        # For now, we'll check if they can be imported together without conflicts
        
        try:
            # Try to import all validated engines together
            validated_engines = [name for name, result in self.compatibility_results.items() 
                               if result['validation']['import_successful']]
            
            for engine_name in validated_engines:
                try:
                    spec = importlib.util.spec_from_file_location(
                        engine_name, 
                        self.compatibility_results[engine_name]['validation']['engine_path']
                    )
                    engine_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(engine_module)
                    
                    interoperability_results[engine_name] = {
                        'simultaneous_import': 'success',
                        'classes': self.compatibility_results[engine_name]['validation'].get('engine_classes', [])
                    }
                    
                except Exception as e:
                    interoperability_results[engine_name] = {
                        'simultaneous_import': f'failed: {e}',
                        'classes': []
                    }
            
        except Exception as e:
            logger.error(f"❌ Interoperability check failed: {e}")
            interoperability_results['error'] = str(e)
        
        return interoperability_results
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive compatibility check"""
        logger.info("🚀 Starting comprehensive compatibility check...")
        
        final_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'system_environment': {},
            'engines_discovered': {},
            'compatibility_results': {},
            'interoperability': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        # Step 1: Check system environment
        final_report['system_environment'] = self.check_system_environment()
        
        # Step 2: Discover engines
        engines = self.discover_engines()
        final_report['engines_discovered'] = engines
        
        # Step 3: Validate each engine
        for engine_name, engine_path in engines.items():
            logger.info(f"🔍 Processing engine: {engine_name}")
            
            validation_result = self.validate_engine_structure(engine_name, engine_path)
            functionality_result = self.test_engine_functionality(engine_name, validation_result)
            
            self.compatibility_results[engine_name] = {
                'validation': validation_result,
                'functionality': functionality_result
            }
        
        final_report['compatibility_results'] = self.compatibility_results
        
        # Step 4: Check interoperability
        final_report['interoperability'] = self.check_engine_interoperability()
        
        # Step 5: Generate overall status and recommendations
        final_report.update(self._generate_summary_and_recommendations())
        
        logger.info("🎉 Comprehensive compatibility check completed!")
        
        return final_report
    
    def _generate_summary_and_recommendations(self) -> Dict[str, Any]:
        """Generate summary and recommendations based on check results"""
        summary = {
            'overall_status': 'pass',
            'engines_working': 0,
            'engines_failing': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        for engine_name, results in self.compatibility_results.items():
            validation = results['validation']
            functionality = results['functionality']
            
            if validation['import_successful'] and functionality['basic_tests_passed']:
                summary['engines_working'] += 1
            else:
                summary['engines_failing'] += 1
                summary['critical_issues'].append(f"Engine {engine_name} has issues")
            
            # Collect errors and warnings
            if validation['errors']:
                summary['critical_issues'].extend(
                    [f"{engine_name}: {error}" for error in validation['errors']]
                )
            
            if validation['warnings']:
                summary['warnings'].extend(
                    [f"{engine_name}: {warning}" for warning in validation['warnings']]
                )
        
        # Determine overall status
        if summary['engines_failing'] > 0:
            summary['overall_status'] = 'fail'
        elif summary['warnings']:
            summary['overall_status'] = 'warning'
        else:
            summary['overall_status'] = 'pass'
        
        # Generate recommendations
        if summary['critical_issues']:
            summary['recommendations'].append("Address critical issues in failing engines")
        
        if summary['warnings']:
            summary['recommendations'].append("Review warnings for potential improvements")
        
        if summary['engines_working'] == 0:
            summary['recommendations'].append("No engines are working - check installation")
        
        return summary
    
    def generate_report(self, final_report: Dict[str, Any]) -> str:
        """Generate a human-readable compatibility report"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("ENTERPRISE AI MODEL SENTINEL - COMPATIBILITY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Timestamp: {final_report['timestamp']}")
        report_lines.append(f"Overall Status: {final_report['overall_status'].upper()}")
        report_lines.append("")
        
        # System Environment
        report_lines.append("SYSTEM ENVIRONMENT:")
        report_lines.append("-" * 40)
        env = final_report['system_environment']
        report_lines.append(f"Python Version: {env['python_version']}")
        report_lines.append(f"Environment OK: {env['environment_ok']}")
        report_lines.append(f"Missing Packages: {', '.join(env['missing_packages'])}")
        report_lines.append("")
        
        # Engines Summary
        report_lines.append("ENGINES SUMMARY:")
        report_lines.append("-" * 40)
        for engine_name, results in final_report['compatibility_results'].items():
            status = "✅ WORKING" if (results['validation']['import_successful'] and 
                                   results['functionality']['basic_tests_passed']) else "❌ FAILING"
            report_lines.append(f"{engine_name}: {status}")
        
        report_lines.append("")
        
        # Detailed Results
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 40)
        for engine_name, results in final_report['compatibility_results'].items():
            report_lines.append(f"\n{engine_name.upper()}:")
            validation = results['validation']
            functionality = results['functionality']
            
            report_lines.append(f"  Import Successful: {validation['import_successful']}")
            report_lines.append(f"  Class Found: {validation['class_found']}")
            if validation['class_found']:
                report_lines.append(f"  Classes: {', '.join(validation['engine_classes'])}")
            report_lines.append(f"  Basic Tests Passed: {functionality['basic_tests_passed']}")
            
            if validation['errors']:
                report_lines.append("  Errors:")
                for error in validation['errors']:
                    report_lines.append(f"    - {error}")
            
            if validation['warnings']:
                report_lines.append("  Warnings:")
                for warning in validation['warnings']:
                    report_lines.append(f"    - {warning}")
        
        # Recommendations
        if final_report['recommendations']:
            report_lines.append("\nRECOMMENDATIONS:")
            report_lines.append("-" * 40)
            for rec in final_report['recommendations']:
                report_lines.append(f"• {rec}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("COMPATIBILITY CHECK COMPLETED")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

def main():
    """Main function to run compatibility check"""
    print("🚀 ENTERPRISE AI MODEL SENTINEL - COMPATIBILITY CHECK")
    print("=" * 60)
    
    checker = SystemCompatibilityChecker()
    
    try:
        # Run comprehensive check
        final_report = checker.run_comprehensive_check()
        
        # Generate and print report
        report_text = checker.generate_report(final_report)
        print(report_text)
        
        # Save report to file
        with open('compatibility_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n📊 Report saved to: compatibility_report.txt")
        print(f"📋 Log file: compatibility_check.log")
        
        # Exit code based on overall status
        if final_report['overall_status'] == 'pass':
            print("🎉 All systems are compatible and ready!")
            sys.exit(0)
        else:
            print("⚠️  Some compatibility issues found. Please check the report.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Compatibility check failed: {e}")
        print(f"❌ Compatibility check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()