"""
ENTERPRISE AI Model Sentinel - Compatibility Check (Simple Version)
PRODUCTION-READY SYSTEM - ENTERPRISE GRADE
Developer: Saleh Asaad Abughabra
Email: saleh87alally@gmail.com
License: MIT - Enterprise
Compatibility Check for All System Engines - Simple Version
"""

import os
import sys
import importlib
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

# Setup simple logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compatibility_check_simple.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class SystemCompatibilityChecker:
    """
    Simple compatibility checker without emojis
    """
    
    def __init__(self):
        self.engines_path = os.path.join(os.path.dirname(__file__), 'engines')
        self.required_packages = [
            'numpy', 'pandas', 'scipy', 'sklearn', 
            'matplotlib', 'seaborn'
        ]
        self.engine_modules = {}
        self.compatibility_results = {}
    
    def check_system_environment(self) -> Dict[str, Any]:
        """Check system environment and dependencies"""
        logger.info("Checking system environment...")
        
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
                if package == 'sklearn':
                    module = importlib.import_module('sklearn')
                else:
                    module = importlib.import_module(package)
                package_versions[package] = getattr(module, '__version__', 'Unknown')
                logger.info(f"FOUND {package}: {package_versions[package]}")
            except ImportError as e:
                missing_packages.append(package)
                logger.warning(f"MISSING {package}: {e}")
        
        environment_info['package_versions'] = package_versions
        environment_info['missing_packages'] = missing_packages
        environment_info['environment_ok'] = len(missing_packages) == 0
        
        return environment_info
    
    def discover_engines(self) -> Dict[str, str]:
        """Discover all available engines"""
        logger.info("Discovering available engines...")
        
        engines = {}
        
        if not os.path.exists(self.engines_path):
            logger.error(f"Engines directory not found: {self.engines_path}")
            return engines
        
        for file in os.listdir(self.engines_path):
            if file.endswith('.py') and not file.startswith('__'):
                engine_name = file[:-3]
                engine_path = os.path.join(self.engines_path, file)
                engines[engine_name] = engine_path
                logger.info(f"Found engine: {engine_name}")
        
        return engines
    
    def validate_engine_structure(self, engine_name: str, engine_path: str) -> Dict[str, Any]:
        """Validate engine structure"""
        logger.info(f"Validating engine: {engine_name}")
        
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
            
            # Try to import
            spec = importlib.util.spec_from_file_location(engine_name, engine_path)
            if spec is None:
                validation_result['errors'].append(f"Could not create spec for {engine_name}")
                return validation_result
            
            engine_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(engine_module)
            
            validation_result['import_successful'] = True
            
            # Find engine classes
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
                
                # Test instantiation
                main_class_name = engine_classes[0]
                main_class = getattr(engine_module, main_class_name)
                
                try:
                    engine_instance = main_class()
                    validation_result['instance_created'] = True
                    
                    # Discover methods
                    methods = [method for method in dir(engine_instance) 
                              if not method.startswith('_') and callable(getattr(engine_instance, method))]
                    validation_result['methods_available'] = methods
                    
                    logger.info(f"SUCCESS - Engine {engine_name} validated")
                    logger.info(f"  Classes: {engine_classes}")
                    logger.info(f"  Methods: {len(methods)} available")
                    
                except Exception as e:
                    validation_result['errors'].append(f"Failed to instantiate {main_class_name}: {e}")
                    logger.error(f"FAILED - Could not instantiate {engine_name}: {e}")
            
            else:
                validation_result['warnings'].append("No engine class found")
                logger.warning(f"WARNING - No engine class in {engine_name}")
        
        except Exception as e:
            validation_result['errors'].append(f"Import failed: {e}")
            logger.error(f"FAILED - Could not import {engine_name}: {e}")
        
        return validation_result
    
    def run_comprehensive_check(self) -> Dict[str, Any]:
        """Run comprehensive compatibility check"""
        logger.info("Starting comprehensive compatibility check...")
        
        final_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'system_environment': {},
            'engines_discovered': {},
            'compatibility_results': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        # Check system environment
        final_report['system_environment'] = self.check_system_environment()
        
        # Discover engines
        engines = self.discover_engines()
        final_report['engines_discovered'] = engines
        
        # Validate each engine
        for engine_name, engine_path in engines.items():
            logger.info(f"Processing engine: {engine_name}")
            
            validation_result = self.validate_engine_structure(engine_name, engine_path)
            self.compatibility_results[engine_name] = {
                'validation': validation_result
            }
        
        final_report['compatibility_results'] = self.compatibility_results
        
        # Generate summary
        final_report.update(self._generate_summary_and_recommendations())
        
        logger.info("Compatibility check completed!")
        return final_report
    
    def _generate_summary_and_recommendations(self) -> Dict[str, Any]:
        """Generate summary and recommendations"""
        summary = {
            'overall_status': 'PASS',
            'engines_working': 0,
            'engines_failing': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        for engine_name, results in self.compatibility_results.items():
            validation = results['validation']
            
            if validation['import_successful'] and validation['class_found']:
                summary['engines_working'] += 1
            else:
                summary['engines_failing'] += 1
                summary['critical_issues'].append(f"Engine {engine_name} has issues")
            
            # Collect errors
            if validation['errors']:
                summary['critical_issues'].extend(
                    [f"{engine_name}: {error}" for error in validation['errors']]
                )
        
        # Determine overall status
        if summary['engines_failing'] > 0:
            summary['overall_status'] = 'FAIL'
        else:
            summary['overall_status'] = 'PASS'
        
        # Generate recommendations
        if summary['critical_issues']:
            summary['recommendations'].append("Fix engine import issues")
        
        env = self.check_system_environment()
        if not env['environment_ok']:
            summary['recommendations'].append(f"Install missing packages: {', '.join(env['missing_packages'])}")
        
        return summary
    
    def generate_report(self, final_report: Dict[str, Any]) -> str:
        """Generate human-readable compatibility report"""
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("ENTERPRISE AI MODEL SENTINEL - COMPATIBILITY REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Timestamp: {final_report['timestamp']}")
        report_lines.append(f"Overall Status: {final_report['overall_status']}")
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
            validation = results['validation']
            status = "WORKING" if (validation['import_successful'] and validation['class_found']) else "FAILING"
            report_lines.append(f"{engine_name}: {status}")
        
        report_lines.append("")
        
        # Detailed Results
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 40)
        for engine_name, results in final_report['compatibility_results'].items():
            report_lines.append(f"\n{engine_name.upper()}:")
            validation = results['validation']
            
            report_lines.append(f"  Import Successful: {validation['import_successful']}")
            report_lines.append(f"  Class Found: {validation['class_found']}")
            if validation['class_found']:
                report_lines.append(f"  Classes: {', '.join(validation['engine_classes'])}")
            
            if validation['errors']:
                report_lines.append("  Errors:")
                for error in validation['errors']:
                    report_lines.append(f"    - {error}")
        
        # Recommendations
        if final_report['recommendations']:
            report_lines.append("\nRECOMMENDATIONS:")
            report_lines.append("-" * 40)
            for rec in final_report['recommendations']:
                report_lines.append(f"* {rec}")
        
        report_lines.append("\n" + "=" * 80)
        report_lines.append("COMPATIBILITY CHECK COMPLETED")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

def main():
    """Main function to run compatibility check"""
    print("ENTERPRISE AI MODEL SENTINEL - COMPATIBILITY CHECK")
    print("=" * 60)
    
    checker = SystemCompatibilityChecker()
    
    try:
        # Run comprehensive check
        final_report = checker.run_comprehensive_check()
        
        # Generate and print report
        report_text = checker.generate_report(final_report)
        print(report_text)
        
        # Save report to file
        with open('compatibility_report_simple.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\nReport saved to: compatibility_report_simple.txt")
        print(f"Log file: compatibility_check_simple.log")
        
        # Exit code based on overall status
        if final_report['overall_status'] == 'PASS':
            print("All systems are compatible and ready!")
            sys.exit(0)
        else:
            print("Some compatibility issues found. Please check the report.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Compatibility check failed: {e}")
        print(f"Compatibility check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()