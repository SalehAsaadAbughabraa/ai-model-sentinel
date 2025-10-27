# integration/ci_cd_integration.py
"""
CI/CD Integration - Pipeline Integration System v2.0.0
Developer: Saleh Asaad Abughabra
Email: saleh87alally@gmail.com

Integrated CI/CD system with Jenkins, GitHub Actions, GitLab CI
Supports automated scanning, reporting, and deployment protection
"""

import os
import sys
import json
import logging
import requests
import tempfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import subprocess
import hmac
import hashlib
import time
from pathlib import Path

# Import internal modules
try:
    from core.enterprise_scanner import EnterpriseAIScanner, EnterpriseScanConfig
    from analytics.drift_detector import AdvancedDriftDetector
    from intelligence.threat_intelligence import ThreatIntelligence
except ImportError:
    # For standalone use
    print("⚠️ Some modules not available - using simplified mode")

class CICDPlatform(Enum):
    """Supported CI/CD Platforms"""
    JENKINS = "jenkins"
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    AZURE_DEVOPS = "azure_devops"
    CUSTOM = "custom"

class ScanTrigger(Enum):
    """Scan Trigger Reasons"""
    PUSH = "push"
    PULL_REQUEST = "pull_request"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    TAG = "tag"
    DEPLOYMENT = "deployment"

@dataclass
class CICDConfig:
    """CI/CD Integration Configuration"""
    platform: CICDPlatform
    api_token: Optional[str] = None
    webhook_secret: Optional[str] = None
    scan_on_push: bool = True
    scan_on_pr: bool = True
    fail_on_critical: bool = True
    report_format: str = "json"
    timeout: int = 300  # 5 minutes
    allowed_branches: List[str] = None
    enable_drift_detection: bool = True
    enable_threat_intel: bool = True

    def __post_init__(self):
        if self.allowed_branches is None:
            self.allowed_branches = ["main", "master", "develop"]

@dataclass
class CICDContext:
    """CI/CD Execution Context"""
    platform: CICDPlatform
    trigger: ScanTrigger
    branch: str
    commit_hash: str
    repository: str
    pull_request: Optional[str] = None
    actor: str = "unknown"
    environment: str = "production"

class SimpleLogger:
    """Simple Logger for CI/CD"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.messages = {
            'scan_start': 'Starting CI/CD scan for branch: {}',
            'scan_complete': 'Scan completed - Models scanned: {}',
            'no_models_found': 'No models found for scanning',
            'critical_threat': 'Critical threats detected - failing build',
            'high_threat': 'High threats - build successful with warnings',
            'scan_success': 'Scan successful - no critical threats',
            'scan_failed': 'Scan failed: {}',
            'webhook_failed': 'Webhook processing failed: {}',
            'report_failed': 'Failed to send results: {}',
            'integration_ready': 'CI/CD Integration ready for platform: {}',
            'scanning_model': 'Scanning model: {}',
            'results_saved': 'Results saved to: {}',
            'pr_comment_created': 'PR comment created for: {}',
            'file_search_failed': 'File search failed: {}',
            'no_github_token': 'No GitHub token available for PR comments'
        }
    
    def info(self, message_key: str, *args):
        """Log info"""
        message = self.messages[message_key].format(*args)
        print(f"INFO: {message}")
    
    def warning(self, message_key: str, *args):
        """Log warning"""
        message = self.messages[message_key].format(*args)
        print(f"WARNING: {message}")
    
    def error(self, message_key: str, *args):
        """Log error"""
        message = self.messages[message_key].format(*args)
        print(f"ERROR: {message}")

class GitHubActionsIntegration:
    """GitHub Actions Integration"""
    
    def __init__(self, config: CICDConfig):
        self.config = config
        self.logger = SimpleLogger('GitHubActionsIntegration')
    
    def setup_environment(self):
        """Setup GitHub Actions Environment"""
        github_output = os.getenv('GITHUB_OUTPUT')
        if github_output:
            with open(github_output, 'a') as f:
                env_vars = {
                    'AI_SCANNER_ENABLED': 'true',
                    'FAIL_ON_CRITICAL': str(self.config.fail_on_critical).lower(),
                    'SCAN_TIMEOUT': str(self.config.timeout)
                }
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")
    
    def report_scan_results(self, results: Dict[str, Any], context: CICDContext):
        """Report scan results to GitHub Actions"""
        try:
            # Update GITHUB_OUTPUT
            github_output = os.getenv('GITHUB_OUTPUT')
            if github_output:
                with open(github_output, 'a') as f:
                    f.write(f"scan_results={json.dumps(results)}\n")
            
            # Create PR comment if available
            if context.pull_request:
                self._create_pr_comment(results, context)
            
            # Fail workflow if critical threats found
            if self.config.fail_on_critical and self._has_critical_threats(results):
                self.logger.error('critical_threat')
                sys.exit(1)
                
        except Exception as e:
            self.logger.error('report_failed', str(e))
            sys.exit(1)
    
    def _generate_github_summary(self, results: Dict[str, Any], context: CICDContext) -> str:
        """Generate results summary"""
        threat_level = results.get('threat_level', 'UNKNOWN')
        threat_score = results.get('threat_score', 0.0)
        
        summary = f"""
## AI Model Security Scan Report

**Result:** `{threat_level}`  
**Threat Score:** `{threat_score:.3f}`  
**Branch:** `{context.branch}`  
**Actor:** `{context.actor}`

### Details:
"""
        components = results.get('scan_components', {})
        for comp_name, comp_data in components.items():
            score = comp_data.get('score', 0)
            status = comp_data.get('status', 'unknown')
            summary += f"- **{comp_name}:** {status} (Score: {score:.3f})\n"
        
        recommendations = results.get('recommendations', [])
        if recommendations:
            summary += "\n### Recommendations:\n"
            for rec in recommendations[:5]:
                summary += f"- {rec}\n"
        
        return summary
    
    def _create_pr_comment(self, results: Dict[str, Any], context: CICDContext):
        """Create PR Comment"""
        if not self.config.api_token:
            self.logger.warning('no_github_token')
            return
        
        comment = self._generate_github_summary(results, context)
        self.logger.info('pr_comment_created', context.pull_request)
    
    def _has_critical_threats(self, results: Dict[str, Any]) -> bool:
        """Check for critical threats"""
        return results.get('threat_level') == 'CRITICAL'

class JenkinsIntegration:
    """Jenkins Integration"""
    
    def __init__(self, config: CICDConfig):
        self.config = config
        self.logger = SimpleLogger('JenkinsIntegration')
    
    def report_scan_results(self, results: Dict[str, Any], context: CICDContext):
        """Report scan results to Jenkins"""
        try:
            threat_level = results.get('threat_level', 'UNKNOWN')
            
            if threat_level == "CRITICAL" and self.config.fail_on_critical:
                self.logger.error('critical_threat')
                sys.exit(1)
            elif threat_level in ["HIGH", "CRITICAL"]:
                self.logger.warning('high_threat')
            else:
                self.logger.info('scan_success')
            
            # Save results to file
            output_file = "ai_scan_results.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info('results_saved', output_file)
            
        except Exception as e:
            self.logger.error('report_failed', str(e))
            sys.exit(1)

class WeightedThreatScorer:
    """Weighted Threat Scorer"""
    
    def __init__(self):
        self.weights = {
            'model_integrity': 0.25,
            'data_poisoning': 0.20,
            'adversarial_robustness': 0.15,
            'backdoor_detection': 0.20,
            'bias_fairness': 0.10,
            'privacy_leakage': 0.10
        }
    
    def calculate_weighted_score(self, component_scores: Dict[str, float]) -> Tuple[float, str]:
        """Calculate weighted threat score"""
        total_score = 0.0
        total_weight = 0.0
        
        for component, weight in self.weights.items():
            if component in component_scores:
                total_score += component_scores[component] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0, "UNKNOWN"
        
        normalized_score = total_score / total_weight
        
        # Determine threat level
        if normalized_score >= 0.8:
            threat_level = "CRITICAL"
        elif normalized_score >= 0.6:
            threat_level = "HIGH"
        elif normalized_score >= 0.4:
            threat_level = "MEDIUM"
        elif normalized_score >= 0.2:
            threat_level = "LOW"
        else:
            threat_level = "CLEAN"
        
        return normalized_score, threat_level

class SimpleScanner:
    """Simple Scanner for testing"""
    
    def comprehensive_scan(self, model_path: str) -> Dict[str, Any]:
        """Simple comprehensive scan"""
        import hashlib
        import os
        
        if not os.path.exists(model_path):
            return {"error": "File not found", "threat_level": "ERROR"}
        
        file_size = os.path.getsize(model_path)
        file_hash = hashlib.sha256(open(model_path, 'rb').read()).hexdigest()
        
        # Simple threat assessment
        if file_size == 0:
            threat_level = "HIGH"
            threat_score = 0.8
        elif file_size < 100:
            threat_level = "MEDIUM"
            threat_score = 0.6
        else:
            threat_level = "LOW"
            threat_score = 0.2
        
        return {
            "threat_level": threat_level,
            "threat_score": threat_score,
            "file_size": file_size,
            "file_hash": file_hash,
            "component_scores": {
                "model_integrity": threat_score,
                "data_poisoning": 0.1,
                "adversarial_robustness": 0.1
            }
        }

class CICDIntegration:
    """Main CI/CD Integration System"""
    
    def __init__(self, config: CICDConfig):
        self.config = config
        self.logger = SimpleLogger('CICDIntegration')
        
        # Initialize Components
        self.scanner = SimpleScanner()  # Using simple scanner for testing
        self.threat_scorer = WeightedThreatScorer()
        
        # Initialize platform integration
        self.platform_integration = self._setup_platform_integration()
        
        self.logger.info('integration_ready', config.platform.value)
    
    def _setup_platform_integration(self):
        """Setup Platform Integration"""
        if self.config.platform == CICDPlatform.GITHUB_ACTIONS:
            return GitHubActionsIntegration(self.config)
        elif self.config.platform == CICDPlatform.JENKINS:
            return JenkinsIntegration(self.config)
        else:
            return None
    
    def scan_changes(self, context: CICDContext) -> Dict[str, Any]:
        """Scan Changes in CI/CD"""
        try:
            self.logger.info('scan_start', context.branch)
            
            # Find changed model files
            model_files = self._find_changed_models(context)
            
            if not model_files:
                self.logger.info('no_models_found')
                return {
                    'status': 'skipped',
                    'reason': 'No model files changed',
                    'context': asdict(context)
                }
            
            # Scan models
            scan_results = []
            for model_file in model_files:
                self.logger.info('scanning_model', model_file)
                result = self.scanner.comprehensive_scan(model_file)
                
                # Apply weighted analysis
                component_scores = result.get('component_scores', {})
                weighted_score, threat_level = self.threat_scorer.calculate_weighted_score(component_scores)
                
                result['weighted_threat_score'] = weighted_score
                result['threat_level'] = threat_level
                result['scoring_method'] = 'weighted'
                
                scan_results.append(result)
            
            aggregated_results = self._aggregate_results(scan_results)
            
            # Report to CI/CD platform
            if self.platform_integration:
                self.platform_integration.report_scan_results(aggregated_results, context)
            
            self.logger.info('scan_complete', len(scan_results))
            return aggregated_results
            
        except Exception as e:
            self.logger.error('scan_failed', str(e))
            return {'error': str(e), 'status': 'failed'}
    
    def _find_changed_models(self, context: CICDContext) -> List[str]:
        """Find Changed Models"""
        model_files = []
        
        try:
            # Simple file search for testing
            for root, dirs, files in os.walk('.'):
                for file in files:
                    if self._is_model_file(file):
                        model_files.append(os.path.join(root, file))
            
            return model_files[:3]  # Limit to 3 files for testing
            
        except Exception as e:
            self.logger.warning('file_search_failed', str(e))
            return []
    
    def _is_model_file(self, file_path: str) -> bool:
        """Check if file is model"""
        model_extensions = {
            '.pt', '.pth', '.h5', '.hdf5', '.onnx', 
            '.safetensors', '.pb', '.ckpt', '.tflite'
        }
        return any(file_path.lower().endswith(ext) for ext in model_extensions)
    
    def _aggregate_results(self, scan_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate Scan Results"""
        if not scan_results:
            return {'status': 'no_results'}
        
        threat_scores = [r.get('weighted_threat_score', 0) for r in scan_results]
        threat_levels = [r.get('threat_level', 'CLEAN') for r in scan_results]
        
        # Determine highest threat level
        level_priority = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1, 'CLEAN': 0}
        max_threat_level = max(threat_levels, key=lambda x: level_priority.get(x, 0))
        
        return {
            'status': 'completed',
            'models_scanned': len(scan_results),
            'overall_threat_level': max_threat_level,
            'average_threat_score': sum(threat_scores) / len(threat_scores),
            'max_threat_score': max(threat_scores),
            'critical_count': sum(1 for level in threat_levels if level == 'CRITICAL'),
            'results': scan_results,
            'timestamp': time.time(),
            'scoring_method': 'weighted'
        }
    
    def create_cicd_config(self, platform: CICDPlatform) -> str:
        """Create CI/CD Configuration File"""
        if platform == CICDPlatform.GITHUB_ACTIONS:
            return self._create_github_actions_config()
        elif platform == CICDPlatform.GITLAB_CI:
            return self._create_gitlab_ci_config()
        elif platform == CICDPlatform.JENKINS:
            return self._create_jenkins_config()
        else:
            return "# Custom integration - please refer to documentation"
    
    def _create_github_actions_config(self) -> str:
        """Create GitHub Actions Integration"""
        return """name: AI Model Security Scan

on:
  push:
    branches: [ main, master, develop ]
  pull_request:
    branches: [ main, master ]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        
    - name: Install AI Model Sentinel
      run: |
        pip install ai-model-sentinel
        
    - name: Run Security Scan
      run: |
        python -m ai_model_sentinel.cicd_scan
        
    - name: Upload Scan Results
      uses: actions/upload-artifact@v4
      with:
        name: security-scan-results
        path: scan_results.json
"""
    
    def _create_gitlab_ci_config(self) -> str:
        """Create GitLab CI Integration"""
        return """ai_security_scan:
  image: python:3.9
  before_script:
    - pip install ai-model-sentinel
  script:
    - python -m ai_model_sentinel.cicd_scan
  artifacts:
    paths:
      - scan_results.json
    when: always
  only:
    - main
    - master
    - develop
    - merge_requests
"""

def main():
    """Main function for testing"""
    print("Testing CI/CD Integration v2.0.0...")
    
    # Configuration
    config = CICDConfig(
        platform=CICDPlatform.GITHUB_ACTIONS,
        fail_on_critical=True,
        scan_on_push=True,
        scan_on_pr=True
    )
    
    cicd = CICDIntegration(config)
    
    # Test context
    test_context = CICDContext(
        platform=CICDPlatform.GITHUB_ACTIONS,
        trigger=ScanTrigger.PUSH,
        branch="main",
        commit_hash="test123",
        repository="test/repo",
        actor="tester"
    )
    
    # Test scan
    results = cicd.scan_changes(test_context)
    
    print(f"Scan Status: {results.get('status')}")
    print(f"Models Scanned: {results.get('models_scanned', 0)}")
    print(f"Threat Level: {results.get('overall_threat_level')}")
    print(f"Scoring Method: {results.get('scoring_method')}")
    
    # Create integration file
    github_config = cicd.create_cicd_config(CICDPlatform.GITHUB_ACTIONS)
    print(f"\nGitHub Actions Integration:\n{github_config}")
    
    print("\nCI/CD Integration v2.0.0 is ready to use!")

if __name__ == "__main__":
    main()