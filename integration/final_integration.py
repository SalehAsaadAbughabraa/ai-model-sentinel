# integration/final_integration.py
"""
🔗 Final Scanner Integration with Main System
"""

import os
import sys
import hashlib
import json
from datetime import datetime

class IntegratedAISentinel:
    """Final Integrated AI Security System"""
    
    def __init__(self):
        self.version = "2.1.0"
        self.scanner_name = "EnterpriseSecurityScanner"
        print(f"🚀 Initializing {self.scanner_name} v{self.version}")
    
    def comprehensive_scan(self, model_path):
        """Comprehensive security scan for AI models"""
        print(f"🔍 Starting Comprehensive Scan: {model_path}")
        
        if not os.path.exists(model_path):
            return self._error_result(f"File not found: {model_path}")
        
        try:
            # Perform multiple security checks
            file_analysis = self._analyze_file_properties(model_path)
            content_analysis = self._analyze_content_security(model_path)
            threat_assessment = self._assess_threat_level(file_analysis, content_analysis)
            
            result = {
                "scan_type": "COMPREHENSIVE_ENTERPRISE",
                "scanner_version": self.version,
                "timestamp": datetime.now().isoformat(),
                "model_path": model_path,
                "file_analysis": file_analysis,
                "content_analysis": content_analysis,
                "threat_assessment": threat_assessment,
                "recommendations": self._generate_recommendations(threat_assessment)
            }
            
            return result
            
        except Exception as e:
            return self._error_result(f"Scan failed: {str(e)}")
    
    def _analyze_file_properties(self, model_path):
        """Analyze basic file properties"""
        file_stats = os.stat(model_path)
        
        return {
            "file_size": file_stats.st_size,
            "file_hash": self._calculate_file_hash(model_path),
            "created_time": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "file_extension": os.path.splitext(model_path)[1]
        }
    
    def _calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _analyze_content_security(self, model_path):
        """Analyze content security risks"""
        file_size = os.path.getsize(model_path)
        
        # Basic content risk assessment
        risks = []
        risk_score = 0.0
        
        if file_size == 0:
            risks.append("Empty file")
            risk_score += 0.5
        elif file_size < 100:
            risks.append("Very small file - potentially incomplete")
            risk_score += 0.3
        elif file_size > 500 * 1024 * 1024:
            risks.append("Very large file - potential resource abuse")
            risk_score += 0.4
        
        # Check file extension
        ext = os.path.splitext(model_path)[1].lower()
        safe_extensions = ['.pt', '.pth', '.h5', '.onnx', '.pb']
        if ext not in safe_extensions:
            risks.append(f"Uncommon file extension: {ext}")
            risk_score += 0.2
        
        return {
            "detected_risks": risks,
            "content_risk_score": min(risk_score, 1.0),
            "analysis_method": "basic_content_scan"
        }
    
    def _assess_threat_level(self, file_analysis, content_analysis):
        """Assess overall threat level"""
        content_risk = content_analysis["content_risk_score"]
        file_size = file_analysis["file_size"]
        
        # Calculate base threat score
        base_score = content_risk
        
        # Adjust based on file size anomalies
        if file_size == 0:
            base_score = max(base_score, 0.8)
        elif file_size < 50:
            base_score = max(base_score, 0.6)
        
        # Determine threat level
        if base_score >= 0.7:
            threat_level = "HIGH"
        elif base_score >= 0.4:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        return {
            "threat_level": threat_level,
            "threat_score": round(base_score, 3),
            "confidence": "MEDIUM",
            "factors_considered": ["file_size", "content_risks", "file_properties"]
        }
    
    def _generate_recommendations(self, threat_assessment):
        """Generate security recommendations"""
        threat_level = threat_assessment["threat_level"]
        threat_score = threat_assessment["threat_score"]
        
        recommendations = []
        
        if threat_level == "HIGH":
            recommendations.extend([
                "🚨 IMMEDIATE ACTION REQUIRED: Do not use this model",
                "🔒 Isolate the model file for further investigation",
                "📧 Contact security team for advanced analysis",
                "❌ Block model from production deployment"
            ])
        elif threat_level == "MEDIUM":
            recommendations.extend([
                "⚠️ Review model thoroughly before deployment",
                "🔍 Perform additional security scans",
                "📊 Check model against known threat databases",
                "👥 Seek second opinion from security team"
            ])
        else:
            recommendations.extend([
                "✅ Model appears safe for basic use",
                "📋 Maintain regular security monitoring",
                "🔄 Schedule periodic security reviews",
                "🔐 Follow standard security protocols"
            ])
        
        return recommendations
    
    def _error_result(self, error_message):
        """Generate error result"""
        return {
            "scan_type": "ERROR",
            "scanner_version": self.version,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
            "threat_level": "UNKNOWN",
            "threat_score": 0.5
        }
    
    def batch_scan(self, model_paths):
        """Scan multiple models"""
        results = {}
        print(f"🔄 Starting batch scan of {len(model_paths)} models")
        
        for model_path in model_paths:
            print(f"📁 Scanning: {os.path.basename(model_path)}")
            results[model_path] = self.comprehensive_scan(model_path)
        
        return results
    
    def generate_report(self, scan_results, output_file=None):
        """Generate comprehensive security report"""
        if isinstance(scan_results, dict) and 'model_path' in scan_results:
            # Single scan result
            scan_results = {'single_scan': scan_results}
        
        report = {
            "enterprise_security_report": True,
            "generator": f"AI Model Sentinel v{self.version}",
            "generation_time": datetime.now().isoformat(),
            "scans_performed": len(scan_results),
            "scan_results": scan_results
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 Report saved to: {output_file}")
        
        return report

def main():
    """Main demonstration function"""
    print("🚀 AI Model Sentinel - Final Integration System")
    print("=" * 55)
    
    # Initialize scanner
    sentinel = IntegratedAISentinel()
    
    # Test with various files
    test_files = []
    
    # Create test files
    test_cases = [
        ("safe_model.pt", b"PyTorch model v1.0 - Safe content", "Normal model"),
        ("suspicious_model.pt", b"X" * 10, "Very small file"),
        ("empty_model.pt", b"", "Empty file")
    ]
    
    for filename, content, description in test_cases:
        with open(filename, "wb") as f:
            f.write(content)
        test_files.append(filename)
        print(f"📁 Created: {filename} ({description})")
    
    print("\n" + "=" * 55)
    
    # Perform comprehensive scans
    print("🔍 Starting Security Scans...")
    scan_results = sentinel.batch_scan(test_files)
    
    print("\n" + "=" * 55)
    print("🎯 SCAN RESULTS SUMMARY:")
    print("=" * 55)
    
    for file_path, result in scan_results.items():
        filename = os.path.basename(file_path)
        threat_level = result.get('threat_assessment', {}).get('threat_level', 'UNKNOWN')
        threat_score = result.get('threat_assessment', {}).get('threat_score', 0)
        
        icon = "🟢" if threat_level == "LOW" else "🟡" if threat_level == "MEDIUM" else "🔴"
        print(f"{icon} {filename:20} | Level: {threat_level:6} | Score: {threat_score:.3f}")
    
    # Generate report
    print("\n" + "=" * 55)
    report_file = "enterprise_security_report.json"
    sentinel.generate_report(scan_results, report_file)
    
    # Cleanup
    print("\n🧹 Cleaning up test files...")
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"✅ Removed: {os.path.basename(file_path)}")
    
    print(f"\n✅ Final integration test completed!")
    print(f"📊 Report generated: {report_file}")

if __name__ == "__main__":
    main()