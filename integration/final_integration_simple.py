"""
Developer: Saleh Asaad Abughabra
Email: saleh87alally@gmail.com
"""

import os
import sys
import hashlib
import re
from datetime import datetime

# Add current path for import
current_dir = os.path.dirname(os.path.dirname(__file__)) if '__file__' in locals() else os.getcwd()
sys.path.append(current_dir)

class SimpleThreatDetector:
    """Simple threat scanner that works without complex imports"""
    
    def __init__(self):
        self.suspicious_patterns = [
            rb"backdoor", rb"trojan", rb"malicious",
            rb"poison", rb"adversarial", rb"evasion",
            rb"exec", rb"system", rb"shell"
        ]
    
    def simple_scan(self, file_path):
        """Simple scan"""
        try:
            if not os.path.exists(file_path):
                return {"error": "File not found", "threat_level": "ERROR"}
            
            file_size = os.path.getsize(file_path)
            
            # File size check
            size_score = 0.0
            if file_size > 500 * 1024 * 1024:  # > 500MB
                size_score = 0.4
            elif file_size > 100 * 1024 * 1024:  # > 100MB
                size_score = 0.2
            elif file_size < 100:  # < 100 bytes
                size_score = 0.3
            
            # Content check
            content_score = 0.0
            detected_threats = []
            
            with open(file_path, "rb") as f:
                content = f.read(8192)  # 8KB
            
            for pattern in self.suspicious_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    content_score += 0.2
                    threat_name = pattern.decode('utf-8', errors='ignore') if isinstance(pattern, bytes) else str(pattern)
                    detected_threats.append(threat_name[:20])
            
            # Calculate final score
            threat_score = min(size_score + content_score, 1.0)
            
            # Determine threat level
            if threat_score >= 0.7:
                threat_level = "HIGH"
            elif threat_score >= 0.4:
                threat_level = "MEDIUM"
            elif threat_score >= 0.2:
                threat_level = "LOW"
            else:
                threat_level = "CLEAN"
            
            return {
                "threat_level": threat_level,
                "threat_score": round(threat_score, 3),
                "file_size": file_size,
                "detected_threats": detected_threats[:3],  # Only first 3 threats
                "analysis_method": "simple_detector",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "threat_level": "ERROR",
                "threat_score": 0.5,
                "error": str(e),
                "analysis_method": "simple_error"
            }

class SimpleIntegratedScanner:
    """Simple Integrated Scanner"""
    
    def __init__(self):
        self.detector = SimpleThreatDetector()
        self.version = "2.0.0"
        self.author = "Saleh Asaad Abughabra"
    
    def comprehensive_scan(self, model_path):
        """Simple comprehensive scan"""
        print(f"🔍 Starting scan: {model_path}")
        print(f"⚡ Scanner: SimpleIntegratedScanner v{self.version}")
        
        # Security scan
        security_result = self.detector.simple_scan(model_path)
        
        # Final result
        result = {
            "enterprise_scan": True,
            "scanner_version": self.version,
            "author": self.author,
            "model_path": model_path,
            "security_analysis": security_result
        }
        
        return result
    
    def batch_scan(self, model_paths):
        """Batch scan"""
        results = []
        for model_path in model_paths:
            result = self.comprehensive_scan(model_path)
            results.append(result)
        return results

def main():
    """Main test function"""
    print("🚀 AI Model Sentinel - Simple Version")
    print("==========================================")
    
    scanner = SimpleIntegratedScanner()
    
    # Create test file
    test_model = "test_model.pt"
    try:
        with open(test_model, "wb") as f:
            f.write(b"PyTorch model parameters - AI Model Sentinel Test")
        
        print(f"📁 Scanning file: {test_model}")
        result = scanner.comprehensive_scan(test_model)
        
        security = result["security_analysis"]
        
        print("\n🎯 Scan Results:")
        print(f"🛡️ Threat Level: {security['threat_level']}")
        print(f"🎯 Threat Score: {security['threat_score']}")
        print(f"📁 File Size: {security['file_size']:,} bytes")
        print(f"⚡ Analysis Method: {security['analysis_method']}")
        
        if security.get('detected_threats'):
            print(f"🎯 Detected Threats: {', '.join(security['detected_threats'])}")
        
        if security.get('error'):
            print(f"❌ Error: {security['error']}")
        
        print(f"\n✅ Scan completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
    
    finally:
        # Cleanup
        if os.path.exists(test_model):
            os.remove(test_model)
            print(f"🧹 Test file cleaned up")

if __name__ == "__main__":
    main()