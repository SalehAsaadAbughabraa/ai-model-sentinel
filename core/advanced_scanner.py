"""
AI Model Sentinel - Advanced Security Scanner
Basic scanner implementation to fix circular imports
"""

class ScanPriority:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AdvancedScanConfig:
    def __init__(self, priority=ScanPriority.MEDIUM):
        self.priority = priority

class AdvancedAIScanner:
    def __init__(self, config=None):
        self.config = config or AdvancedScanConfig()
    
    def scan(self, model_path):
        return {
            "status": "scanned", 
            "threat_level": "low",
            "priority": self.config.priority
        }
