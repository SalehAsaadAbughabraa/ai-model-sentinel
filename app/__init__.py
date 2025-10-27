# Global configuration for all engines
class SentinelConfig:
    def __init__(self):
        # Configuration dictionary
        self.config = {
            "ml_engine": {"enabled": True},
            "fusion_engine": {"enabled": True},
            "quantum_engine": {"enabled": True},
            "global": {"security_level": "classified_tier_1"}
        }
        # Direct attributes for engine compatibility
        self.ML_ENGINE_WEIGHT = 0.8
        self.QUANTUM_ENGINE_WEIGHT = 0.9
        self.FUSION_ENGINE_WEIGHT = 0.85
        self.BEHAVIORAL_ENGINE_WEIGHT = 0.75
        self.SIGNATURE_ENGINE_WEIGHT = 0.7
        self.CRITICAL_THRESHOLD = 0.9
        self.HIGH_THRESHOLD = 0.7
        self.MEDIUM_THRESHOLD = 0.5
        self.LOW_THRESHOLD = 0.3
        self.SECURITY_LEVEL = "CLASSIFIED_TIER_1"
        self.ENVIRONMENT = "production"
    
    def get(self, key, default=None):
        return self.config.get(key, default)

# Create global instance
config = SentinelConfig()

# Also make SentinelConfig available directly
SentinelConfig = SentinelConfig
