class SentinelConfig:
    def __init__(self):
        self.config = {
            'ml_engine': {'enabled': True, 'model_path': 'models/ml_model.pkl'},
            'fusion_engine': {'enabled': True, 'fusion_mode': 'quantum_enhanced'},
            'quantum_engine': {'enabled': True, 'quantum_level': 'cosmic'},
            'global': {'security_level': 'classified_tier_1'}
        }
    def get(self, key, default=None):
        return self.config.get(key, default)

sentinel_config = SentinelConfig()
