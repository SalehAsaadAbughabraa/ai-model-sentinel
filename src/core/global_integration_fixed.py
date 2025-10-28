import sys
sys.path.append('.')

class GlobalIntegrationSystem:
    def __init__(self):
        self.status = 'ACTIVE'
        self.version = '2.0.0'
        self.components = {}
        self.load_components()
    
    def load_components(self):
        try:
            from dynamic_engine_fixed import DynamicRuleEngineFixed
            from quantum_engines_fixed import QuantumFingerprintEngine
            from advanced_database_fixed import db_connector
            from enterprise_security import security_engine
            from enterprise_backup import backup_system
            
            self.components = {
                'DynamicEngine': DynamicRuleEngineFixed(),
                'QuantumEngine': QuantumFingerprintEngine(),
                'Database': db_connector,
                'Security': security_engine,
                'Backup': backup_system
            }
            print('All components loaded successfully')
            return True
        except Exception as e:
            print('Load error:', e)
            return False
    
    def get_status(self):
        status = {}
        for name, comp in self.components.items():
            if hasattr(comp, 'status'):
                status[name] = comp.status
            elif hasattr(comp, 'is_connected'):
                status[name] = 'CONNECTED' if comp.is_connected() else 'DISCONNECTED'
            else:
                status[name] = 'ACTIVE'
        return status

global_system = GlobalIntegrationSystem()
print('Global integration system ready')
