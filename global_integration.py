import sys
sys.path.append('.')

class GlobalIntegrationSystem:
    def __init__(self):
        self.status = 'ACTIVE'
        self.version = '2.0.0'
        self.load_all_components()
    
    def load_all_components(self):
        # Load all fixed engines
        from dynamic_engine_fixed import DynamicRuleEngineFixed
        from quantum_engines_fixed import QuantumFingerprintEngine, QuantumNeuralFingerprintEngine, ProductionQuantumSecurityEngine
        from pbkdf2_universal import PBKDF2
        from advanced_database_fixed import db_connector
        from enterprise_security import security_engine, threat_monitor
        from enterprise_backup import backup_system
        from core.models.user_models import UserRole
        
        self.components = {
            'DynamicRuleEngine': DynamicRuleEngineFixed(),
            'QuantumFingerprintEngine': QuantumFingerprintEngine(),
            'QuantumNeuralFingerprintEngine': QuantumNeuralFingerprintEngine(),
            'ProductionQuantumSecurityEngine': ProductionQuantumSecurityEngine(),
            'DatabaseConnector': db_connector,
            'SecurityEngine': security_engine,
            'ThreatMonitor': threat_monitor,
            'BackupSystem': backup_system,
            'UserRole': UserRole
        }
        print('All system components loaded successfully')
    
    def get_system_status(self):
        status = {}
        for name, component in self.components.items():
            if hasattr(component, 'status'):
                status[name] = component.status
            elif hasattr(component, 'is_connected'):
                status[name] = 'CONNECTED' if component.is_connected() else 'DISCONNECTED'
            else:
                status[name] = 'ACTIVE'
        return status
    
    def run_health_check(self):
        print('=== SYSTEM HEALTH CHECK ===')
        status = self.get_system_status()
        for component, state in status.items():
            print(f'{component}: {state}')
        
        active_components = sum(1 for state in status.values() if state in ['ACTIVE', 'CONNECTED'])
        total_components = len(status)
        
        print(f'Health Score: {active_components}/{total_components} ({active_components/total_components*100:.1f}%)')
        return active_components == total_components

global_system = GlobalIntegrationSystem()
print('Global integration system ready')
