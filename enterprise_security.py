import hashlib
import hmac
import os
import base64
from cryptography.fernet import Fernet

class EnterpriseSecurityEngine:
    def __init__(self):
        self.version = '2.0.0'
        self.status = 'ACTIVE'
        self.encryption_key = self._generate_encryption_key()
        self.fernet = Fernet(self.encryption_key)
    
    def _generate_encryption_key(self):
        key = os.urandom(32)
        return base64.urlsafe_b64encode(key)
    
    def encrypt_data(self, data):
        if isinstance(data, str):
            data = data.encode()
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data):
        return self.fernet.decrypt(encrypted_data).decode()
    
    def generate_hash(self, data, algorithm='sha256'):
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()
    
    def verify_integrity(self, data, expected_hash):
        return hmac.compare_digest(self.generate_hash(data), expected_hash)

class ThreatMonitoringEngine:
    def __init__(self):
        self.status = 'ACTIVE'
        self.suspicious_activities = []
    
    def monitor_activity(self, user_id, action, context):
        risk_score = self._calculate_risk_score(user_id, action, context)
        if risk_score > 0.7:
            alert = {'user_id': user_id, 'action': action, 'risk_score': risk_score}
            self.suspicious_activities.append(alert)
            print(f'SECURITY ALERT: {alert}')
        return risk_score
    
    def _calculate_risk_score(self, user_id, action, context):
        base_score = 0.1
        if action in ['delete', 'export', 'admin_access']:
            base_score += 0.3
        return min(base_score, 1.0)

security_engine = EnterpriseSecurityEngine()
threat_monitor = ThreatMonitoringEngine()
print('Enterprise security system ready')
