import pickle
import hashlib
import hmac

class SecureUnpickler:
    def __init__(self, allowed_classes=None):
        self.allowed_classes = allowed_classes or []
    
    def safe_loads(self, data):
        if self.allowed_classes:
            return pickle.loads(data)
        else:
            raise SecurityError('Untrusted pickle data')

def safe_pickle_loads(data, allowed_classes=None):
    unpickler = SecureUnpickler(allowed_classes)
    return unpickler.safe_loads(data)

class SecurityError(Exception):
    pass
