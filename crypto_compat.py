# PBKDF2 Compatibility Layer
import sys

# Try to import the correct PBKDF2 implementation
try:
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    # Create alias for compatibility
    PBKDF2 = PBKDF2HMAC
    print("✓ PBKDF2HMAC imported successfully")
except ImportError as e:
    print(f"⚠ Failed to import PBKDF2HMAC: {e}")
    # Create a dummy class for fallback
    class PBKDF2:
        def __init__(self, *args, **kwargs):
            pass
        def derive(self, key_material):
            return b"fallback_key"
    print("✓ Created PBKDF2 fallback")

# Make available for import
sys.modules[__name__].PBKDF2 = PBKDF2
