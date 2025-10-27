# Final PBKDF2 Fix
import sys
import cryptography.hazmat.primitives.kdf.pbkdf2

# Directly patch the module
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2 = PBKDF2HMAC

print("✓ PBKDF2 permanently patched in cryptography module")

# Also create global alias
sys.modules[__name__].PBKDF2 = PBKDF2HMAC
