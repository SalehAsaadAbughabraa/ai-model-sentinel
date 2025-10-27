import sys
sys.path.append('.')

# Patch PBKDF2 globally
import cryptography.hazmat.primitives.kdf.pbkdf2
from pbkdf2_universal import PBKDF2
cryptography.hazmat.primitives.kdf.pbkdf2.PBKDF2 = PBKDF2

print('Global PBKDF2 patch applied')
