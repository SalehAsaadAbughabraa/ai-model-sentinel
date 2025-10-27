import hashlib
import hmac
import struct

class UniversalPBKDF2:
    def __init__(self, algorithm, length, salt, iterations, backend=None):
        self.algorithm = algorithm
        self.length = length
        self.salt = salt
        self.iterations = iterations
    
    def derive(self, key_material):
        algo = getattr(hashlib, self.algorithm.lower())
        key = b''
        block_index = 1
        while len(key) < self.length:
            u = hmac.new(key_material, self.salt + struct.pack('>I', block_index), algo).digest()
            block = u
            for _ in range(self.iterations - 1):
                u = hmac.new(key_material, u, algo).digest()
                block = bytes(a ^ b for a, b in zip(block, u))
            key += block
            block_index += 1
        return key[:self.length]

PBKDF2 = UniversalPBKDF2
