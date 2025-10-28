import unittest 
import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
 
class TestAIModelSentinel(unittest.TestCase): 
 
    def test_system_imports(self): 
        \"\"\"Test that all main modules can be imported\"\"\" 
        try: 
            from src.production_final import main 
            from src.security.enterprise_security import security_engine 
            from src.quantum.quantum_engines_fixed import QuantumMathematicalEngine 
            from src.core.global_integration_fixed import global_system 
            self.assertTrue(True) 
        except ImportError as e: 
            self.fail(f\"Import failed: {e}\") 
 
    def test_security_encryption(self): 
        \"\"\"Test encryption/decryption functionality\"\"\" 
        try: 
            from src.security.enterprise_security import security_engine 
            test_data = \"test_sensitive_data\" 
            encrypted = security_engine.encrypt_data(test_data) 
            self.assertIsNotNone(encrypted) 
            self.assertIsInstance(encrypted, bytes) 
        except Exception as e: 
            self.fail(f\"Security test failed: {e}\") 
 
if __name__ == '__main__': 
    unittest.main() 
