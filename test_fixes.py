import sys 
import os 
sys.path.append('.') 
 
print('?? Testing System Fixes') 
print('=' * 40) 
 
# Test 1: Global Integration 
print('1. Testing global_integration_fixed...') 
try: 
    from src.core.global_integration_fixed import global_system 
    print('   ? global_integration_fixed - FIXED') 
    print(f'   System status: {global_system.get_status()}') 
except Exception as e: 
    print(f'   ? global_integration_fixed - STILL BROKEN: {e}') 
 
# Test 2: Quantum Mathematical Engine 
print('2. Testing QuantumMathematicalEngine...') 
try: 
    from src.quantum.quantum_math_engine import QuantumMathematicalEngine 
    q = QuantumMathematicalEngine() 
    stability = q.compute_stability('test_model') 
    print(f'   ? QuantumMathematicalEngine - FIXED: {stability}') 
except Exception as e: 
    print(f'   ? QuantumMathematicalEngine - STILL BROKEN: {e}') 
 
# Test 3: Security Engine 
print('3. Testing security engine...') 
try: 
    from src.security.enterprise_security import security_engine 
    encrypted = security_engine.encrypt_data('test_data') 
    print('   ? Security Engine - WORKING') 
    print(f'   Encryption test: {type(encrypted)}') 
except Exception as e: 
    print(f'   ? Security Engine - BROKEN: {e}') 
 
# Test 4: Backup System 
print('4. Testing backup system...') 
try: 
    from src.core.enterprise_backup import backup_system 
    print('   ? Backup System - WORKING') 
    print(f'   Backup status: {backup_system.get_status()}') 
except Exception as e: 
    print(f'   ? Backup System - BROKEN: {e}') 
 
print('=' * 40) 
print('Fix testing completed!') 
