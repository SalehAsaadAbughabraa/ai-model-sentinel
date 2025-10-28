import sys 
import os 
sys.path.append(os.path.dirname(__file__)) 
 
def main(): 
    print("AI Model Sentinel Enterprise v2.0.0") 
    print("=" * 50) 
 
    # Import and test all systems 
    try: 
        from src.core.global_integration_fixed import global_system 
        from src.security.enterprise_security import security_engine 
        from src.quantum.quantum_math_engine import QuantumMathematicalEngine 
        from src.core.enterprise_backup import backup_system 
 
        print("? All systems imported successfully") 
        print(f"System Status: {global_system.get_status()}") 
 
        # Test quantum engine 
        quantum_engine = QuantumMathematicalEngine() 
        stability = quantum_engine.compute_stability("production_model") 
        print(f"Quantum Stability: {stability}") 
 
        # Test backup system 
        backup_status = backup_system.get_status() 
        print(f"Backup Status: {backup_status}") 
 
        print("?? System ready for production!") 
 
    except Exception as e: 
        print(f"? System initialization failed: {e}") 
 
if __name__ == "__main__": 
    main() 
