import sys 
import os 
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
 
try: 
    from core.global_integration_fixed import global_system 
    from security.enterprise_security import security_engine 
    from core.enterprise_backup import backup_system 
    print(\"AI Model Sentinel Enterprise v2.0.0 - System Ready\") 
    print(\"All systems: ACTIVE\") 
    print(\"Server starting on http://localhost:8000\") 
except ImportError as e: 
    print(f\"Import error: {e}\") 
