# Quick Troubleshooting Reference

## Immediate Actions

### 1. System Won't Start
```bash
# Check dependencies
pip install -r requirements.txt

# Verify configuration
python config_validator.py

# Test basic functionality
python -c "import sentinel_core; print('Core OK')"
```

### 2. Engine Failure
```python
# Restart specific engine
from engine_manager import restart_engine
restart_engine('MLEngine')

# Check engine status
from system_status import get_engine_health
print(get_engine_health())
```

### 3. Performance Issues
```bash
# Monitor resources
python tools/monitor.py --cpu --memory --disk

# Clear caches
python tools/clear_cache.py
```

### 4. Database Issues
```bash
# Check database integrity
python tools/db_check.py

# Backup immediately
python tools/backup.py --quick
```

## Common Commands

| Issue | Command |
|-------|---------|
| Check all engines | python -c "from core import status; status()" |
| View logs | tail -f logs/system.log |
| Test API | curl http://localhost:8000/health |
| Backup data | python tools/backup.py --full |
| Update system | git pull && pip install -r requirements.txt |

## Emergency Contacts
- **System Admin:** saleh87alally@gmail.com
- **Dev Team:** saleh87alally@gmail.com
- **Security:** saleh87alally@gmail.com
