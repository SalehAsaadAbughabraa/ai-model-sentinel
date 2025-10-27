import json
from pathlib import Path
from datetime import datetime

class TroubleshootingGenerator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "enterprise_sentinel_docs_v2"
    
    def analyze_common_issues(self):
        print("Analyzing common issues and errors...")
        
        common_issues = {
            "import_errors": [
                {
                    "issue": "ModuleNotFoundError for quantum libraries",
                    "symptoms": "ImportError: No module named 'qiskit'",
                    "causes": ["Missing quantum computing dependencies", "Virtual environment not activated"],
                    "solutions": [
                        "pip install qiskit cryptography",
                        "Activate virtual environment: source venv/bin/activate",
                        "Check Python path and environment variables"
                    ]
                },
                {
                    "issue": "Database connection errors",
                    "symptoms": "sqlite3.OperationalError: unable to open database file",
                    "causes": ["Database file permissions", "File path incorrect", "Database locked"],
                    "solutions": [
                        "Check file permissions: chmod 644 database.db",
                        "Verify database path in configuration",
                        "Ensure no other process is using the database"
                    ]
                }
            ],
            "performance_issues": [
                {
                    "issue": "High memory usage in ML engines",
                    "symptoms": "Memory usage > 80%, slow response times",
                    "causes": ["Large model files", "Memory leaks", "Inefficient data processing"],
                    "solutions": [
                        "Implement model caching",
                        "Use smaller batch sizes",
                        "Monitor and optimize data pipelines"
                    ]
                },
                {
                    "issue": "Quantum engine timeout",
                    "symptoms": "Quantum calculations taking too long",
                    "causes": ["Complex quantum circuits", "Hardware limitations", "Algorithm complexity"],
                    "solutions": [
                        "Simplify quantum circuits",
                        "Use classical pre-processing",
                        "Implement circuit optimization"
                    ]
                }
            ],
            "security_issues": [
                {
                    "issue": "SSL/TLS certificate errors",
                    "symptoms": "SSLError: certificate verify failed",
                    "causes": ["Expired certificates", "Self-signed certificates", "Certificate chain issues"],
                    "solutions": [
                        "Update SSL certificates",
                        "Configure certificate verification",
                        "Use proper CA bundles"
                    ]
                }
            ]
        }
        
        return common_issues
    
    def generate_troubleshooting_guide(self):
        print("Generating comprehensive troubleshooting guide...")
        
        issues = self.analyze_common_issues()
        
        guide_content = """# Enterprise AI Sentinel - Troubleshooting Guide

## Quick Diagnosis

### 1. Check System Status
```bash
# Check if all engines are running
python -c "from sentinel_core import SystemStatus; print(SystemStatus().get_engine_health())"

# Monitor system resources
python tools/system_monitor.py
```

### 2. Verify Dependencies
```bash
# Check installed packages
pip list | grep -E "(qiskit|tensorflow|torch|flask)"

# Validate quantum dependencies
python -c "import qiskit; print(f'Qiskit version: {qiskit.__version__}')"
```

## Common Issues and Solutions

"""

        for category, issue_list in issues.items():
            guide_content += f"\n### {category.replace('_', ' ').title()}\n\n"
            
            for issue in issue_list:
                guide_content += f"#### {issue['issue']}\n\n"
                guide_content += f"**Symptoms:** {issue['symptoms']}\n\n"
                guide_content += "**Possible Causes:**\n"
                for cause in issue['causes']:
                    guide_content += f"- {cause}\n"
                
                guide_content += "\n**Solutions:**\n"
                for solution in issue['solutions']:
                    guide_content += f"- {solution}\n"
                guide_content += "\n" + "-" * 50 + "\n\n"
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        guide_content += f"""
## Emergency Procedures

### Engine Failure Recovery
```python
# Restart failed engines
from sentinel_core import EngineManager

manager = EngineManager()
failed_engines = manager.get_failed_engines()

for engine in failed_engines:
    print("Restarting engine...")
    manager.restart_engine(engine)
```

### Database Recovery
```bash
# Backup current database
cp enterprise_sentinel_2025.db backup/db_backup.db

# Restore from backup
cp backup/db_backup.db enterprise_sentinel_2025.db
```

### Performance Troubleshooting
```python
# Monitor engine performance
from performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
slow_engines = monitor.identify_bottlenecks()

for engine_name, engine_metrics in slow_engines.items():
    print(f"Engine performance data available")
```

## Log Analysis

### Common Error Patterns
```bash
# Search for errors in logs
grep -i "error" logs/system.log
grep -i "exception" logs/system.log
grep -i "failed" logs/system.log

# Analyze recent issues
tail -100 logs/system.log | grep -E "(ERROR|CRITICAL)"
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test specific engine
from engines.ml_engine import MLEngine
engine = MLEngine()
engine.initialize(debug=True)
```

## Support Resources

- **Documentation:** docs/README.md
- **API Reference:** api/README.md
- **Performance Reports:** reports/
- **Issue Tracker:** GitHub Issues

---

*Last Updated: {current_time}*
""".format(current_time=current_time)
        
        guide_file = self.docs_dir / "troubleshooting" / "troubleshooting_guide.md"
        guide_file.write_text(guide_content, encoding='utf-8')
        
        print(f"Troubleshooting guide generated: {guide_file}")
        return guide_content
    
    def create_quick_reference(self):
        print("Creating quick reference card...")
        
        quick_ref = """# Quick Troubleshooting Reference

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
"""

        ref_file = self.docs_dir / "troubleshooting" / "quick_reference.md"
        ref_file.write_text(quick_ref, encoding='utf-8')
        
        print(f"Quick reference created: {ref_file}")

if __name__ == "__main__":
    generator = TroubleshootingGenerator(".")
    generator.generate_troubleshooting_guide()
    generator.create_quick_reference()
    print("Troubleshooting documentation completed!")

