# Testing Guide

This document explains how to test AI Model Sentinel effectively.

## 🧪 Basic Testing

### Quick Smoke Test
```bash
# Test basic functionality
python military_scanner.py --help

# Test with safe file
python military_scanner.py safe_test.py

# Test with dangerous file  
python military_scanner.py dangerous_test.py
Performance Testing
bash
# Measure scan time
python -c "
import time
from military_scanner import AdvancedMilitaryScanner
scanner = AdvancedMilitaryScanner()
start = time.time()
result = scanner.scan_file('safe_test.py')
print(f'Scan time: {time.time()-start:.3f}s')
"
📊 Test Cases
File Type Testing
Python scripts (.py)

AI models (.pkl, .h5)

Binary files

Text files

Large files (>1MB)

Threat Level Testing
CLEAN (0.0-0.2) - safe_test.py

LOW (0.2-0.4) - low_risk.py

MEDIUM (0.4-0.6) - medium_risk.py

HIGH (0.6-0.8) - high_risk.py

CRITICAL (0.8-1.0) - critical_risk.py

🐛 Bug Reporting Template
When reporting bugs, please include:

markdown
## Description
[What happened]

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [See error]

## Expected Behavior
[What should have happened]

## Actual Behavior
[What actually happened]

## Environment
- OS: [e.g. Windows 11]
- Python: [e.g. 3.13]
- Version: [e.g. 1.0.0]

## Additional Context
[Logs, screenshots, etc.]
🔧 Development Testing
Adding New Tests
Create test file in appropriate category

Verify threat level detection

Test performance impact

Update this document

Test Automation
bash
# Basic test script
python -c "
from military_scanner import AdvancedMilitaryScanner
scanner = AdvancedMilitaryScanner()

test_files = ['safe_test.py', 'dangerous_test.py']
for file in test_files:
    result = scanner.scan_file(file)
    print(f'{file}: {result[\\\"threat_level_display\\\"]}')
"