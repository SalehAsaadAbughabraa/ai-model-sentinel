# AI Model Sentinel 🔒

[![Version](https://img.shields.io/badge/version-0.1.3-blue.svg)](https://pypi.org/project/ai-model-sentinel/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ai-model-sentinel.svg)](https://pypi.org/project/ai-model-sentinel/)

Enterprise-grade security framework for protecting AI models against sophisticated threats, inference attacks, and data extraction attempts.

 Overview
AI Model Sentinel is a comprehensive security framework designed to protect machine learning models from various threats including model inversion attacks, membership inference attacks, adversarial examples, and data extraction attempts.

 Key Features

 Advanced Protection Mechanisms
- AI-Powered monitoring and drift detection
- Real-time threat detection and analysis
- Comprehensive model scanning capabilities
- Automated reporting and dashboard

 Supported Model Formats
- TensorFlow (.h5, .pb, .savedmodel)
- PyTorch (.pt, .pth) 
- ONNX (.onnx)
- Scikit-learn (.pkl, .joblib)
- Keras (.keras)

 Enterprise Ready
- Production-grade monitoring
- Comprehensive management interface
- RESTful API support

 Quick Start

 Installation
```bash
pip install ai-model-sentinel==0.1.3

Important Usage Note
Due to a temporary CLI issue in v0.1.3, please use Python import method:

python
# Scan for AI models
python -c "from ai_model_sentinel.cli import scan; import sys; sys.argv = ['scan', '--verbose']; scan()"

# Generate detailed report
python -c "from ai_model_sentinel.cli import report; import sys; sys.argv = ['report', '--output', 'scan_report.html']; report()"

# Monitor models in production
python -c "from ai_model_sentinel.cli import monitor; import sys; sys.argv = ['monitor', '--path', '.', '--interval', '60']; monitor()"
Expected Output
text
🔍 Scanning path: .
📊 Scan Results:
   Found 2 AI models
   Scanned 5 directories
   Duration: 1200ms

📁 Models found:
   1. ./model.h5 (h5)
   2. ./model.pb (pb)
Complete Usage Examples
Basic Scanning
python
from ai_model_sentinel.cli import scan
import sys

 Basic scan
sys.argv = ['scan']
scan()

 Deep scan with verbose output
sys.argv = ['scan', '--deep', '--verbose']
scan()
Advanced Monitoring
python
from ai_model_sentinel.cli import monitor
import sys

 Monitor with custom interval
sys.argv = ['monitor', '--path', '/path/to/models', '--interval', '300']
monitor()
Report Generation
python
from ai_model_sentinel.cli import report
import sys

 HTML report
sys.argv = ['report', '--output', 'security_report.html', '--format', 'html']
report()

 JSON report
sys.argv = ['report', '--output', 'analysis.json', '--format', 'json']
report()
Version Information
Current Version: 0.1.3
Complete package structure reorganization

Enhanced CLI with full command support

Fixed import issues and module paths

Improved file generation system

Added verbose and deep scan options

Known Issues
CLI commands require Python import method (fixed in upcoming v0.1.4)

Temporary workaround provided above

Architecture
text
ai-model-sentinel/
├── core/                 # Core scanning and detection
├── dashboard/           # Web interface components
├── simulation/          # Performance analysis
├── utils/               # Utility functions
└── cli.py              # Command line interface
API Reference
Core Functions
scan(): Comprehensive model scanning

monitor(): Continuous model monitoring

report(): Detailed report generation

dashboard(): Web-based management interface

Configuration Options
--path: Target directory for scanning

--deep: Enable deep scanning mode

--verbose: Detailed output information

--format: Report output format (html/json)

Contributing
We welcome contributions from the community:

Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

Support
GitHub Issues: Bug reports and feature requests

Documentation: Usage examples and API reference

Community: Collaborative development and support

License
MIT License - see LICENSE file for complete details.

Acknowledgments
Open source community contributions

AI security research advancements

Early adopters and testers

Note: This is a beta release. CLI functionality will be fully restored in version 0.1.4. Current workaround provides complete functionality via Python imports.

