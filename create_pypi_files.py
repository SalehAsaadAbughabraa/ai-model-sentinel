from pathlib import Path
import json

class PyPIPreparer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        
    def create_setup_py(self):
        setup_content = '''from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-model-sentinel",
    version="2.0.0",
    author="Saleh Asaad Abughabr",
    author_email="saleh87alally@gmail.com",
    description="Enterprise AI Model Monitoring and Security System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SalehAsaadAbughabraa/ai-model-sentinel",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "sqlite3",
    ],
    extras_require={
        "quantum": ["qiskit>=0.34.0"],
        "security": ["cryptography>=3.4.0"],
        "ml": ["tensorflow>=2.6.0", "torch>=1.9.0"],
    },
    entry_points={
        "console_scripts": [
            "ai-sentinel=app.main:main",
        ],
    },
)
'''
        setup_file = self.project_root / "setup.py"
        setup_file.write_text(setup_content, encoding='utf-8')
        print("✅ setup.py created")
    
    def create_setup_cfg(self):
        setup_cfg_content = '''[metadata]
name = ai-model-sentinel
version = 2.0.0
author = Saleh Asaad Abughabr
author_email = saleh87alally@gmail.com
description = Enterprise AI Model Monitoring and Security System
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/SalehAsaadAbughabraa/ai-model-sentinel
classifiers =
    Development Status :: 4 - Beta
    Intended Audience :: Developers
    License :: OSI Approved :: MIT License
    Operating System :: OS Independent
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11

[options]
packages = find:
python_requires = >=3.8
install_requires =
    numpy>=1.21.0
    pandas>=1.3.0
    scikit-learn>=1.0.0

[options.extras_require]
quantum = qiskit>=0.34.0
security = cryptography>=3.4.0
ml = tensorflow>=2.6.0; torch>=1.9.0

[options.entry_points]
console_scripts =
    ai-sentinel = app.main:main
'''
        setup_cfg_file = self.project_root / "setup.cfg"
        setup_cfg_file.write_text(setup_cfg_content, encoding='utf-8')
        print("✅ setup.cfg created")
    
    def create_pyproject_toml(self):
        pyproject_content = '''[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ai-model-sentinel"
version = "2.0.0"
description = "Enterprise AI Model Monitoring and Security System"
authors = [
    {name = "Saleh Asaad Abughabr", email = "saleh87alally@gmail.com"}
]
readme = {file = "README.md", content-type = "text/markdown"}
requires-python = ">=3.8"
keywords = ["ai", "machine-learning", "security", "monitoring", "quantum"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

dependencies = [
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
]

[project.optional-dependencies]
quantum = ["qiskit>=0.34.0"]
security = ["cryptography>=3.4.0"]
ml = ["tensorflow>=2.6.0", "torch>=1.9.0"]

[project.urls]
Homepage = "https://github.com/SalehAsaadAbughabraa/ai-model-sentinel"
Documentation = "https://github.com/SalehAsaadAbughabraa/ai-model-sentinel#readme"

[project.scripts]
ai-sentinel = "app.main:main"
'''
        pyproject_file = self.project_root / "pyproject.toml"
        pyproject_file.write_text(pyproject_content, encoding='utf-8')
        print("✅ pyproject.toml created")
    
    def create_license(self):
        license_content = '''MIT License

Copyright (c) 2025 Saleh Asaad Abughabr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
        license_file = self.project_root / "LICENSE"
        license_file.write_text(license_content, encoding='utf-8')
        print("✅ LICENSE file created")
    
    def create_readme(self):
        readme_content = '''# AI Model Sentinel v2.0.0

Enterprise AI Model Monitoring, Security, and Management System

## Features

- 🔍 **AI Model Monitoring**: Real-time monitoring of ML models
- 🔒 **Security Engine**: Threat detection and analysis  
- ⚛️ **Quantum Integration**: Quantum-enhanced algorithms
- 📊 **Performance Analytics**: Comprehensive model analytics
- 🛡️ **Enterprise Security**: Advanced security protocols

## Installation

```bash
# Basic installation
pip install ai-model-sentinel

# With quantum features
pip install ai-model-sentinel[quantum]

# With security features  
pip install ai-model-sentinel[security]

# With ML features
pip install ai-model-sentinel[ml]
Quick Start
python
from ai_model_sentinel import SentinelCore

# Initialize the system
sentinel = SentinelCore()
sentinel.initialize_engines()

# Access engines
ml_engine = sentinel.get_engine('MLEngine')
security_engine = sentinel.get_engine('SecurityEngine')
Documentation
Comprehensive documentation available in the enterprise_sentinel_docs_v2/ directory.

Requirements
Python 3.8+

8GB RAM recommended

SQLite database

License
MIT License - See LICENSE file for details.

Developer
Saleh Asaad Abughabr
Email: saleh87alally@gmail.com
GitHub: SalehAsaadAbughabraa
'''
readme_file = self.project_root / "README.md"
readme_file.write_text(readme_content, encoding='utf-8')
print("✅ README.md created")

text
def create_init_files(self):
    # إنشاء __init__.py في المجلدات الرئيسية
    directories = ["app", "engines", "tools"]
    
    for dir_name in directories:
        init_file = self.project_root / dir_name / "__init__.py"
        if not init_file.exists():
            init_file.write_text('''"""
AI Model Sentinel - {0} Module
Version 2.0.0
Developer: Saleh Asaad Abughabr
""".format(dir_name.title())
''', encoding='utf-8')
print(f"✅ init.py created in {dir_name}")

if name == "main":
preparer = PyPIPreparer(".")
preparer.create_setup_py()
preparer.create_setup_cfg()
preparer.create_pyproject_toml()
preparer.create_license()
preparer.create_readme()
preparer.create_init_files()
print("🎉 PyPI preparation completed!")