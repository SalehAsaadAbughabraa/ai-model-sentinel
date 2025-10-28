# fix_pypi_setup.py
from pathlib import Path

class PyPIFixer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
    
    def fix_pyproject_toml(self):
        print("Fixing pyproject.toml...")
        
        pyproject_content = '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
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

[project.scripts]
ai-sentinel = "app.main:main"
'''
        
        pyproject_file = self.project_root / "pyproject.toml"
        pyproject_file.write_text(pyproject_content, encoding='utf-8')
        print("✅ pyproject.toml fixed")
    
    def create_simple_setup(self):
        print("Creating simple setup.py...")
        
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
    packages=find_packages(include=["app", "engines", "tools"]),
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
    ],
    extras_require={
        "quantum": ["qiskit>=0.34.0"],
        "security": ["cryptography>=3.4.0"],
        "ml": ["tensorflow>=2.6.0", "torch>=1.9.0"],
    },
)
'''
        
        setup_file = self.project_root / "setup.py"
        setup_file.write_text(setup_content, encoding='utf-8')
        print("✅ setup.py created")
    
    def create_manifest(self):
        print("Creating MANIFEST.in...")
        
        manifest_content = '''include README.md
include LICENSE
include pyproject.toml
recursive-include app *.py
recursive-include engines *.py
recursive-include tools *.py
recursive-include enterprise_sentinel_docs_v2 *.md *.json
'''
        
        manifest_file = self.project_root / "MANIFEST.in"
        manifest_file.write_text(manifest_content, encoding='utf-8')
        print("✅ MANIFEST.in created")
    
    def create_init_py(self, directory):
        init_file = self.project_root / directory / "__init__.py"
        if not init_file.exists():
            init_file.write_text('''"""
AI Model Sentinel - {} Module
Version 2.0.0
Developer: Saleh Asaad Abughabr
"""
'''.format(directory.title()), encoding='utf-8')
            print(f"✅ __init__.py created in {directory}")
    
    def setup_package_structure(self):
        print("Setting up package structure...")
        
        # إنشاء ملفات __init__.py في المجلدات الرئيسية
        directories = ["app", "engines", "tools"]
        for dir_name in directories:
            self.create_init_py(dir_name)
        
        # التأكد من وجود مجلدات Python package
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir()
                self.create_init_py(dir_name)

if __name__ == "__main__":
    fixer = PyPIFixer(".")
    fixer.fix_pyproject_toml()
    fixer.create_simple_setup()
    fixer.create_manifest()
    fixer.setup_package_structure()
    print("🎉 PyPI setup fixed!")