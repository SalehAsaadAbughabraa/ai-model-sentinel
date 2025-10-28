from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ai-model-sentinel",
    version="2.0.1",
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
