from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt") as f:
    dev_requirements = f.read().splitlines()

setup(
    name="ai-sentinel",
    version="1.0.0",
    description="Enterprise AI Model Security Scanner",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AI Security Team",
    author_email="security@company.com",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "prod": open("requirements-prod.txt").read().splitlines(),
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "ai-sentinel=ai_sentinel.cli:main",
        ],
    },
)