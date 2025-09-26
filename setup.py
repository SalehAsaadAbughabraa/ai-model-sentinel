from setuptools import setup, find_packages

setup(
    name="ai-security-scanner",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'pefile',
        'lief',
        'scikit-learn',
        'numpy',
        'psutil'
    ],
    entry_points={
        'console_scripts': [
            'aiscan=ai_security_scanner:main',
        ],
    },
    author="Security Team",
    description="AI-Powered Military-Grade Security Scanner",
    python_requires='>=3.8',
)