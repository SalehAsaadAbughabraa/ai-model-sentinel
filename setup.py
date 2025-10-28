from setuptools import setup, find_packages 
 
setup( 
    name=\"ai-model-sentinel\", 
    version=\"2.0.0\", 
    packages=find_packages(where=\"src\"), 
    package_dir={\"\": \"src\"}, 
    install_requires=[ 
        \"flask>=2.3.3\", 
        \"waitress>=2.1.2\", 
        \"cryptography>=41.0.3\", 
        \"numpy>=1.24.3\", 
        \"torch>=2.0.1\", 
        \"duckdb>=0.8.1\" 
    ], 
    python_requires=\">=3.8\", 
) 
