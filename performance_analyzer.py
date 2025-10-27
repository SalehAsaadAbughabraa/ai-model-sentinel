
import json
from pathlib import Path
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "enterprise_sentinel_docs_v2"
    
    def analyze_engine_complexity(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            
            return {
                "total_lines": len(lines),
                "code_lines": len(code_lines),
                "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
                "complexity": "LOW" if len(code_lines) < 100 else "MEDIUM" if len(code_lines) < 500 else "HIGH"
            }
        except Exception as e:
            return {"total_lines": 0, "code_lines": 0, "comment_lines": 0, "complexity": "UNKNOWN"}
    
    def generate_performance_report(self):
        print("Generating performance and complexity analysis...")
        
        analysis_file = self.docs_dir / "project_analysis.json"
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        engines_data = []
        engine_files = analysis_data.get('sample_engine_files', [])[:15]
        
        for file_path in engine_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                complexity = self.analyze_engine_complexity(full_path)
                engines_data.append({
                    "name": Path(file_path).stem,
                    "file": file_path,
                    "complexity": complexity
                })
        
        performance_report = {
            "analysis_date": datetime.now().isoformat(),
            "engines_analyzed": len(engines_data),
            "complexity_distribution": {
                "HIGH": len([e for e in engines_data if e["complexity"]["complexity"] == "HIGH"]),
                "MEDIUM": len([e for e in engines_data if e["complexity"]["complexity"] == "MEDIUM"]),
                "LOW": len([e for e in engines_data if e["complexity"]["complexity"] == "LOW"])
            },
            "engines": engines_data,
            "recommendations": [
                "Monitor high-complexity engines for performance",
                "Consider refactoring engines with 500+ code lines",
                "Add caching for frequently used engines"
            ]
        }
        
        report_file = self.docs_dir / "reports" / "performance_analysis.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(performance_report, f, indent=2)
        
        print(f"Performance report generated: {report_file}")
        return performance_report
    
    def create_usage_examples(self):
        print("Creating usage examples...")
        
        examples_content = """# Enterprise AI Sentinel - Usage Examples

## Quick Start Examples

### 1. Basic Engine Initialization
```python
from enterprise_sentinel import SentinelCore

# Initialize the main system
sentinel = SentinelCore()
sentinel.initialize_engines()

# Access specific engines
ml_engine = sentinel.get_engine('MLEngine')
quantum_engine = sentinel.get_engine('QuantumEngine')
```

### 2. Machine Learning Operations
```python
# Train a model using ML Engine
training_data = load_training_dataset()
model_config = {
    'algorithm': 'random_forest',
    'parameters': {'n_estimators': 100}
}

model_id = ml_engine.train_model(
    data=training_data,
    config=model_config
)

# Make predictions
predictions = ml_engine.predict(model_id, new_data)
```

### 3. Quantum-Enhanced Operations
```python
# Use quantum engine for complex calculations
quantum_result = quantum_engine.solve_optimization(
    problem_matrix=problem_data,
    method='quantum_annealing'
)

# Quantum cryptographic operations
encrypted_data = quantum_engine.encrypt(
    data=sensitive_data,
    algorithm='quantum_key_distribution'
)
```

### 4. Security Monitoring
```python
# Monitor for threats
threat_report = security_engine.analyze_threats(
    system_logs=recent_logs,
    sensitivity='high'
)

if threat_report['threat_level'] > 0.7:
    security_engine.trigger_alert(threat_report)
```

### 5. Data Processing Pipeline
```python
# Process large datasets
data_engine = sentinel.get_engine('DataQualityEngine')

cleaned_data = data_engine.clean_dataset(
    raw_data=dataset,
    quality_threshold=0.95
)

analytics_result = data_engine.analyze_trends(cleaned_data)
```

## Integration Patterns

### REST API Integration
```python
import requests

# Call Sentinel API
response = requests.post(
    'http://localhost:8000/api/analyze',
    json={'data': input_data},
    headers={'Authorization': 'Bearer YOUR_TOKEN'}
)

analysis_result = response.json()
```

### Batch Processing
```python
# Process multiple items
def batch_process(items):
    results = []
    for item in items:
        result = ml_engine.process_item(item)
        results.append(result)
    return results
```

## Best Practices
1. Always initialize engines before use
2. Monitor engine health regularly
3. Use appropriate error handling
4. Implement proper logging
5. Follow security guidelines
"""

        examples_file = self.docs_dir / "usage_examples.md"
        examples_file.write_text(examples_content, encoding='utf-8')
        
        print(f"Usage examples created: {examples_file}")

if __name__ == "__main__":
    analyzer = PerformanceAnalyzer(".")
    analyzer.generate_performance_report()
    analyzer.create_usage_examples()
    print("Performance analysis and examples completed!")
