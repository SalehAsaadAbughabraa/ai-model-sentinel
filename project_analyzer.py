# project_analyzer.py
import os
import json
from pathlib import Path

class ProjectAnalyzer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.analysis_result = {}
    
    def analyze_project_structure(self):
        print("Analyzing project structure...")
        
        # Find all Python files
        python_files = list(self.project_root.rglob("*.py"))
        print(f"Found {len(python_files)} Python files")
        
        # Find configuration files
        config_files = list(self.project_root.rglob("*.json")) + list(self.project_root.rglob("*.yaml")) + list(self.project_root.rglob("*.yml"))
        print(f"Found {len(config_files)} configuration files")
        
        # Find SQLite databases
        db_files = list(self.project_root.rglob("*.db")) + list(self.project_root.rglob("*.sqlite"))
        print(f"Found {len(db_files)} database files")
        
        # Look for engine-related files
        engine_files = []
        engine_keywords = ["engine", "ml", "quantum", "security", "analytics", "monitor", "fusion"]
        
        for py_file in python_files:
            file_str = str(py_file).lower()
            if any(keyword in file_str for keyword in engine_keywords):
                engine_files.append(py_file)
        
        print(f"Found {len(engine_files)} engine-related Python files")
        
        self.analysis_result = {
            "total_python_files": len(python_files),
            "engine_files_count": len(engine_files),
            "config_files_count": len(config_files),
            "database_files_count": len(db_files),
            "sample_python_files": [str(f.relative_to(self.project_root)) for f in python_files[:15]],
            "sample_engine_files": [str(f.relative_to(self.project_root)) for f in engine_files[:10]],
            "database_files": [str(f.relative_to(self.project_root)) for f in db_files]
        }
        
        return self
    
    def generate_structure_report(self):
        report_path = self.project_root / "enterprise_sentinel_docs_v2" / "project_analysis.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_result, f, indent=2, ensure_ascii=False)
        
        print(f"Project analysis saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("PROJECT ANALYSIS SUMMARY")
        print("="*50)
        print(f"📁 Total Python files: {self.analysis_result['total_python_files']}")
        print(f"🔧 Engine-related files: {self.analysis_result['engine_files_count']}")
        print(f"⚙️ Configuration files: {self.analysis_result['config_files_count']}")
        print(f"💾 Database files: {self.analysis_result['database_files_count']}")
        print("="*50)
        
        return self.analysis_result

if __name__ == "__main__":
    analyzer = ProjectAnalyzer(".")
    analysis = analyzer.analyze_project_structure().generate_structure_report()