# advanced_analyzer.py
import json
import ast
from pathlib import Path

class EngineAnalyzer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "enterprise_sentinel_docs_v2"
    
    def analyze_engine_structure(self, engine_path):
        try:
            with open(engine_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            return {
                "functions": functions[:10],
                "classes": classes[:5],
                "imports": imports[:15]
            }
        except:
            return {"functions": [], "classes": [], "imports": []}
    
    def generate_detailed_docs(self):
        print("Generating advanced engine analysis...")
        
        analysis_file = self.docs_dir / "project_analysis.json"
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        engine_files = analysis_data.get('sample_engine_files', [])[:5]
        
        for file_path in engine_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                engine_name = Path(file_path).stem
                analysis = self.analyze_engine_structure(full_path)
                
                doc_path = self.docs_dir / "engines" / f"{engine_name}.md"
                if doc_path.exists():
                    current_content = doc_path.read_text(encoding='utf-8')
                    
                    new_section = f"""
## Code Analysis

### Classes Found:
{chr(10).join(f"- {cls}" for cls in analysis['classes'])}

### Main Functions:
{chr(10).join(f"- {func}" for func in analysis['functions'])}

### Key Imports:
{chr(10).join(f"- {imp}" for imp in analysis['imports'])}
"""
                    
                    updated_content = current_content.replace(
                        "## Troubleshooting", 
                        new_section + "\n## Troubleshooting"
                    )
                    
                    doc_path.write_text(updated_content, encoding='utf-8')
                    print(f"Updated documentation for: {engine_name}")
        
        print("Advanced analysis completed!")

if __name__ == "__main__":
    analyzer = EngineAnalyzer(".")
    analyzer.generate_detailed_docs()