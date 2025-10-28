# dependency_analyzer.py
import ast
import json
from pathlib import Path
from collections import defaultdict

class DependencyAnalyzer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "enterprise_sentinel_docs_v2"
        self.dependencies = defaultdict(list)
    
    def analyze_imports(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            return imports
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []
    
    def find_internal_dependencies(self):
        print("Analyzing internal dependencies between engines...")
        
        analysis_file = self.docs_dir / "project_analysis.json"
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        engine_files = analysis_data.get('sample_engine_files', [])
        
        for file_path in engine_files[:20]:
            full_path = self.project_root / file_path
            if full_path.exists():
                engine_name = Path(file_path).stem
                imports = self.analyze_imports(full_path)
                
                internal_deps = []
                for imp in imports:
                    for other_engine in engine_files:
                        if Path(other_engine).stem in imp and Path(other_engine).stem != engine_name:
                            internal_deps.append(Path(other_engine).stem)
                
                if internal_deps:
                    self.dependencies[engine_name] = list(set(internal_deps))
        
        return self.dependencies
    
    def generate_dependency_graph(self):
        print("Generating dependency graph...")
        
        graph_content = "digraph EngineDependencies {\n"
        graph_content += "  rankdir=LR;\n  node [shape=box, style=filled, fillcolor=lightblue];\n"
        
        for engine, deps in self.dependencies.items():
            for dep in deps:
                graph_content += f'  "{engine}" -> "{dep}";\n'
        
        graph_content += "}"
        
        graph_file = self.docs_dir / "diagrams" / "engine_dependencies.dot"
        graph_file.write_text(graph_content, encoding='utf-8')
        
        print(f"Dependency graph generated: {graph_file}")
        return graph_content
    
    def update_engine_docs_with_deps(self):
        print("Updating engine documentation with dependencies...")
        
        for engine, deps in self.dependencies.items():
            doc_path = self.docs_dir / "engines" / f"{engine}.md"
            if doc_path.exists():
                content = doc_path.read_text(encoding='utf-8')
                
                if "## Dependencies" in content and "Dependencies will be analyzed automatically" in content:
                    deps_section = "## Dependencies\n\n### Internal Dependencies:\n"
                    if deps:
                        deps_section += "\n".join(f"- {dep}" for dep in deps) + "\n"
                    else:
                        deps_section += "- No internal dependencies detected\n"
                    
                    deps_section += "\n### External Dependencies:\n- To be analyzed in detail"
                    
                    updated_content = content.replace(
                        "## Dependencies\n- Dependencies will be analyzed automatically", 
                        deps_section
                    )
                    
                    doc_path.write_text(updated_content, encoding='utf-8')
                    print(f"Updated dependencies for: {engine}")
    
    def save_dependency_report(self):
        report = {
            "analysis_date": "2025-10-28",
            "total_engines_analyzed": len(self.dependencies),
            "dependencies_found": dict(self.dependencies),
            "most_connected_engines": sorted(
                self.dependencies.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            )[:5]
        }
        
        report_file = self.docs_dir / "reports" / "dependency_analysis.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"Dependency report saved: {report_file}")

if __name__ == "__main__":
    analyzer = DependencyAnalyzer(".")
    analyzer.find_internal_dependencies()
    analyzer.generate_dependency_graph()
    analyzer.update_engine_docs_with_deps()
    analyzer.save_dependency_report()
    print("Dependency analysis completed!")