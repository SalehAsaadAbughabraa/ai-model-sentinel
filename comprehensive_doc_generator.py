import json
import subprocess
from pathlib import Path
from datetime import datetime

class EnterpriseDocGenerator:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "enterprise_sentinel_docs_v2"
        self.engines_data = []
        
    def load_analysis_data(self):
        analysis_file = self.docs_dir / "project_analysis.json"
        with open(analysis_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def detect_engines_automatically(self):
        print("Automatically detecting and categorizing engines...")
        
        analysis = self.load_analysis_data()
        engine_files = analysis.get('sample_engine_files', [])
        
        detected_engines = []
        for file_path in engine_files[:100]:
            full_path = self.project_root / file_path
            if full_path.exists():
                engine_name = Path(file_path).stem
                detected_engines.append({
                    "name": engine_name,
                    "path": file_path,
                    "full_path": str(full_path),
                    "category": self._categorize_engine(engine_name),
                    "size_kb": full_path.stat().st_size / 1024
                })
        
        self.engines_data = detected_engines
        print(f"Detected {len(self.engines_data)} engines for documentation")
        return self
    
    def _categorize_engine(self, engine_name):
        name_lower = engine_name.lower()
        if "quantum" in name_lower:
            return "Quantum"
        elif "ml" in name_lower or "ai" in name_lower or "model" in name_lower:
            return "AI/ML"
        elif "security" in name_lower or "threat" in name_lower or "audit" in name_lower:
            return "Security"
        elif "data" in name_lower or "database" in name_lower or "analytics" in name_lower:
            return "Data"
        elif "fusion" in name_lower or "monitor" in name_lower:
            return "Fusion/Monitoring"
        else:
            return "Other"
    
    def generate_engine_documentation(self):
        print("Generating engine documentation...")
        
        template = """# {name}

**Category:** {category}  
**Path:** `{path}`  
**File Size:** {size_kb:.1f} KB  
**Status:** Active  
**Last Updated:** {timestamp}

## Overview
Automated documentation for {name} engine.

## File Location
{path}

## Dependencies
- Dependencies will be analyzed automatically

## API Endpoints
- API endpoints to be documented

## Usage Examples
# Example usage will be generated

## Troubleshooting
Common issues and solutions will be documented here."""
        
        for engine in self.engines_data:
            engine_doc = template.format(
                name=engine["name"],
                category=engine["category"],
                path=engine["path"],
                size_kb=engine["size_kb"],
                timestamp=datetime.now().strftime("%Y-%m-%d")
            )
            
            doc_path = self.docs_dir / "engines" / f"{engine['name']}.md"
            doc_path.write_text(engine_doc, encoding='utf-8')
        
        print(f"Generated documentation for {len(self.engines_data)} engines")
        return self
    
    def generate_main_readme(self):
        print("Generating main README...")
        
        categories = {}
        for engine in self.engines_data:
            cat = engine["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(engine)
        
        readme_content = f"""# Enterprise AI Sentinel v2.0 - Comprehensive Documentation

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Total Engines Documented:** {len(self.engines_data)}

## Quick Navigation

### Engine Categories:"""
        
        for category, engines in categories.items():
            readme_content += f"\n#### {category} ({len(engines)} engines)\n"
            for engine in engines:
                readme_content += f"- [{engine['name']}](engines/{engine['name']}.md)\n"
        
        readme_path = self.docs_dir / "README.md"
        readme_path.write_text(readme_content, encoding='utf-8')
        print("Main README generated")
        return self

if __name__ == "__main__":
    generator = EnterpriseDocGenerator(".")
    (generator
     .detect_engines_automatically()
     .generate_engine_documentation()
     .generate_main_readme())
    
    print("Documentation generation completed!")
    print(f"Documentation location: {generator.docs_dir}")