# api_documentor.py
import json
import re
from pathlib import Path

class APIDocumentor:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "enterprise_sentinel_docs_v2"
    
    def extract_api_endpoints(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            flask_patterns = [
                r"@app\.route\(['\"]([^'\"]+)['\"](?:,\s*methods=\[([^\]]+)\])?",
                r"@blueprint\.route\(['\"]([^'\"]+)['\"]",
                r"@api\.route\(['\"]([^'\"]+)['\"]"
            ]
            
            endpoints = []
            for pattern in flask_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    endpoint = match.group(1)
                    methods = match.group(2) if match.lastindex > 1 else "['GET']"
                    endpoints.append({
                        "path": endpoint,
                        "methods": methods,
                        "file": str(file_path)
                    })
            
            return endpoints
        except Exception as e:
            print(f"Error extracting APIs from {file_path}: {e}")
            return []
    
    def scan_for_apis(self):
        print("Scanning for API endpoints...")
        
        analysis_file = self.docs_dir / "project_analysis.json"
        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        
        all_endpoints = []
        python_files = analysis_data.get('sample_python_files', [])[:30]
        
        for file_path in python_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                endpoints = self.extract_api_endpoints(full_path)
                if endpoints:
                    all_endpoints.extend(endpoints)
                    print(f"Found {len(endpoints)} endpoints in {file_path}")
        
        return all_endpoints
    
    def generate_api_documentation(self, endpoints):
        print("Generating API documentation...")
        
        api_docs = {
            "openapi": "3.0.0",
            "info": {
                "title": "Enterprise AI Sentinel API",
                "version": "2.0.0",
                "description": "Automatically generated API documentation"
            },
            "paths": {}
        }
        
        for endpoint in endpoints:
            path = endpoint["path"]
            methods = eval(endpoint["methods"]) if endpoint["methods"] else ["GET"]
            
            for method in methods:
                method_lower = method.strip("'").lower()
                if path not in api_docs["paths"]:
                    api_docs["paths"][path] = {}
                
                api_docs["paths"][path][method_lower] = {
                    "summary": f"Auto-generated {method} endpoint",
                    "description": f"Located in: {endpoint['file']}",
                    "responses": {
                        "200": {
                            "description": "Successful operation"
                        }
                    }
                }
        
        api_file = self.docs_dir / "api" / "openapi.json"
        with open(api_file, 'w', encoding='utf-8') as f:
            json.dump(api_docs, f, indent=2)
        
        print(f"API documentation generated: {api_file}")
        return api_docs
    
    def create_api_readme(self, endpoints):
        print("Creating API overview...")
        
        readme_content = """# Enterprise AI Sentinel - API Documentation

## Overview
Automatically generated API documentation for all detected endpoints.

## Endpoints
"""
        
        for endpoint in endpoints:
            methods = eval(endpoint["methods"]) if endpoint["methods"] else ["GET"]
            readme_content += f"""
### {endpoint['path']}
- **Methods:** {', '.join(methods)}
- **File:** `{endpoint['file']}`
"""
        
        readme_file = self.docs_dir / "api" / "README.md"
        readme_file.write_text(readme_content, encoding='utf-8')
        print(f"API README created: {readme_file}")

if __name__ == "__main__":
    documentor = APIDocumentor(".")
    endpoints = documentor.scan_for_apis()
    
    if endpoints:
        documentor.generate_api_documentation(endpoints)
        documentor.create_api_readme(endpoints)
        print(f"API documentation completed! Found {len(endpoints)} endpoints")
    else:
        print("No API endpoints found. Creating placeholder documentation...")
        
        placeholder_api = {
            "openapi": "3.0.0",
            "info": {
                "title": "Enterprise AI Sentinel API",
                "version": "2.0.0",
                "description": "API endpoints will be automatically detected"
            },
            "paths": {
                "/api/engines": {
                    "get": {
                        "summary": "List all engines",
                        "responses": {"200": {"description": "List of engines"}}
                    }
                }
            }
        }
        
        api_file = documentor.docs_dir / "api" / "openapi.json"
        with open(api_file, 'w', encoding='utf-8') as f:
            json.dump(placeholder_api, f, indent=2)
        
        print("Placeholder API documentation created")