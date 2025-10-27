# setup_documentation.py
import shutil
from pathlib import Path
from datetime import datetime

class DocumentationInitializer:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.new_docs_dir = self.project_root / "enterprise_sentinel_docs_v2"
        self.backup_dir = self.project_root / f"docs_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def create_backup(self):
        print("Creating backup of existing documentation...")
        
        old_docs_paths = [
            self.project_root / "docs",
            self.project_root / "documentation", 
            self.project_root / "Documentation"
        ]
        
        for old_path in old_docs_paths:
            if old_path.exists():
                target_dir = self.backup_dir / old_path.name
                shutil.copytree(old_path, target_dir)
                print(f"Backed up: {old_path} -> {target_dir}")
        
        return self
    
    def cleanup_old_docs(self):
        print("Cleaning up old documentation directories...")
        
        docs_to_remove = [
            self.project_root / "docs",
            self.project_root / "documentation",
            self.project_root / "Documentation",
            self.project_root / "build",
            self.project_root / "_build"
        ]
        
        for doc_path in docs_to_remove:
            if doc_path.exists():
                shutil.rmtree(doc_path)
                print(f"Removed: {doc_path}")
        
        return self
    
    def create_new_structure(self):
        print("Creating new documentation structure...")
        
        directories = [
            "engines",
            "api", 
            "security",
            "deployment",
            "troubleshooting",
            "diagrams",
            "reports"
        ]
        
        for directory in directories:
            (self.new_docs_dir / directory).mkdir(parents=True, exist_ok=True)
        
        print(f"New documentation structure created at: {self.new_docs_dir}")
        return self

if __name__ == "__main__":
    project_path = input("Enter project root path: ").strip()
    initializer = DocumentationInitializer(project_path)
    initializer.create_backup().cleanup_old_docs().create_new_structure()