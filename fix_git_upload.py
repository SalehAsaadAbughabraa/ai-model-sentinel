# fix_git_upload.py
import subprocess
from pathlib import Path

class GitUploadFix:
    def __init__(self, project_root):
        self.project_root = Path(project_root)
    
    def create_gitignore(self):
        gitignore_content = """# Database files
*.db
*.sqlite
*.sqlite3
*.duckdb
analytics.duckdb
enterprise_sentinel_2025.db

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Logs
*.log
logs/

# Documentation build
_build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
"""
        
        gitignore_file = self.project_root / ".gitignore"
        gitignore_file.write_text(gitignore_content, encoding='utf-8')
        print("✅ .gitignore file created")
    
    def add_files_safely(self):
        print("Adding files to Git safely...")
        
        # أولاً أضف .gitignore
        subprocess.run(["git", "add", ".gitignore"], cwd=self.project_root)
        
        # أضف الملفات تدريجياً باستثناء قواعد البيانات
        commands = [
            ["git", "add", "*.py"],
            ["git", "add", "*.md"],
            ["git", "add", "*.json"],
            ["git", "add", "*.txt"],
            ["git", "add", "enterprise_sentinel_docs_v2/"],
            ["git", "add", "app/"],
            ["git", "add", "engines/"],
            ["git", "add", "tools/"]
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                if result.returncode == 0:
                    print(f"✅ Added: {cmd[2]}")
                else:
                    print(f"⚠️ Could not add: {cmd[2]} - {result.stderr}")
            except Exception as e:
                print(f"❌ Error adding {cmd[2]}: {e}")
    
    def commit_and_push(self):
        repo_url = "https://github.com/SalehAsaadAbughabraa/ai-model-sentinel.git"
        
        commands = [
            ['git', 'commit', '-m', 'AI Model Sentinel v2.0.0 - Complete system with comprehensive documentation by Saleh Asaad Abughabr'],
            ['git', 'branch', '-M', 'main'],
            ['git', 'remote', 'add', 'origin', repo_url],
            ['git', 'push', '-u', 'origin', 'main']
        ]
        
        for cmd in commands:
            print(f"Executing: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
                if result.returncode == 0:
                    print(f"✅ Success: {cmd[0]}")
                else:
                    print(f"❌ Failed: {cmd[0]} - {result.stderr}")
            except Exception as e:
                print(f"❌ Error: {cmd[0]} - {e}")

if __name__ == "__main__":
    fixer = GitUploadFix(".")
    fixer.create_gitignore()
    fixer.add_files_safely()
    fixer.commit_and_push()