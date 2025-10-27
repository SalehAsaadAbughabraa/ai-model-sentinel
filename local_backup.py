import os
import shutil
import datetime
import zipfile
from pathlib import Path

class LocalBackupSystem:
    def __init__(self):
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"sentinel_backup_{timestamp}.zip"
            
            # Backup critical files
            files_to_backup = [
                "enterprise_sentinel_2025.db",
                "core/",
                "config/",
                "web_interface/",
                "requirements-production.txt"
            ]
            
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for item in files_to_backup:
                    if os.path.exists(item):
                        if os.path.isdir(item):
                            for root, dirs, files in os.walk(item):
                                for file in files:
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, os.path.dirname(item))
                                    zipf.write(file_path, arcname)
                        else:
                            zipf.write(item)
            
            print(f"✅ Backup created: {backup_path}")
            return True
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False

# Create global instance
backup_system = LocalBackupSystem()
