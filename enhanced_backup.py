import os, zipfile, datetime, hashlib
class EnhancedBackupSystem:
    def __init__(self):
        self.backup_dir = "backups"
        os.makedirs(self.backup_dir, exist_ok=True)
    def create_backup(self):
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"{self.backup_dir}/sentinel_full_{timestamp}.zip"
            critical_files = ["enterprise_sentinel_2025.db", "core/", "config/", "web_interface/", "quantum_stubs.py", "universal_pbkdf2.py", "database_connector_fixed.py"]
            with zipfile.ZipFile(backup_file, "w", zipfile.ZIP_DEFLATED) as zipf:
                for item in critical_files:
                    if os.path.exists(item):
                        if os.path.isdir(item):
                            for root, dirs, files in os.walk(item):
                                for file in files:
                                    if not file.endswith(".pyc"):
                                        full_path = os.path.join(root, file)
                                        arcname = os.path.relpath(full_path, ".")
                                        zipf.write(full_path, arcname)
                        else:
                            zipf.write(item)
            with open(backup_file, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            print(f"Backup created: {backup_file}")
            print(f"SHA256: {file_hash}")
            return True
        except Exception as e:
            print(f"Backup failed: {e}")
            return False
backup_system = EnhancedBackupSystem()
print("Enhanced backup system ready")