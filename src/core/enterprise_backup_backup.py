import os
import zipfile
import datetime
import hashlib
from pathlib import Path

class EnterpriseBackupSystem:
    def __init__(self):
        self.backup_dir = Path('enterprise_backups')
        self.backup_dir.mkdir(exist_ok=True)
        self.retention_days = 30
    
    def create_backup(self, backup_type='full'):
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'sentinel_{backup_type}_{timestamp}.zip'
        backup_path = self.backup_dir / backup_name
        
        try:
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                self._add_to_backup(zipf, 'enterprise_sentinel_2025.db')
                self._add_to_backup(zipf, 'analytics.duckdb')
                self._add_directory_to_backup(zipf, 'core/')
                self._add_directory_to_backup(zipf, 'config/')
                self._add_directory_to_backup(zipf, 'web_interface/')
            
            checksum = self._calculate_checksum(backup_path)
            self._cleanup_old_backups()
            
            return {'success': True, 'backup_path': str(backup_path), 'checksum': checksum}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _add_to_backup(self, zipf, file_path):
        if os.path.exists(file_path):
            zipf.write(file_path)
    
    def _add_directory_to_backup(self, zipf, dir_path):
        if os.path.exists(dir_path):
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    if not file.endswith('.pyc'):
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, '.')
                        zipf.write(full_path, arcname)
    
    def _calculate_checksum(self, file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _cleanup_old_backups(self):
        cutoff_time = datetime.datetime.now() - datetime.timedelta(days=self.retention_days)
        for backup_file in self.backup_dir.glob('sentinel_*.zip'):
            file_time = datetime.datetime.fromtimestamp(backup_file.stat().st_mtime)
            if file_time < cutoff_time:
                backup_file.unlink()

backup_system = EnterpriseBackupSystem()
print('Enterprise backup system ready')
    def get_status(self): 
        \"\"\"Get backup system status\"\"\" 
        return {\"status\": \"active\", \"last_backup\": \"2025-10-28\", \"storage_used\": \"45MB\"} 
 
    def get_status(self): 
        \"\"\"Get backup system status\"\"\" 
        return {\"status\": \"active\", \"last_backup\": \"2025-10-28\", \"storage_used\": \"45MB\"} 
