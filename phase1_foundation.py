import os
import sys
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import platform
import psutil

class SecurityScannerConfig:
    def __init__(self):
        self.version = "1.0.0"
        self.codename = "PHOENIX-SHIELD"
        self.max_file_size = 100 * 1024 * 1024
        self.max_threads = os.cpu_count() or 4
        self.scan_timeout = 300
        
        self.supported_formats = {
            'executables': ['.exe', '.dll', '.so', '.bin', '.app', '.msi', '.sys', '.drv', '.efi'],
            'scripts': ['.py', '.js', '.php', '.sh', '.bat', '.ps1', '.vbs', '.asp', '.jsp'],
            'documents': ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.rtf'],
            'archives': ['.zip', '.rar', '.7z', '.tar', '.gz', '.jar', '.war', '.apk'],
            'system_files': ['.sys', '.drv', '.kext', '.efi', '.ocx', '.cpl'],
            'config_files': ['.ini', '.conf', '.cfg', '.xml', '.json', '.yml']
        }

class AdvancedLogger:
    def __init__(self, log_dir="security_scans"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        log_formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        self.logger = logging.getLogger('SecurityScanner')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            file_handler = logging.FileHandler(
                self.log_dir / f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_handler.setFormatter(log_formatter)
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

class FileAnalyzerEngine:
    def __init__(self):
        self.config = SecurityScannerConfig()
        self.logger = AdvancedLogger().logger
        
        self.file_signatures = {
            b'MZ': ('PE/EXE', 'windows_executable'),
            b'\x7fELF': ('ELF', 'linux_executable'),
            b'\xfe\xed\xfa': ('Mach-O', 'macos_executable'),
            b'PK\x03\x04': ('ZIP', 'archive'),
            b'%PDF': ('PDF', 'document'),
            b'{\\rtf': ('RTF', 'document'),
            b'<!DOCTYPE': ('HTML', 'web_file'),
            b'<?xml': ('XML', 'config_file')
        }

    def calculate_hashes(self, file_path):
        hashes = {}
        hash_algorithms = ['md5', 'sha1', 'sha256', 'sha512']
        
        try:
            for algo in hash_algorithms:
                hash_func = getattr(hashlib, algo)()
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_func.update(chunk)
                hashes[algo] = hash_func.hexdigest()
        except Exception as e:
            self.logger.error(f"Hash calculation failed: {e}")
            
        return hashes

    def get_file_metadata(self, file_path):
        try:
            file_stat = file_path.stat()
            
            metadata = {
                'path': str(file_path.resolve()),
                'filename': file_path.name,
                'size': file_stat.st_size,
                'created': file_stat.st_ctime,
                'modified': file_stat.st_mtime,
                'accessed': file_stat.st_atime,
                'permissions': oct(file_stat.st_mode),
                'file_type': self.detect_file_type(file_path),
                'hashes': self.calculate_hashes(file_path)
            }
            
            return metadata
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return None

    def detect_file_type(self, file_path):
        try:
            with open(file_path, 'rb') as f:
                header = f.read(20)
                
            for signature, (name, category) in self.file_signatures.items():
                if header.startswith(signature):
                    return {'name': name, 'category': category}
                    
            return {'name': 'Unknown', 'category': 'unknown'}
        except Exception as e:
            return {'name': f'Error: {e}', 'category': 'error'}

class SystemValidator:
    def __init__(self):
        self.logger = AdvancedLogger().logger
        
    def validate_environment(self):
        checks = {
            'python_version': self.check_python_version(),
            'platform_support': self.check_platform(),
            'disk_space': self.check_disk_space(),
            'memory_available': self.check_memory(),
            'required_modules': self.check_modules()
        }
        
        return all(checks.values()), checks

    def check_python_version(self):
        return sys.version_info >= (3, 8)

    def check_platform(self):
        current_platform = platform.system().lower()
        supported = ['windows', 'linux', 'darwin']
        return current_platform in supported

    def check_disk_space(self):
        try:
            free_gb = psutil.disk_usage('/').free / (1024**3)
            return free_gb > 1.0
        except:
            return True

    def check_memory(self):
        try:
            free_mb = psutil.virtual_memory().available / (1024**2)
            return free_mb > 512
        except:
            return True

    def check_modules(self):
        required = ['hashlib', 'json', 'logging', 'pathlib', 'threading']
        return all(module in sys.modules for module in required)

class CoreSecurityScanner:
    def __init__(self):
        self.config = SecurityScannerConfig()
        self.logger = AdvancedLogger().logger
        self.validator = SystemValidator()
        self.analyzer = FileAnalyzerEngine()
        
        self.scan_session = {
            'session_id': f"SCAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': datetime.now().isoformat(),
            'system_info': self.get_system_info(),
            'scan_results': []
        }

    def get_system_info(self):
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'total_memory': f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            'current_user': os.getenv('USERNAME') or os.getenv('USER')
        }

    def initialize(self):
        self.logger.info(f"Initializing {self.config.codename} v{self.config.version}")
        
        valid, details = self.validator.validate_environment()
        if not valid:
            self.logger.error("System validation failed")
            return False
            
        self.logger.info("System validation passed")
        self.logger.info(f"Available CPU cores: {self.config.max_threads}")
        return True

    def scan_file(self, file_path):
        if not file_path.exists():
            self.logger.warning(f"File not found: {file_path}")
            return None
            
        metadata = self.analyzer.get_file_metadata(file_path)
        if metadata:
            self.scan_session['scan_results'].append(metadata)
            
        return metadata

def main():
    scanner = CoreSecurityScanner()
    
    if not scanner.initialize():
        return 1
        
    test_file = Path(__file__)
    result = scanner.scan_file(test_file)
    
    if result:
        print(f"File: {result['filename']}")
        print(f"Size: {result['size']} bytes")
        print(f"SHA256: {result['hashes']['sha256']}")
        print(f"Type: {result['file_type']['name']}")
        
    scanner.scan_session['end_time'] = datetime.now().isoformat()
    
    session_file = Path("scan_session.json")
    with open(session_file, 'w') as f:
        json.dump(scanner.scan_session, f, indent=2)
        
    print(f"Scan session saved to: {session_file}")
    return 0

if __name__ == "__main__":
    sys.exit(main())