import os
import sys
import hashlib
import struct
import secrets
from pathlib import Path
from datetime import datetime
import numpy as np
import zlib
import warnings
from enum import Enum

warnings.filterwarnings('ignore')

try:
    import yara
    YARA_AVAILABLE = True
except ImportError:
    YARA_AVAILABLE = False
    print("⚠️  YARA not available - signature detection limited")

try:
    import pefile
    PEFILE_AVAILABLE = True
except ImportError:
    PEFILE_AVAILABLE = False
    print("⚠️  pefile not available - PE analysis limited")

class ThreatLevel(Enum):
    CLEAN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class QuantumCryptographicEngine:
    def __init__(self):
        self.quantum_seed = secrets.token_bytes(32)
    
    def generate_quantum_hash(self, data):
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        hash_layers = []
        for i in range(3):
            layer_seed = self.quantum_seed + struct.pack('>I', i)
            hmac_layer = hashlib.pbkdf2_hmac('sha256', data, layer_seed, 1000)
            hash_layers.append(hmac_layer)
        
        final_hash = hashlib.sha3_256(b''.join(hash_layers))
        return final_hash.hexdigest()
    
    def advanced_entropy_analysis(self, data):
        if len(data) == 0:
            return {'shannon': 0.0, 'compression': 0.0}
        
        byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = byte_counts / len(data)
        probabilities = probabilities[probabilities > 0]
        
        shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
        
        compressed_size = len(zlib.compress(data, level=9))
        compression_ratio = compressed_size / len(data)
        
        return {
            'shannon': float(shannon_entropy),
            'compression': float(compression_ratio)
        }

class AdvancedYaraEngine:
    def __init__(self):
        self.rules = self._compile_rules()
    
    def _compile_rules(self):
        if not YARA_AVAILABLE:
            return None
        
        rules = """
        rule MalwareIndicators {
            strings:
                $cmd = "cmd.exe" nocase
                $powershell = "powershell" nocase
                $runtime = "runtime" nocase
                $assembly = "assembly" nocase
            condition:
                any of them
        }
        """
        
        try:
            return yara.compile(source=rules)
        except Exception as e:
            print(f"YARA compilation failed: {e}")
            return None

class DeepPEAnalyzer:
    def __init__(self):
        self.suspicious_imports = {
            'kernel32.dll': ['VirtualAlloc', 'CreateRemoteThread', 'WriteProcessMemory'],
            'user32.dll': ['SetWindowsHookEx', 'GetAsyncKeyState'],
            'advapi32.dll': ['RegSetValue', 'CreateService'],
        }
    
    def analyze_pe_structure(self, file_path):
        if not PEFILE_AVAILABLE:
            return {'available': False}
        
        try:
            pe = pefile.PE(str(file_path))
            analysis = {
                'sections': len(pe.sections),
                'anomalies': []
            }
            
            for section in pe.sections:
                entropy = section.get_entropy()
                if entropy > 7.5:
                    analysis['anomalies'].append(f"High entropy: {entropy:.2f}")
            
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    dll_name = entry.dll.decode('utf-8', errors='ignore')
                    if dll_name.lower() in self.suspicious_imports:
                        analysis['anomalies'].append(f"Suspicious DLL: {dll_name}")
            
            return analysis
            
        except Exception:
            return {'available': False}

class NeuralBehavioralAnalyzer:
    def __init__(self):
        self.suspicious_patterns = [
            'virtualalloc', 'createremotethread', 'writeprocessmemory',
            'regsetvalue', 'createservice', 'schtasks',
            'amsi', 'bypass', 'obfuscate',
            'http', 'socket', 'connect',
            'base64', 'encoded', 'eval', 'exec', 'system'
        ]
    
    def analyze_behavioral_patterns(self, file_data):
        score = 0.0
        detected_patterns = []
        
        try:
            content = file_data.decode('utf-8', errors='ignore').lower()
            
            for pattern in self.suspicious_patterns:
                if pattern in content:
                    score += 0.05
                    detected_patterns.append(pattern)
            
            entropy = self._calculate_entropy(file_data)
            if entropy > 7.0:
                score += 0.3
                detected_patterns.append("high_entropy")
                
        except Exception:
            pass
        
        return {
            'score': min(score, 1.0),
            'detected_patterns': detected_patterns[:10]
        }
    
    def _calculate_entropy(self, data):
        if len(data) == 0:
            return 0.0
        entropy = 0.0
        for x in range(256):
            p_x = data.count(x) / len(data)
            if p_x > 0:
                entropy += -p_x * np.log2(p_x)
        return entropy

class MilitaryGradeScanner:
    def __init__(self):
        self.crypto_engine = QuantumCryptographicEngine()
        self.yara_engine = AdvancedYaraEngine()
        self.pe_analyzer = DeepPEAnalyzer()
        self.behavioral_analyzer = NeuralBehavioralAnalyzer()
        
        self.scan_stats = {
            'files_scanned': 0,
            'threats_detected': 0,
            'scan_times': []
        }
    
    def scan_file(self, file_path):
        start_time = datetime.now()
        
        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()
        except Exception as e:
            return self._create_error_result(file_path, str(e))
        
        detection_details = self._perform_analysis(file_path, file_data)
        threat_score = self._calculate_threat_score(detection_details)
        threat_level = self._determine_threat_level(threat_score)
        quantum_hash = self.crypto_engine.generate_quantum_hash(file_data)
        
        scan_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(threat_level, scan_time)
        
        return {
            'file_path': str(file_path),
            'threat_score': round(threat_score, 3),
            'threat_level': threat_level,
            'detection_details': detection_details,
            'quantum_hash': quantum_hash,
            'scan_time': scan_time
        }
    
    def _perform_analysis(self, file_path, file_data):
        analysis = {}
        
        analysis['yara'] = self._yara_scan(file_data)
        analysis['pe'] = self.pe_analyzer.analyze_pe_structure(file_path)
        analysis['behavioral'] = self.behavioral_analyzer.analyze_behavioral_patterns(file_data)
        analysis['entropy'] = self.crypto_engine.advanced_entropy_analysis(file_data)
        
        return analysis
    
    def _yara_scan(self, file_data):
        if not YARA_AVAILABLE or not self.yara_engine.rules:
            return {'available': False}
        
        try:
            matches = self.yara_engine.rules.match(data=file_data)
            return {
                'available': True,
                'matches': len(matches),
                'rules': [match.rule for match in matches]
            }
        except Exception:
            return {'available': False}
    
    def _calculate_threat_score(self, analysis):
        score = 0.0
        
        yara = analysis.get('yara', {})
        if yara.get('available') and yara.get('matches', 0) > 0:
            score += min(yara['matches'] * 0.2, 0.4)
        
        behavioral = analysis.get('behavioral', {})
        score += behavioral.get('score', 0.0) * 0.4
        
        pe = analysis.get('pe', {})
        if pe.get('anomalies'):
            score += min(len(pe['anomalies']) * 0.1, 0.2)
        
        entropy = analysis.get('entropy', {})
        if entropy.get('shannon', 0) > 7.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _determine_threat_level(self, threat_score):
        if threat_score >= 0.8:
            return ThreatLevel.CRITICAL
        elif threat_score >= 0.6:
            return ThreatLevel.HIGH
        elif threat_score >= 0.4:
            return ThreatLevel.MEDIUM
        elif threat_score >= 0.2:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.CLEAN
    
    def _create_error_result(self, file_path, error):
        return {
            'file_path': str(file_path),
            'threat_score': 0.0,
            'threat_level': ThreatLevel.CLEAN,
            'error': error,
            'quantum_hash': 'ERROR'
        }
    
    def _update_stats(self, threat_level, scan_time):
        self.scan_stats['files_scanned'] += 1
        self.scan_stats['scan_times'].append(scan_time)
        
        if threat_level.value >= ThreatLevel.MEDIUM.value:
            self.scan_stats['threats_detected'] += 1
    
    def get_stats(self):
        times = self.scan_stats['scan_times']
        avg_time = sum(times) / len(times) if times else 0
        return {
            'files_scanned': self.scan_stats['files_scanned'],
            'threats_detected': self.scan_stats['threats_detected'],
            'avg_scan_time': round(avg_time, 3)
        }

def display_banner():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║ 🛡️  AI MODEL SENTINEL - MILITARY GRADE SECURITY SCANNER  🛡️ ║")
    print("║ 🚀        QUANTUM THREAT DETECTION SYSTEM v2.0.0         🚀 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

def main():
    display_banner()
    
    print("🔍 Initializing security scanner...")
    print("⚡ Loading detection engines...")
    print()
    
    scanner = MilitaryGradeScanner()
    
    test_files = [
        "phase2_engines.py",
        "phase1_foundation.py", 
        "ai_security_scanner.py",
        "production_cli.py",
        "core_engine.py",
        "threat_detectors.py"
    ]
    
    existing_files = [f for f in test_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No files found for scanning")
        return
    
    print(f"🎯 Scanning {len(existing_files)} files...")
    print()
    
    results = []
    for file_path in existing_files:
        print(f"🔍 Analyzing: {file_path}")
        result = scanner.scan_file(Path(file_path))
        results.append(result)
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
            continue
        
        level = result['threat_level']
        score = result['threat_score']
        
        level_icon = {
            ThreatLevel.CRITICAL: "🔴",
            ThreatLevel.HIGH: "🟠", 
            ThreatLevel.MEDIUM: "🟡",
            ThreatLevel.LOW: "🟢",
            ThreatLevel.CLEAN: "✅"
        }.get(level, "⚪")
        
        print(f"   {level_icon} Threat: {level.name} (Score: {score:.3f})")
        
        details = result['detection_details']
        behavioral = details.get('behavioral', {})
        if behavioral.get('detected_patterns'):
            patterns = behavioral['detected_patterns']
            print(f"   🧠 Patterns: {len(patterns)} detected")
        
        yara = details.get('yara', {})
        if yara.get('matches', 0) > 0:
            print(f"   📈 YARA: {yara['matches']} rules matched")
    
    print()
    print("📊 SCAN SUMMARY")
    print("═" * 50)
    
    stats = scanner.get_stats()
    print(f"📁 Files scanned: {stats['files_scanned']}")
    print(f"⚠️  Threats detected: {stats['threats_detected']}")
    print(f"⏱️  Average scan time: {stats['avg_scan_time']}s")
    
    threat_counts = {level: 0 for level in ThreatLevel}
    for result in results:
        if 'error' not in result:
            threat_counts[result['threat_level']] += 1
    
    print()
    print("🎯 THREAT BREAKDOWN")
    for level in ThreatLevel:
        count = threat_counts[level]
        icon = {
            ThreatLevel.CRITICAL: "🔴",
            ThreatLevel.HIGH: "🟠",
            ThreatLevel.MEDIUM: "🟡", 
            ThreatLevel.LOW: "🟢",
            ThreatLevel.CLEAN: "✅"
        }[level]
        print(f"   {icon} {level.name}: {count} files")
    
    print()
    print("✅ Scan completed successfully!")

if __name__ == "__main__":
    main()