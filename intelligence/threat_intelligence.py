# intelligence/threat_intelligence.py
"""
🛡️ Threat Intelligence - نظام ذكاء التهديدات المتقدم v2.0.0
المطور: Saleh Asaad Abughabra
البريد: saleh87alally@gmail.com

نظام متكامل لجمع وتحليل تهديدات أمان نماذج الذكاء الاصطناعي
يدعم المصادر العالمية، التحليل التلقائي، والإنذارات الذكية
Comprehensive AI Model Threat Intelligence System
Supports global sources, automated analysis, and smart alerts
"""

import requests
import json
import logging
import time
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import sqlite3
from datetime import datetime, timedelta
import yaml
import re
from pathlib import Path

class Language(Enum):
    """Supported Languages / اللغات المدعومة"""
    ARABIC = "ar"
    ENGLISH = "en"

class ThreatLevel(Enum):
    """Threat Levels / مستويات خطورة التهديدات"""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class ThreatCategory(Enum):
    """Threat Categories / فئات التهديدات"""
    MODEL_POISONING = "model_poisoning"
    BACKDOOR_ATTACK = "backdoor_attack"
    ADVERSARIAL_ATTACK = "adversarial_attack"
    DATA_EXFILTRATION = "data_exfiltration"
    MODEL_STEALING = "model_stealing"
    INFERENCE_ATTACK = "inference_attack"
    PROMPT_INJECTION = "prompt_injection"

@dataclass
class ThreatSignature:
    """Threat Signature / توقيع تهديد"""
    signature_id: str
    category: ThreatCategory
    pattern: str
    description: str
    threat_level: ThreatLevel
    confidence: float
    mitigation: str
    created_at: str
    last_updated: str

@dataclass 
class ThreatIntelligenceConfig:
    """Threat Intelligence Configuration / تهيئة نظام ذكاء التهديدات"""
    update_interval: int = 3600  # 1 hour / ساعة
    threat_feeds: List[str] = None
    local_signatures_path: str = "threat_signatures.yaml"
    enable_auto_update: bool = True
    max_signatures: int = 10000
    language: Language = Language.ENGLISH  # Default language / اللغة الافتراضية

    def __post_init__(self):
        if self.threat_feeds is None:
            self.threat_feeds = [
                "https://raw.githubusercontent.com/SalehAsaadAbughabraa/ai-model-sentinel/main/threat_signatures.yaml",
                "https://threatfeeds.io/ai-security/signatures.json"
            ]

class BilingualManager:
    """Manager for bilingual support / مدير الدعم ثنائي اللغة"""
    
    def __init__(self, language: Language):
        self.language = language
        self.messages = self._load_messages()
    
    def _load_messages(self) -> Dict[str, Dict[str, str]]:
        """Load bilingual messages / تحميل الرسائل ثنائية اللغة"""
        return {
            "system_ready": {
                "en": "✅ Threat Intelligence System Ready",
                "ar": "✅ نظام ذكاء التهديدات جاهز"
            },
            "fetching_threats": {
                "en": "📡 Fetching threats from",
                "ar": "📡 جلب التهديدات من"
            },
            "fetch_failed": {
                "en": "❌ Failed to fetch threats from",
                "ar": "❌ فشل جلب التهديدات من"
            },
            "analysis_failed": {
                "en": "❌ Threat analysis failed",
                "ar": "❌ فشل تحليل التهديدات"
            },
            "signature_check_failed": {
                "en": "⚠️ Signature check failed for",
                "ar": "⚠️ فشل فحص التوقيع لـ"
            },
            "loading_failed": {
                "en": "❌ Failed to load signatures",
                "ar": "❌ فشل تحميل التوقيعات"
            },
            "scan_failed": {
                "en": "❌ Model scan failed",
                "ar": "❌ فشل فحص النموذج"
            },
            "db_save_failed": {
                "en": "❌ Failed to save signatures to database",
                "ar": "❌ فشل حفظ التوقيعات في قاعدة البيانات"
            },
            "threats_detected": {
                "en": "Threats Detected",
                "ar": "التهديدات المكتشفة"
            },
            "total_checks": {
                "en": "Total Checks",
                "ar": "إجمالي الفحوصات"
            },
            "analysis_time": {
                "en": "Analysis Time",
                "ar": "وقت التحليل"
            },
            "max_severity": {
                "en": "Maximum Severity",
                "ar": "أقصى خطورة"
            },
            "file_size": {
                "en": "File Size",
                "ar": "حجم الملف"
            },
            "file_hash": {
                "en": "File Hash",
                "ar": "بصمة الملف"
            },
            "signatures_loaded": {
                "en": "Signatures Loaded",
                "ar": "التوقيعات المحملة"
            },
            "categories": {
                "en": "Categories",
                "ar": "الفئات"
            },
            "threat_levels": {
                "en": "Threat Levels",
                "ar": "مستويات التهديد"
            },
            "testing_system": {
                "en": "🧪 Testing Threat Intelligence System v2.0...",
                "ar": "🧪 اختبار نظام ذكاء التهديدات الإصدار 2.0..."
            },
            "system_operational": {
                "en": "🚀 Threat Intelligence System v2.0 is operational!",
                "ar": "🚀 نظام ذكاء التهديدات الإصدار 2.0 جاهز للعمل!"
            }
        }
    
    def get_message(self, message_key: str) -> str:
        """Get message in current language / الحصول على الرسالة باللغة الحالية"""
        return self.messages.get(message_key, {}).get(self.language.value, message_key)
    
    def get_bilingual(self, en_text: str, ar_text: str) -> str:
        """Get bilingual text / الحصول على نص ثنائي اللغة"""
        return en_text if self.language == Language.ENGLISH else ar_text

class ThreatFeedManager:
    """Threat Feed Manager / مدير مصادر التهديدات"""
    
    def __init__(self, config: ThreatIntelligenceConfig, bilingual: BilingualManager):
        self.config = config
        self.bilingual = bilingual
        self.logger = logging.getLogger('ThreatFeedManager')
    
    def fetch_threat_feeds(self) -> List[Dict[str, Any]]:
        """Fetch threats from external sources / جمع التهديدات من المصادر الخارجية"""
        all_threats = []
        
        for feed_url in self.config.threat_feeds:
            try:
                self.logger.info(f"{self.bilingual.get_message('fetching_threats')}: {feed_url}")
                
                response = requests.get(feed_url, timeout=30)
                if response.status_code == 200:
                    if feed_url.endswith(('.yaml', '.yml')):
                        threats = yaml.safe_load(response.text)
                    else:
                        threats = response.json()
                    
                    if threats:
                        all_threats.extend(self._parse_threat_feed(threats, feed_url))
                        
            except Exception as e:
                self.logger.error(f"{self.bilingual.get_message('fetch_failed')} {feed_url}: {e}")
        
        return all_threats
    
    def _parse_threat_feed(self, threats_data: Any, source: str) -> List[Dict[str, Any]]:
        """Parse threat data / تحليل بيانات التهديدات"""
        parsed_threats = []
        
        try:
            if isinstance(threats_data, list):
                for threat in threats_data:
                    if self._validate_threat_signature(threat):
                        threat['source'] = source
                        parsed_threats.append(threat)
            elif isinstance(threats_data, dict):
                for category, threats in threats_data.items():
                    if isinstance(threats, list):
                        for threat in threats:
                            if self._validate_threat_signature(threat):
                                threat['source'] = source
                                threat['category'] = category
                                parsed_threats.append(threat)
            
            return parsed_threats
            
        except Exception as e:
            self.logger.error(f"{self.bilingual.get_message('fetch_failed')} {source}: {e}")
            return []
    
    def _validate_threat_signature(self, threat: Dict[str, Any]) -> bool:
        """Validate threat signature / التحقق من صحة توقيع التهديد"""
        required_fields = ['pattern', 'description', 'threat_level']
        return all(field in threat for field in required_fields)

class ThreatAnalyzer:
    """Advanced Threat Analyzer / محلل التهديدات المتقدم"""
    
    def __init__(self, bilingual: BilingualManager):
        self.bilingual = bilingual
        self.logger = logging.getLogger('ThreatAnalyzer')
        self.signature_cache = {}
    
    def analyze_model_threats(self, model_data: bytes, signatures: List[ThreatSignature]) -> Dict[str, Any]:
        """Analyze model for threat detection / تحليل النموذج لاكتشاف التهديدات"""
        start_time = time.time()
        threats_found = []
        
        try:
            for signature in signatures:
                threat_result = self._check_signature(model_data, signature)
                if threat_result['detected']:
                    threats_found.append(threat_result)
            
            return {
                'threats_detected': threats_found,
                'total_checks': len(signatures),
                'threats_count': len(threats_found),
                'analysis_time': time.time() - start_time,
                'max_severity': self._get_max_severity(threats_found)
            }
            
        except Exception as e:
            self.logger.error(f"{self.bilingual.get_message('analysis_failed')}: {e}")
            return {'error': str(e)}
    
    def _check_signature(self, model_data: bytes, signature: ThreatSignature) -> Dict[str, Any]:
        """Check specific threat signature / فحص توقيع تهديد محدد"""
        try:
            # Convert pattern to bytes if it's a string
            # تحويل النمط إلى bytes إذا كان نصاً
            if isinstance(signature.pattern, str):
                pattern = signature.pattern.encode('utf-8')
            else:
                pattern = signature.pattern
            
            # Search for pattern in model data
            # البحث عن النمط في بيانات النموذج
            matches = re.findall(pattern, model_data, re.IGNORECASE | re.DOTALL)
            
            detected = len(matches) > 0
            
            return {
                'signature_id': signature.signature_id,
                'category': signature.category.value,
                'threat_level': signature.threat_level.value,
                'description': signature.description,
                'detected': detected,
                'matches_count': len(matches),
                'confidence': signature.confidence if detected else 0.0,
                'mitigation': signature.mitigation
            }
            
        except Exception as e:
            self.logger.warning(f"{self.bilingual.get_message('signature_check_failed')} {signature.signature_id}: {e}")
            return {
                'signature_id': signature.signature_id,
                'detected': False,
                'error': str(e)
            }
    
    def _get_max_severity(self, threats: List[Dict[str, Any]]) -> str:
        """Get maximum threat severity / الحصول على أعلى مستوى خطورة"""
        if not threats:
            return self.bilingual.get_bilingual("none", "لا شيء")
        
        severity_order = {
            'critical': 4,
            'high': 3, 
            'medium': 2,
            'low': 1,
            'info': 0
        }
        
        max_severity = max(
            threats, 
            key=lambda x: severity_order.get(x['threat_level'], 0)
        )
        
        return max_severity['threat_level']

class ThreatIntelligence:
    """Main Threat Intelligence System / نظام ذكاء التهديدات الرئيسي"""
    
    def __init__(self, config: ThreatIntelligenceConfig = None):
        self.config = config or ThreatIntelligenceConfig()
        self.bilingual = BilingualManager(self.config.language)
        self.logger = self._setup_logging()
        
        # Initialize components / تهيئة المكونات
        self.feed_manager = ThreatFeedManager(self.config, self.bilingual)
        self.threat_analyzer = ThreatAnalyzer(self.bilingual)
        self.signatures_db = ThreatSignaturesDB(self.bilingual)
        
        # Load signatures / تحميل التوقيعات
        self.signatures = self._load_signatures()
        self.last_update = datetime.now()
        
        self.logger.info(f"{self.bilingual.get_message('system_ready')} - {len(self.signatures)} {self.bilingual.get_message('signatures_loaded')}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging system / إعداد نظام التسجيل"""
        logger = logging.getLogger('ThreatIntelligence')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_signatures(self) -> List[ThreatSignature]:
        """Load threat signatures / تحميل توقيعات التهديدات"""
        all_signatures = []
        
        try:
            # Load local signatures / تحميل التوقيعات المحلية
            if Path(self.config.local_signatures_path).exists():
                local_signatures = self._load_local_signatures()
                all_signatures.extend(local_signatures)
            
            # Fetch external signatures / جلب التوقيعات من المصادر الخارجية
            if self.config.enable_auto_update:
                external_signatures = self._fetch_external_signatures()
                all_signatures.extend(external_signatures)
            
            # Save to database / حفظ في قاعدة البيانات
            self.signatures_db.save_signatures(all_signatures)
            
            return all_signatures[:self.config.max_signatures]
            
        except Exception as e:
            self.logger.error(f"{self.bilingual.get_message('loading_failed')}: {e}")
            return self._get_default_signatures()
    
    def _load_local_signatures(self) -> List[ThreatSignature]:
        """Load local signatures / تحميل التوقيعات المحلية"""
        try:
            with open(self.config.local_signatures_path, 'r', encoding='utf-8') as f:
                signatures_data = yaml.safe_load(f)
            
            return self._parse_signatures_data(signatures_data, 'local')
            
        except Exception as e:
            self.logger.error(f"{self.bilingual.get_message('loading_failed')}: {e}")
            return []
    
    def _fetch_external_signatures(self) -> List[ThreatSignature]:
        """Fetch external signatures / جلب التوقيعات الخارجية"""
        threats_data = self.feed_manager.fetch_threat_feeds()
        return self._parse_signatures_data(threats_data, 'external')
    
    def _parse_signatures_data(self, data: Any, source: str) -> List[ThreatSignature]:
        """Parse signatures data / تحليل بيانات التوقيعات"""
        signatures = []
        
        try:
            if isinstance(data, list):
                for item in data:
                    signature = self._create_signature_from_data(item, source)
                    if signature:
                        signatures.append(signature)
            elif isinstance(data, dict):
                for category, items in data.items():
                    if isinstance(items, list):
                        for item in items:
                            item['category'] = category
                            signature = self._create_signature_from_data(item, source)
                            if signature:
                                signatures.append(signature)
            
            return signatures
            
        except Exception as e:
            self.logger.error(f"{self.bilingual.get_message('loading_failed')}: {e}")
            return []
    
    def _create_signature_from_data(self, data: Dict[str, Any], source: str) -> Optional[ThreatSignature]:
        """Create signature object from data / إنشاء كائن توقيع من البيانات"""
        try:
            signature_id = data.get('id', hashlib.md5(
                f"{source}_{data.get('pattern', '')}".encode()
            ).hexdigest())
            
            return ThreatSignature(
                signature_id=signature_id,
                category=ThreatCategory(data.get('category', 'model_poisoning')),
                pattern=data.get('pattern', ''),
                description=data.get('description', ''),
                threat_level=ThreatLevel(data.get('threat_level', 'low')),
                confidence=float(data.get('confidence', 0.5)),
                mitigation=data.get('mitigation', self.bilingual.get_bilingual(
                    'No mitigation provided', 
                    'لم يتم تقديم إجراءات علاج'
                )),
                created_at=data.get('created_at', datetime.now().isoformat()),
                last_updated=datetime.now().isoformat()
            )
        except Exception as e:
            self.logger.warning(f"{self.bilingual.get_message('signature_check_failed')}: {e}")
            return None
    
    def _get_default_signatures(self) -> List[ThreatSignature]:
        """Default signatures / التوقيعات الافتراضية"""
        return [
            ThreatSignature(
                signature_id="default_backdoor_1",
                category=ThreatCategory.BACKDOOR_ATTACK,
                pattern=rb"backdoor|trojan|malicious",
                description=self.bilingual.get_bilingual(
                    "Backdoor pattern detection",
                    "كشف نمط الباب الخلفي"
                ),
                threat_level=ThreatLevel.HIGH,
                confidence=0.8,
                mitigation=self.bilingual.get_bilingual(
                    "Remove and retrain model",
                    "إزالة النموذج وإعادة تدريبه"
                ),
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            ),
            ThreatSignature(
                signature_id="default_prompt_injection_1",
                category=ThreatCategory.PROMPT_INJECTION,
                pattern=rb"ignore|override|system|prompt",
                description=self.bilingual.get_bilingual(
                    "Prompt injection attempt detection",
                    "كشف محاولة حقن الأوامر"
                ),
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.7,
                mitigation=self.bilingual.get_bilingual(
                    "Sanitize input and implement guardrails",
                    "تنظيف المدخلات وتنفيذ حواجز الحماية"
                ),
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat()
            )
        ]
    
    def scan_model(self, model_path: str) -> Dict[str, Any]:
        """Scan model for threats / فحص النموذج للكشف عن التهديدات"""
        try:
            with open(model_path, 'rb') as f:
                model_data = f.read()
            
            # Update signatures if needed / تحديث التوقيعات إذا لزم الأمر
            if self._should_update_signatures():
                self.signatures = self._load_signatures()
                self.last_update = datetime.now()
            
            # Analyze threats / تحليل التهديدات
            analysis_result = self.threat_analyzer.analyze_model_threats(model_data, self.signatures)
            
            # Add additional information / إضافة معلومات إضافية
            analysis_result.update({
                'model_path': model_path,
                'file_size': len(model_data),
                'file_hash': hashlib.sha256(model_data).hexdigest(),
                'signatures_version': self.last_update.isoformat(),
                'total_signatures': len(self.signatures)
            })
            
            # Convert keys to bilingual / تحويل المفاتيح إلى ثنائية اللغة
            return self._make_bilingual_result(analysis_result)
            
        except Exception as e:
            self.logger.error(f"{self.bilingual.get_message('scan_failed')}: {e}")
            return {'error': str(e)}
    
    def _make_bilingual_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Make result keys bilingual / جعل مفاتيح النتيجة ثنائية اللغة"""
        bilingual_keys = {
            'threats_detected': self.bilingual.get_message('threats_detected'),
            'total_checks': self.bilingual.get_message('total_checks'),
            'threats_count': self.bilingual.get_message('threats_detected'),
            'analysis_time': self.bilingual.get_message('analysis_time'),
            'max_severity': self.bilingual.get_message('max_severity'),
            'model_path': self.bilingual.get_bilingual('Model Path', 'مسار النموذج'),
            'file_size': self.bilingual.get_message('file_size'),
            'file_hash': self.bilingual.get_message('file_hash'),
            'signatures_version': self.bilingual.get_bilingual('Signatures Version', 'إصدار التوقيعات'),
            'total_signatures': self.bilingual.get_message('signatures_loaded')
        }
        
        bilingual_result = {}
        for key, value in result.items():
            bilingual_key = bilingual_keys.get(key, key)
            bilingual_result[bilingual_key] = value
        
        return bilingual_result
    
    def _should_update_signatures(self) -> bool:
        """Check if signatures should be updated / التحقق إذا كان يجب تحديث التوقيعات"""
        time_since_update = datetime.now() - self.last_update
        return time_since_update.total_seconds() > self.config.update_interval
    
    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get threat statistics / الحصول على إحصائيات التهديدات"""
        stats = {
            'total_signatures': len(self.signatures),
            'last_update': self.last_update.isoformat(),
            'categories': self._get_category_stats(),
            'threat_levels': self._get_threat_level_stats()
        }
        
        return self._make_bilingual_result(stats)
    
    def _get_category_stats(self) -> Dict[str, int]:
        """Category statistics / إحصائيات الفئات"""
        categories = {}
        for signature in self.signatures:
            category = signature.category.value
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _get_threat_level_stats(self) -> Dict[str, int]:
        """Threat level statistics / إحصائيات مستويات الخطورة"""
        levels = {}
        for signature in self.signatures:
            level = signature.threat_level.value
            levels[level] = levels.get(level, 0) + 1
        return levels

class ThreatSignaturesDB:
    """Threat Signatures Database / قاعدة بيانات توقيعات التهديدات"""
    
    def __init__(self, bilingual: BilingualManager, db_path: str = "threat_signatures.db"):
        self.bilingual = bilingual
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database / تهيئة قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threat_signatures (
                signature_id TEXT PRIMARY KEY,
                category TEXT,
                pattern TEXT,
                description TEXT,
                threat_level TEXT,
                confidence REAL,
                mitigation TEXT,
                created_at TEXT,
                last_updated TEXT,
                source TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_signatures(self, signatures: List[ThreatSignature]):
        """Save signatures to database / حفظ التوقيعات في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for signature in signatures:
                cursor.execute('''
                    INSERT OR REPLACE INTO threat_signatures 
                    (signature_id, category, pattern, description, threat_level, 
                     confidence, mitigation, created_at, last_updated, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signature.signature_id,
                    signature.category.value,
                    signature.pattern if isinstance(signature.pattern, str) else str(signature.pattern),
                    signature.description,
                    signature.threat_level.value,
                    signature.confidence,
                    signature.mitigation,
                    signature.created_at,
                    signature.last_updated,
                    'system'
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"{self.bilingual.get_message('db_save_failed')}: {e}")

# Example usage and testing / مثال للاستخدام والاختبار
if __name__ == "__main__":
    # Test with English / اختبار بالإنجليزية
    print("\n" + "="*50)
    print("Testing with English Language")
    print("="*50)
    
    config_en = ThreatIntelligenceConfig(
        enable_auto_update=False,
        max_signatures=100,
        language=Language.ENGLISH
    )
    
    threat_intel_en = ThreatIntelligence(config_en)
    
    # Display statistics / عرض الإحصائيات
    stats_en = threat_intel_en.get_threat_statistics()
    print(f"✅ {stats_en.get('Signatures Loaded', 0)} signatures loaded")
    print(f"📊 Categories: {stats_en.get('Categories', {})}")
    print(f"🎯 Threat Levels: {stats_en.get('Threat Levels', {})}")
    
    # Test with Arabic / اختبار بالعربية
    print("\n" + "="*50)
    print("اختبار باللغة العربية")
    print("="*50)
    
    config_ar = ThreatIntelligenceConfig(
        enable_auto_update=False,
        max_signatures=100,
        language=Language.ARABIC
    )
    
    threat_intel_ar = ThreatIntelligence(config_ar)
    
    # عرض الإحصائيات
    stats_ar = threat_intel_ar.get_threat_statistics()
    print(f"✅ {stats_ar.get('التوقيعات المحملة', 0)} توقيع محمل")
    print(f"📊 الفئات: {stats_ar.get('الفئات', {})}")
    print(f"🎯 مستويات التهديد: {stats_ar.get('مستويات التهديد', {})}")
    
    print(f"\n{threat_intel_en.bilingual.get_message('system_operational')}")