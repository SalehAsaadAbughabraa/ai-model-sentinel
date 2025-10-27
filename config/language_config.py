# config/language_config.py
"""
🌐 نظام الترجمة المتعدد اللغات - AI Model Sentinel
"""

from enum import Enum

class Language(Enum):
    ARABIC = "ar"
    ENGLISH = "en"
    FRENCH = "fr"
    SPANISH = "es"

class Translator:
    """نظام الترجمة الديناميكي"""
    
    def __init__(self, language: Language = Language.ARABIC):
        self.language = language
        self.translations = self._load_translations()
    
    def _load_translations(self) -> dict:
        """تحميل الترجمات"""
        return {
            # 🛡️ التهديدات والأمان
            "THREAT_LEVEL_CLEAN": {
                "ar": "✅ نظيف",
                "en": "✅ CLEAN", 
                "fr": "✅ PROPRE",
                "es": "✅ LIMPIO"
            },
            "THREAT_LEVEL_LOW": {
                "ar": "🟢 منخفض",
                "en": "🟢 LOW",
                "fr": "🟢 FAIBLE", 
                "es": "🟢 BAJO"
            },
            "THREAT_LEVEL_MEDIUM": {
                "ar": "🟡 متوسط",
                "en": "🟡 MEDIUM",
                "fr": "🟡 MOYEN",
                "es": "🟡 MEDIO"
            },
            "THREAT_LEVEL_HIGH": {
                "ar": "🟠 مرتفع", 
                "en": "🟠 HIGH",
                "fr": "🟠 ÉLEVÉ",
                "es": "🟠 ALTO"
            },
            "THREAT_LEVEL_CRITICAL": {
                "ar": "🔴 حرج",
                "en": "🔴 CRITICAL",
                "fr": "🔴 CRITIQUE",
                "es": "🔴 CRÍTICO"
            },
            
            # 📊 رسائل النظام
            "SCAN_STARTED": {
                "ar": "🔍 بدء الفحص الشامل",
                "en": "🔍 Starting comprehensive scan",
                "fr": "🔍 Démarrage de l'analyse complète",
                "es": "🔍 Iniciando escaneo completo"
            },
            "SCAN_COMPLETED": {
                "ar": "✅ اكتمل الفحص",
                "en": "✅ Scan completed", 
                "fr": "✅ Analyse terminée",
                "es": "✅ Escaneo completado"
            },
            "FILE_NOT_FOUND": {
                "ar": "❌ الملف غير موجود",
                "en": "❌ File not found",
                "fr": "❌ Fichier introuvable",
                "es": "❌ Archivo no encontrado"
            },
            
            # 💡 التوصيات
            "RECOMMENDATION_CRITICAL": {
                "ar": "🚨 إزالة النموذج فوراً من البيئة التشغيلية",
                "en": "🚨 Remove model immediately from production",
                "fr": "🚨 Retirer immédiatement le modèle de la production",
                "es": "🚨 Eliminar modelo inmediatamente de producción"
            },
            "RECOMMENDATION_MONITOR": {
                "ar": "📊 مراقبة سلوك النموذج عن كثب",
                "en": "📊 Monitor model behavior closely", 
                "fr": "📊 Surveiller étroitement le comportement du modèle",
                "es": "📊 Monitorear comportamiento del modelo de cerca"
            }
        }
    
    def translate(self, key: str) -> str:
        """ترجمة النص"""
        return self.translations.get(key, {}).get(self.language.value, key)
    
    def set_language(self, language: Language):
        """تغيير اللغة"""
        self.language = language

# المترجم العام
translator = Translator()