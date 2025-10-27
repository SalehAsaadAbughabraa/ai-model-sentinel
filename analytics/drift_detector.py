# analytics/drift_detector.py
from typing import Optional, Dict, Any, List
from enum import Enum
import numpy as np

class DriftType(Enum):
    COVARIATE = "covariate"
    CONCEPT = "concept" 
    LABEL = "label"

class DriftConfig:
    def __init__(self, threshold: float = 0.05, method: str = "ks_test"):
        self.threshold = threshold
        self.method = method

class AdvancedDriftDetector:
    def __init__(self, config: DriftConfig = None):
        self.config = config or DriftConfig()
        self.drift_history = []
        
    def detect_drift(self, reference_data, current_data) -> Dict[str, Any]:
        """
        كشف الانحراف بين البيانات المرجعية والبيانات الحالية
        """
        try:
            # تحويل إلى numpy arrays للتحليل
            ref_data = np.array(reference_data)
            curr_data = np.array(current_data)
            
            # حساب إحصائيات بسيطة
            ref_mean = np.mean(ref_data) if len(ref_data) > 0 else 0
            curr_mean = np.mean(curr_data) if len(curr_data) > 0 else 0
            ref_std = np.std(ref_data) if len(ref_data) > 0 else 1
            curr_std = np.std(curr_data) if len(curr_data) > 0 else 1
            
            # حساب مسافة الانحراف المبسطة
            mean_diff = abs(ref_mean - curr_mean)
            std_diff = abs(ref_std - curr_std)
            
            # تحديد إذا كان هناك انحراف
            drift_detected = (mean_diff > self.config.threshold * ref_std or 
                            std_diff > self.config.threshold * ref_std)
            
            result = {
                "drift_detected": drift_detected,
                "confidence": min(0.95, (mean_diff + std_diff) / (2 * ref_std)),
                "drift_type": DriftType.COVARIATE.value if drift_detected else None,
                "metrics": {
                    "mean_difference": float(mean_diff),
                    "std_difference": float(std_diff),
                    "reference_mean": float(ref_mean),
                    "current_mean": float(curr_mean)
                },
                "message": "Drift detected" if drift_detected else "No significant drift"
            }
            
            self.drift_history.append(result)
            return result
            
        except Exception as e:
            return {
                "drift_detected": False,
                "confidence": 0.0,
                "drift_type": None,
                "error": str(e),
                "message": "Error in drift detection"
            }
    
    def get_drift_history(self) -> List[Dict[str, Any]]:
        """الحصول على سجل الانحرافات"""
        return self.drift_history
    
    def reset_history(self):
        """مسح سجل الانحرافات"""
        self.drift_history.clear()

# دوال مساعدة
def create_default_detector() -> AdvancedDriftDetector:
    """إنشاء كاشف انحراف بالإعدادات الافتراضية"""
    return AdvancedDriftDetector(DriftConfig(threshold=0.05))

def detect_batch_drift(reference_batch, current_batch, config=None):
    """كشف الانحراف للدفعات"""
    detector = AdvancedDriftDetector(config)
    return detector.detect_drift(reference_batch, current_batch)