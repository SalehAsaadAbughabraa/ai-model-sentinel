# Test Phase 2 Components 
import numpy as np 
from core.threat_detector import ThreatDetector 
from core.adaptive_persona import AdaptivePersona 
from core.response_engine import ResponseEngine 
 
print("Testing Phase 2 - Adaptive Intelligence") 
 
# Test Threat Detector 
print("1. Testing Threat Detector...") 
detector = ThreatDetector() 
normal_data = np.random.normal(0.5, 0.1, (100, 10)) 
detector.fit(normal_data) 
 
test_data = np.random.normal(2.0, 0.5, (10,))  # Anomalous data 
results = detector.detect_anomalies(test_data) 
threat_level = detector.calculate_threat_level(results) 
print(f"Threat level: {threat_level:.2f}") 
 
# Test Adaptive Persona 
print("2. Testing Adaptive Persona...") 
persona = AdaptivePersona() 
for i in range(5): 
    detected_persona = persona.analyze_behavior(np.random.rand(10, 10)) 
strategy = persona.get_response_strategy(detected_persona) 
print(f"Detected persona: {detected_persona}") 
print(f"Response strategy: {strategy}") 
 
# Test Response Engine 
print("3. Testing Response Engine...") 
response_engine = ResponseEngine() 
sample_data = np.random.rand(5, 5) 
response = response_engine.generate_response(sample_data, strategy) 
print(f"Original data shape: {sample_data.shape}") 
print(f"Response data shape: {response.shape}") 
 
print("Phase 2 components tested successfully!") 
