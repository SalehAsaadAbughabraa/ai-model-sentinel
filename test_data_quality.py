import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.data_quality_engine import DataQualityEngine

print('Testing Data Quality Engine')
engine = DataQualityEngine()

test_cases = {
    'clean_data': np.random.randn(1000),
    'noisy_data': np.random.randn(1000) + np.random.randn(1000) * 10,
    'sparse_data': np.array([1, 2, 3] + [0] * 997),
    'outlier_data': np.concatenate([np.random.randn(950), np.random.randn(50) * 100]),
    'corrupted_data': np.array([np.nan, np.inf, -np.inf] + list(np.random.randn(997)))
}

results = []
for name, data in test_cases.items():
    print('Testing ' + name + ':')
    integrity = engine.analyze_data_integrity(data)
    noise_sensitivity = engine.analyze_noise_sensitivity(data)
    print('Integrity: ' + str(integrity))
    print('Noise Sensitivity: ' + str(noise_sensitivity))
    avg_quality = (sum(integrity.values()) + sum(noise_sensitivity.values())) / (len(integrity) + len(noise_sensitivity))
    print('Average Quality Score: ' + str(round(avg_quality, 3)))
    results.append(avg_quality)

print('DATA QUALITY ENGINE TEST COMPLETED - Average Score: ' + str(round(sum(results)/len(results), 3)))