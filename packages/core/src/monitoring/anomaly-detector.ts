import { logger } from '../utils/logger';

export class AnomalyDetector {
  static detectAnomalies(data: number[]): { anomalies: number[]; score: number } {
    logger.info('Detecting anomalies...');
    
    // Simple anomaly detection placeholder
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const stdDev = Math.sqrt(data.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / data.length);
    
    const anomalies = data.filter((value) => Math.abs(value - mean) > 2 * stdDev);
    
    return {
      anomalies,
      score: anomalies.length / data.length
    };
  }

  static isAnomalous(value: number, threshold: number = 0.1): boolean {
    // Simple threshold-based anomaly detection
    return value > threshold;
  }
}