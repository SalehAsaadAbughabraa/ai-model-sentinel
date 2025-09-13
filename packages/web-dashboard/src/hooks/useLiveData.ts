import { useState, useEffect } from 'react';

export interface LiveMetrics {
  accuracy: number;
  dataDrift: number;
  activeAlerts: number;
  inferenceTime: number;
  lastUpdated: string;
}

export const useLiveData = (updateInterval: number = 5000) => {
  const [metrics, setMetrics] = useState<LiveMetrics>({
    accuracy: 92.3,
    dataDrift: 0.15,
    activeAlerts: 3,
    inferenceTime: 45,
    lastUpdated: new Date().toISOString()
  });

  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        accuracy: Math.max(85, Math.min(95, prev.accuracy + (Math.random() - 0.5))),
        dataDrift: Math.max(0.05, Math.min(0.3, prev.dataDrift + (Math.random() - 0.5) * 0.05)),
        activeAlerts: Math.max(0, Math.min(10, prev.activeAlerts + (Math.random() > 0.8 ? 1 : 0))),
        inferenceTime: Math.max(30, Math.min(100, prev.inferenceTime + (Math.random() - 0.5) * 5)),
        lastUpdated: new Date().toISOString()
      }));
    }, updateInterval);

    return () => clearInterval(interval);
  }, [updateInterval]);

  return metrics;
};