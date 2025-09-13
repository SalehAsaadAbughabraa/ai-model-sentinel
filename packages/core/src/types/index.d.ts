export interface ModelMetrics {
    accuracy: number;
    precision: number;
    recall: number;
    f1Score: number;
    inferenceTime: number;
    timestamp: Date;
}
export interface DataDriftResult {
    score: number;
    confidence: number;
    features: string[];
    detectedAt: Date;
    severity: 'low' | 'medium' | 'high' | 'critical';
}
export interface MonitoringConfig {
    modelId: string;
    checkInterval: number;
    driftThreshold: number;
    performanceThreshold: number;
}
export interface Alert {
    id: string;
    type: 'drift' | 'performance' | 'anomaly';
    severity: 'low' | 'medium' | 'high' | 'critical';
    message: string;
    timestamp: Date;
    metadata: Record<string, any>;
}
//# sourceMappingURL=index.d.ts.map