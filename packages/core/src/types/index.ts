// Core types for AI model monitoring
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

export interface ComplianceReport {
  complianceId: string;
  modelId: string;
  status: 'compliant' | 'non-compliant' | 'pending';
  regulations: string[];
  assessedAt: Date;
  expiresAt: Date;
  details: Record<string, any>;
}

export interface SecurityAssessment {
  assessmentId: string;
  modelId: string;
  score: number;
  vulnerabilities: string[];
  assessedAt: Date;
  recommendations: string[];
}

export interface BlockchainAudit {
  auditId: string;
  transactionHash: string;
  blockNumber: number;
  timestamp: Date;
  action: string;
  details: Record<string, any>;
}

export interface ComplianceCertificate {
  certificateId: string;
  modelId: string;
  issuedAt: Date;
  validUntil: Date;
  issuingAuthority: string;
  standards: string[];
}