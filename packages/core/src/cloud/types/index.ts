export interface CloudProviderConfig {
  provider: 'aws' | 'azure' | 'gcp' | 'huggingface';
  credentials: {
    accessKeyId?: string;
    secretAccessKey?: string;
    region?: string;
    projectId?: string;
    subscriptionId?: string;
  };
}

export interface DeploymentConfig {
  modelPath: string;
  target: string;
  runtime: 'python' | 'nodejs' | 'docker';
  memorySize?: number;
  timeout?: number;
}

export interface MonitoringConfig {
  metrics: string[];
  alertThresholds: {
    [key: string]: number;
  };
  notificationEmails?: string[];
}

export interface CloudStorageConfig {
  bucketName: string;
  filePath: string;
  isPublic?: boolean;
}

export interface TrainingJobConfig {
  datasetPath: string;
  modelType: string;
  hyperparameters: { [key: string]: any };
  instanceType: string;
}

export interface AppConfig {
  modelPath: string;
  checkInterval: number;
  apiKey?: string;
  alerts?: {
    email?: string;
    slackWebhook?: string;
  };
}

export interface ScanResult {
  models: Array<{
    filePath: string;
    format: string;
    size: number;
    detectedAt: Date;
  }>;
  stats: {
    directoriesScanned: number;
    filesScanned: number;
    scanDuration: number;
  };
}

export interface MonitorConfig {
  modelPath: string;
  checkInterval: number;
  apiKey?: string;
}

export interface ReportOptions {
  format: 'json' | 'html' | 'csv';
  outputPath?: string;
}

export interface ReportResult {
  filePath: string;
  format: string;
  generatedAt: Date;
}