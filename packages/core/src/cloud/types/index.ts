export type CloudProvider = 'aws' | 'azure' | 'gcp' | 'ibm' | 'oracle' | 'huggingface';

export interface CloudProviderConfig {
  provider: CloudProvider;
  credentials: {
    accessKeyId?: string;
    secretAccessKey?: string;
    region?: string;
    projectId?: string;
    subscriptionId?: string;
    connectionString?: string;
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