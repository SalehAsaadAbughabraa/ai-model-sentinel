import { 
  CloudProviderConfig, 
  DeploymentConfig, 
  MonitoringConfig, 
  CloudStorageConfig, 
  TrainingJobConfig 
} from './types';

export abstract class BaseCloudProvider {
  protected config: CloudProviderConfig;

  constructor(config: CloudProviderConfig) {
    this.config = config;
  }

  abstract initialize(): Promise<void>;
  abstract authenticate(): Promise<boolean>;
  abstract deployModel(config: DeploymentConfig): Promise<string>;
  abstract undeployModel(deploymentId: string): Promise<void>;
  abstract uploadModel(config: CloudStorageConfig): Promise<string>;
  abstract downloadModel(bucketName: string, filePath: string, localPath: string): Promise<void>;
  abstract setupMonitoring(config: MonitoringConfig): Promise<void>;
  abstract getMetrics(deploymentId: string, startTime: Date, endTime: Date): Promise<any>;
  abstract listDeployments(): Promise<string[]>;
  abstract getDeploymentStatus(deploymentId: string): Promise<string>;
  abstract healthCheck(): Promise<boolean>;

  protected handleError(error: any, context: string): never {
    const errorMessage = `Cloud provider error in ${context}: ${error.message}`;
    console.error(errorMessage);
    throw new Error(errorMessage);
  }
}