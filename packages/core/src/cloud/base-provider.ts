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

  // Methods that all cloud providers must implement
  abstract initialize(): Promise<void>;
  abstract authenticate(): Promise<boolean>;
  
  // Deployment methods
  abstract deployModel(config: DeploymentConfig): Promise<string>;
  abstract undeployModel(deploymentId: string): Promise<void>;
  
  // Storage methods
  abstract uploadModel(config: CloudStorageConfig): Promise<string>;
  abstract downloadModel(bucketName: string, filePath: string, localPath: string): Promise<void>;
  
  // Monitoring methods
  abstract setupMonitoring(config: MonitoringConfig): Promise<void>;
  abstract getMetrics(deploymentId: string, startTime: Date, endTime: Date): Promise<any>;
  
  // Training methods (optional)
  async startTrainingJob(config: TrainingJobConfig): Promise<string> {
    throw new Error('Training not supported by this provider');
  }
  
  async getTrainingStatus(jobId: string): Promise<any> {
    throw new Error('Training not supported by this provider');
  }

  // Utility methods
  abstract listDeployments(): Promise<string[]>;
  abstract getDeploymentStatus(deploymentId: string): Promise<string>;
  
  // Health check
  abstract healthCheck(): Promise<boolean>;

  // Common error handling
  protected handleError(error: any, context: string): never {
    const errorMessage = `Cloud provider error in ${context}: ${error.message}`;
    console.error(errorMessage);
    throw new Error(errorMessage);
  }
}