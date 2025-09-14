import { AWSProvider } from './providers/aws/aws-provider';
import { AzureProvider } from './providers/azure/azure-provider';
import { GCPProvider } from './providers/gcp/gcp-provider';
import { BaseCloudProvider } from './base-provider';
import { 
  CloudProviderConfig, 
  DeploymentConfig, 
  MonitoringConfig, 
  CloudStorageConfig, 
  TrainingJobConfig,
  CloudProvider 
} from './types';

export class CloudProviderFactory {
  static createProvider(config: CloudProviderConfig): BaseCloudProvider {
    switch (config.provider) {
      case 'aws':
        return new AWSProvider(config);
      case 'azure':
        return new AzureProvider(config);
      case 'gcp':
        return new GCPProvider(config);
      case 'ibm':
        throw new Error('IBM provider coming soon');
      case 'oracle':
        throw new Error('Oracle provider coming soon');
      case 'huggingface':
        throw new Error('Hugging Face provider coming soon');
      default:
        throw new Error(`Unsupported cloud provider: ${config.provider}`);
    }
  }

  static getAvailableProviders(): CloudProvider[] {
    return ['aws', 'azure', 'gcp', 'ibm', 'oracle', 'huggingface'];
  }

  static validateConfig(config: CloudProviderConfig): string[] {
    const errors: string[] = [];

    if (!config.provider) {
      errors.push('Provider is required');
    }

    if (!config.credentials) {
      errors.push('Credentials are required');
    } else {
      switch (config.provider) {
        case 'aws':
          if (!config.credentials.accessKeyId) errors.push('AWS accessKeyId is required');
          if (!config.credentials.secretAccessKey) errors.push('AWS secretAccessKey is required');
          break;
        case 'azure':
          if (!config.credentials.connectionString) errors.push('Azure connectionString is required');
          break;
        case 'gcp':
          if (!config.credentials.projectId) errors.push('GCP projectId is required');
          break;
      }
    }

    return errors;
  }
}

export { AWSProvider } from './providers/aws/aws-provider';
export { AzureProvider } from './providers/azure/azure-provider';
export { GCPProvider } from './providers/gcp/gcp-provider';

export type { 
  CloudProviderConfig, 
  DeploymentConfig, 
  MonitoringConfig, 
  CloudStorageConfig, 
  TrainingJobConfig,
  CloudProvider 
} from './types';