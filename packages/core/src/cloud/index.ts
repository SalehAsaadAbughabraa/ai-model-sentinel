import { BaseCloudProvider } from './base-provider';
import { CloudProviderConfig } from './types';
import { AWSProvider } from './providers/aws/aws-provider';

export class CloudProviderFactory {
  static createProvider(config: CloudProviderConfig): BaseCloudProvider {
    switch (config.provider) {
      case 'aws':
        return new AWSProvider(config);
      case 'azure':
        throw new Error('Azure provider not yet implemented');
      case 'gcp':
        throw new Error('GCP provider not yet implemented');
      case 'huggingface':
        throw new Error('Hugging Face provider not yet implemented');
      default:
        throw new Error(`Unsupported cloud provider: ${config.provider}`);
    }
  }

  static getAvailableProviders(): string[] {
    return ['aws', 'azure', 'gcp', 'huggingface'];
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
          if (!config.credentials.subscriptionId) errors.push('Azure subscriptionId is required');
          break;
        case 'gcp':
          if (!config.credentials.projectId) errors.push('GCP projectId is required');
          break;
      }
    }

    return errors;
  }
}

// Export individual providers for direct usage
export { AWSProvider } from './providers/aws/aws-provider';
export type { CloudProviderConfig } from './types';

// Re-export types for convenience
export type { DeploymentConfig, MonitoringConfig, CloudStorageConfig, TrainingJobConfig } from './types';