import { BaseCloudProvider } from './base-provider';
import { CloudProviderConfig, CloudProvider } from './types';

export class CloudProviderFactory {
  static createProvider(config: CloudProviderConfig): BaseCloudProvider {
    throw new Error('Cloud providers not implemented yet');
  }

  static getAvailableProviders(): CloudProvider[] {
    return ['aws', 'azure', 'gcp'];
  }
}

export type { CloudProviderConfig, CloudProvider } from './types';