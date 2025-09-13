import { ConfigManager, AppConfig } from './config-manager';

export class MonitoringService {
  private configManager: ConfigManager;
  private isMonitoring: boolean = false;

  constructor(configManager: ConfigManager) {
    this.configManager = configManager;
  }

  async startMonitoring(config?: AppConfig): Promise<void> {
    if (this.isMonitoring) {
      console.warn('Monitoring is already running');
      return;
    }

    const monitoringConfig = config || this.configManager.getConfig();
    
    console.log('Starting monitoring service...');
    console.log(`Monitoring path: ${monitoringConfig.modelPath}`);
    console.log(`Check interval: ${monitoringConfig.checkInterval}s`);

    this.isMonitoring = true;
    
    return new Promise((resolve) => {
      setTimeout(() => {
        console.log('Monitoring service started successfully');
        resolve();
      }, 1000);
    });
  }

  stopMonitoring(): void {
    if (!this.isMonitoring) {
      console.warn('Monitoring is not running');
      return;
    }

    this.isMonitoring = false;
    console.log('Monitoring service stopped');
  }

  getStatus(): { isMonitoring: boolean; config: AppConfig } {
    return {
      isMonitoring: this.isMonitoring,
      config: this.configManager.getConfig()
    };
  }
}