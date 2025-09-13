import { MonitoringService, ConfigManager } from '@ai-model-sentinel/core';

export const handleStartCommand = async (options: any) => {
  try {
    const configManager = new ConfigManager();
    
    let config;
    if (options.config) {
      config = configManager.loadConfig(options.config);
    } else {
      config = configManager.getConfig();
    }

    const monitoringService = new MonitoringService(configManager);
    await monitoringService.startMonitoring(config);
    
    console.log('Monitoring service started successfully');
  } catch (error) {
    console.error('Start command failed:', error);
  }
};