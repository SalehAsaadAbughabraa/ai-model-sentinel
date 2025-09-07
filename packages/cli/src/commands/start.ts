import { Command } from 'commander';
import { MonitoringService, ConfigManager } from '@ai-model-sentinel/core';
import { logger } from '@ai-model-sentinel/core';

export const startCommand = new Command()
  .name('start')
  .description('Start monitoring AI models')
  .option('-c, --config <path>', 'Path to config file')
  .option('-i, --interval <ms>', 'Monitoring interval in milliseconds')
  .action(async (options) => {
    try {
      logger.info('Starting AI Model Sentinel...');
      
      // Load configuration
      const config = options.config 
        ? ConfigManager.loadFromFile(options.config)
        : ConfigManager.loadFromEnv();
      
      // Override interval if provided
      if (options.interval) {
        config.monitoring.interval = parseInt(options.interval);
      }

      // Create and start monitoring service
      const monitoringService = new MonitoringService(config);
      monitoringService.startMonitoring();

      logger.info('Monitoring started successfully!');
      logger.info(`Model ID: ${config.modelId}`);
      logger.info(`Interval: ${config.monitoring.interval}ms`);

      // Handle graceful shutdown
      process.on('SIGINT', () => {
        logger.info('Shutting down...');
        monitoringService.stopMonitoring();
        process.exit(0);
      });

    } catch (error) {
      logger.error('Failed to start monitoring:', error);
      process.exit(1);
    }
  });