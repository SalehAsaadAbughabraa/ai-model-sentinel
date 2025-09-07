import { Command } from 'commander';
import { ConfigManager } from '@ai-model-sentinel/core';
import { logger } from '@ai-model-sentinel/core';

export const configCommand = new Command()
  .name('config')
  .description('Manage configuration')
  .option('--show', 'Show current configuration')
  .option('--validate', 'Validate configuration')
  .action(async (options) => {
    try {
      if (options.show) {
        const config = ConfigManager.loadFromEnv();
        logger.info('Current configuration:');
        console.log(JSON.stringify(config, null, 2));
      }

      if (options.validate) {
        const config = ConfigManager.loadFromEnv();
        logger.info('Configuration is valid!');
      }

    } catch (error) {
      logger.error('Configuration error:', error);
      process.exit(1);
    }
  });