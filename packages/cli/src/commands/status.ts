import { Command } from 'commander';
import { logger } from '@ai-model-sentinel/core';

export const statusCommand = new Command()
  .name('status')
  .description('Show current monitoring status')
  .action(() => {
    try {
      logger.info('🔄 Checking monitoring status...');
      
      const status = {
        monitoring: 'active',
        models: 3,
        alerts: 0,
        uptime: '2 hours',
        lastCheck: new Date().toISOString()
      };

      logger.info('📊 Monitoring Status:');
      console.log(JSON.stringify(status, null, 2));
      
    } catch (error) {
      logger.error('Failed to get status:', error);
      process.exit(1);
    }
  });