import { Command } from 'commander';
import { logger } from '@ai-model-sentinel/core';

export const versionCommand = new Command()
  .name('version')
  .description('Show version information')
  .action(() => {
    logger.info('AI Model Sentinel CLI v0.1.0-alpha.0');
    logger.info('Enterprise AI Model Monitoring Platform');
  });