#!/usr/bin/env node

import { Command } from 'commander';
import { startCommand } from '../commands/start';
import { configCommand } from '../commands/config';
import { versionCommand } from '../commands/version';
import { statusCommand } from '../commands/status';
import { alertsCommand } from '../commands/alerts';
import { modelsCommand } from '../commands/models';
import { logger } from '@ai-model-sentinel/core';

const program = new Command();

program
  .name('ai-model-sentinel')
  .description('Enterprise AI Model Monitoring CLI')
  .version('0.1.0-alpha.0');

// Add commands
program.addCommand(startCommand);
program.addCommand(configCommand);
program.addCommand(versionCommand);
program.addCommand(statusCommand);
program.addCommand(alertsCommand);
program.addCommand(modelsCommand);

// Global error handling
program.configureOutput({
  writeErr: (str) => logger.error(str),
  writeOut: (str) => logger.info(str)
});

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  logger.error('Uncaught exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Parse arguments
program.parseAsync(process.argv).catch((error) => {
  logger.error('Command failed:', error);
  process.exit(1);
});