#!/usr/bin/env node
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var commander_1 = require("commander");
var start_1 = require("../commands/start");
var config_1 = require("../commands/config");
var version_1 = require("../commands/version");
var core_1 = require("@ai-model-sentinel/core");
var program = new commander_1.Command();
program
    .name('ai-model-sentinel')
    .description('Enterprise AI Model Monitoring CLI')
    .version('0.1.0-alpha.0');
// Add commands
program.addCommand(start_1.startCommand);
program.addCommand(config_1.configCommand);
program.addCommand(version_1.versionCommand);
// Global error handling
program.configureOutput({
    writeErr: function (str) { return core_1.logger.error(str); },
    writeOut: function (str) { return core_1.logger.info(str); }
});
// Handle uncaught errors
process.on('uncaughtException', function (error) {
    core_1.logger.error('Uncaught exception:', error);
    process.exit(1);
});
process.on('unhandledRejection', function (reason, promise) {
    core_1.logger.error('Unhandled rejection at:', promise, 'reason:', reason);
    process.exit(1);
});
// Parse arguments
program.parseAsync(process.argv).catch(function (error) {
    core_1.logger.error('Command failed:', error);
    process.exit(1);
});
