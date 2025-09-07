"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.versionCommand = void 0;
var commander_1 = require("commander");
var core_1 = require("@ai-model-sentinel/core");
exports.versionCommand = new commander_1.Command()
    .name('version')
    .description('Show version information')
    .action(function () {
    core_1.logger.info('AI Model Sentinel CLI v0.1.0-alpha.0');
    core_1.logger.info('Enterprise AI Model Monitoring Platform');
});
