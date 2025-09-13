"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ConfigManager = void 0;
var zod_1 = require("zod");
var logger_1 = require("../utils/logger");
var fs = require("fs");
// Configuration schema validation
var ConfigSchema = zod_1.z.object({
    modelId: zod_1.z.string().min(1),
    monitoring: zod_1.z.object({
        enabled: zod_1.z.boolean().default(true),
        interval: zod_1.z.number().min(1000).default(5000),
        driftThreshold: zod_1.z.number().min(0).max(1).default(0.1),
        performanceThreshold: zod_1.z.number().min(0).max(1).default(0.8)
    }),
    alerts: zod_1.z.object({
        enabled: zod_1.z.boolean().default(true),
        providers: zod_1.z.array(zod_1.z.enum(['console', 'slack', 'email', 'webhook'])).default(['console']),
        slackWebhookUrl: zod_1.z.string().url().optional(),
        emailRecipients: zod_1.z.array(zod_1.z.string().email()).default([])
    }),
    storage: zod_1.z.object({
        type: zod_1.z.enum(['memory', 'redis', 'postgres']).default('memory'),
        redisUrl: zod_1.z.string().url().optional(),
        postgresUrl: zod_1.z.string().url().optional()
    })
});
var ConfigManager = /** @class */ (function () {
    function ConfigManager(initialConfig) {
        if (initialConfig === void 0) { initialConfig = {}; }
        this.config = this.validateConfig(initialConfig);
    }
    ConfigManager.prototype.validateConfig = function (config) {
        try {
            return ConfigSchema.parse(config);
        }
        catch (error) {
            logger_1.logger.error('Invalid configuration:', error);
            throw new Error('Configuration validation failed');
        }
    };
    ConfigManager.prototype.getConfig = function () {
        return this.config;
    };
    ConfigManager.prototype.updateConfig = function (newConfig) {
        this.config = this.validateConfig(__assign(__assign({}, this.config), newConfig));
        logger_1.logger.info('Configuration updated successfully');
    };
    ConfigManager.loadFromEnv = function () {
        var _a, _b;
        var envConfig = {
            modelId: process.env.MODEL_ID,
            monitoring: {
                enabled: process.env.MONITORING_ENABLED !== 'false',
                interval: process.env.MONITORING_INTERVAL ? parseInt(process.env.MONITORING_INTERVAL) : undefined,
                driftThreshold: process.env.DRIFT_THRESHOLD ? parseFloat(process.env.DRIFT_THRESHOLD) : undefined,
                performanceThreshold: process.env.PERFORMANCE_THRESHOLD ? parseFloat(process.env.PERFORMANCE_THRESHOLD) : undefined
            },
            alerts: {
                enabled: process.env.ALERTS_ENABLED !== 'false',
                providers: (_a = process.env.ALERT_PROVIDERS) === null || _a === void 0 ? void 0 : _a.split(','),
                slackWebhookUrl: process.env.SLACK_WEBHOOK_URL,
                emailRecipients: (_b = process.env.EMAIL_RECIPIENTS) === null || _b === void 0 ? void 0 : _b.split(',')
            },
            storage: {
                type: process.env.STORAGE_TYPE,
                redisUrl: process.env.REDIS_URL,
                postgresUrl: process.env.POSTGRES_URL
            }
        };
        return ConfigSchema.parse(envConfig);
    };
    ConfigManager.loadFromFile = function (filePath) {
        try {
            var fileContent = fs.readFileSync(filePath, 'utf8');
            var configData = JSON.parse(fileContent);
            var configManager = new ConfigManager(configData);
            return configManager.getConfig();
        }
        catch (error) {
            logger_1.logger.error("Failed to load config from file ".concat(filePath, ":"), error);
            throw new Error("Configuration file error: ".concat(error.message));
        }
    };
    return ConfigManager;
}());
exports.ConfigManager = ConfigManager;
