"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.createLogger = exports.logger = void 0;
var pino_1 = require("pino");
// Create a simple logger for CLI environment
exports.logger = (0, pino_1.default)({
    level: process.env.LOG_LEVEL || 'info',
    transport: process.env.NODE_ENV === 'development' ? {
        target: 'pino-pretty',
        options: {
            colorize: true,
            translateTime: 'SYS:standard',
            ignore: 'pid,hostname'
        }
    } : undefined
});
// Logger instance with context
var createLogger = function (context) {
    return exports.logger.child({ context: context });
};
exports.createLogger = createLogger;
