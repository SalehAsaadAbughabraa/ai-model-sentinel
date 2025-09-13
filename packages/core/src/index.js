"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __exportStar = (this && this.__exportStar) || function(m, exports) {
    for (var p in m) if (p !== "default" && !Object.prototype.hasOwnProperty.call(exports, p)) __createBinding(exports, m, p);
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AlertManager = exports.ConfigManager = exports.MonitoringService = void 0;
// Main exports for the core package
__exportStar(require("./monitoring"), exports);
__exportStar(require("./governance"), exports);
__exportStar(require("./types"), exports);
__exportStar(require("./utils/logger"), exports);
__exportStar(require("./config"), exports);
__exportStar(require("./alerts"), exports);
// Export the advanced monitoring service as default
var monitoring_service_1 = require("./monitoring/monitoring-service");
Object.defineProperty(exports, "MonitoringService", { enumerable: true, get: function () { return monitoring_service_1.AdvancedMonitoringService; } });
var config_1 = require("./config");
Object.defineProperty(exports, "ConfigManager", { enumerable: true, get: function () { return config_1.ConfigManager; } });
var alerts_1 = require("./alerts");
Object.defineProperty(exports, "AlertManager", { enumerable: true, get: function () { return alerts_1.AlertManager; } });
