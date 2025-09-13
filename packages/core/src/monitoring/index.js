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
exports.MonitoringService = void 0;
// Core monitoring functionality
__exportStar(require("./drift-detector"), exports);
__exportStar(require("./performance-monitor"), exports);
__exportStar(require("./anomaly-detector"), exports);
// Main monitoring service
var MonitoringService = /** @class */ (function () {
    function MonitoringService() {
        this.isMonitoring = false;
    }
    MonitoringService.prototype.startMonitoring = function () {
        this.isMonitoring = true;
        console.log('Monitoring started...');
    };
    MonitoringService.prototype.stopMonitoring = function () {
        this.isMonitoring = false;
        console.log('Monitoring stopped...');
    };
    MonitoringService.prototype.getStatus = function () {
        return this.isMonitoring;
    };
    return MonitoringService;
}());
exports.MonitoringService = MonitoringService;
