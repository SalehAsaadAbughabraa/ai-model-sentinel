"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
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
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.AdvancedMonitoringService = void 0;
var _1 = require(".");
var alerts_1 = require("../alerts");
var config_1 = require("../config");
var logger_1 = require("../utils/logger");
var AdvancedMonitoringService = /** @class */ (function (_super) {
    __extends(AdvancedMonitoringService, _super);
    function AdvancedMonitoringService(config) {
        if (config === void 0) { config = {}; }
        var _this = _super.call(this) || this;
        var configManager = new config_1.ConfigManager(config);
        _this.config = configManager.getConfig();
        _this.alertManager = new alerts_1.AlertManager(_this.config);
        return _this;
    }
    AdvancedMonitoringService.prototype.startMonitoring = function () {
        if (this.getStatus()) {
            logger_1.logger.warn('Monitoring is already running');
            return;
        }
        _super.prototype.startMonitoring.call(this);
        this.scheduleMonitoring();
        logger_1.logger.info("Monitoring started with interval: ".concat(this.config.monitoring.interval, "ms"));
    };
    AdvancedMonitoringService.prototype.stopMonitoring = function () {
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = undefined;
        }
        _super.prototype.stopMonitoring.call(this);
        logger_1.logger.info('Monitoring stopped');
    };
    AdvancedMonitoringService.prototype.scheduleMonitoring = function () {
        var _this = this;
        this.intervalId = setInterval(function () {
            _this.runMonitoringCycle().catch(function (error) {
                logger_1.logger.error('Monitoring cycle failed:', error);
            });
        }, this.config.monitoring.interval);
    };
    AdvancedMonitoringService.prototype.runMonitoringCycle = function () {
        return __awaiter(this, void 0, void 0, function () {
            var driftResult, metrics, error_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 5, , 7]);
                        logger_1.logger.debug('Starting monitoring cycle...');
                        driftResult = _1.DriftDetector.detectDrift();
                        if (!(driftResult.score > this.config.monitoring.driftThreshold)) return [3 /*break*/, 2];
                        return [4 /*yield*/, this.handleDriftDetection(driftResult)];
                    case 1:
                        _a.sent();
                        _a.label = 2;
                    case 2:
                        metrics = _1.PerformanceMonitor.trackMetrics();
                        if (!(metrics.accuracy < this.config.monitoring.performanceThreshold)) return [3 /*break*/, 4];
                        return [4 /*yield*/, this.handlePerformanceDegradation(metrics)];
                    case 3:
                        _a.sent();
                        _a.label = 4;
                    case 4:
                        logger_1.logger.debug('Monitoring cycle completed successfully');
                        return [3 /*break*/, 7];
                    case 5:
                        error_1 = _a.sent();
                        logger_1.logger.error('Monitoring cycle failed:', error_1);
                        return [4 /*yield*/, this.alertManager.sendAlert({
                                id: "error-".concat(Date.now()),
                                type: 'anomaly',
                                severity: 'high',
                                message: "Monitoring cycle failed: ".concat(error_1.message),
                                timestamp: new Date(),
                                metadata: { error: error_1.message }
                            })];
                    case 6:
                        _a.sent();
                        return [3 /*break*/, 7];
                    case 7: return [2 /*return*/];
                }
            });
        });
    };
    AdvancedMonitoringService.prototype.handleDriftDetection = function (driftResult) {
        return __awaiter(this, void 0, void 0, function () {
            var alert;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        alert = {
                            id: "drift-".concat(Date.now()),
                            type: 'drift',
                            severity: driftResult.severity,
                            message: "Data drift detected with score: ".concat(driftResult.score.toFixed(3)),
                            timestamp: new Date(),
                            metadata: driftResult
                        };
                        return [4 /*yield*/, this.alertManager.sendAlert(alert)];
                    case 1:
                        _a.sent();
                        logger_1.logger.warn("Data drift alert sent: ".concat(driftResult.score));
                        return [2 /*return*/];
                }
            });
        });
    };
    AdvancedMonitoringService.prototype.handlePerformanceDegradation = function (metrics) {
        return __awaiter(this, void 0, void 0, function () {
            var alert;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        alert = {
                            id: "performance-".concat(Date.now()),
                            type: 'performance',
                            severity: 'medium',
                            message: "Performance degradation detected. Accuracy: ".concat(metrics.accuracy.toFixed(3)),
                            timestamp: new Date(),
                            metadata: metrics
                        };
                        return [4 /*yield*/, this.alertManager.sendAlert(alert)];
                    case 1:
                        _a.sent();
                        logger_1.logger.warn("Performance alert sent. Accuracy: ".concat(metrics.accuracy));
                        return [2 /*return*/];
                }
            });
        });
    };
    AdvancedMonitoringService.prototype.updateConfig = function (newConfig) {
        var configManager = new config_1.ConfigManager(__assign(__assign({}, this.config), newConfig));
        this.config = configManager.getConfig();
        this.alertManager = new alerts_1.AlertManager(this.config);
        // Restart monitoring with new interval if changed
        if (this.getStatus()) {
            this.stopMonitoring();
            this.startMonitoring();
        }
        logger_1.logger.info('Configuration updated and monitoring restarted');
    };
    return AdvancedMonitoringService;
}(_1.MonitoringService));
exports.AdvancedMonitoringService = AdvancedMonitoringService;
