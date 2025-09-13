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
exports.GovernanceService = void 0;
// Governance and compliance functionality
__exportStar(require("./model-validator"), exports);
__exportStar(require("./compliance-checker"), exports);
// Main governance service
var GovernanceService = /** @class */ (function () {
    function GovernanceService() {
    }
    GovernanceService.prototype.validateModel = function () {
        // Placeholder for model validation logic
        return true;
    };
    GovernanceService.prototype.checkCompliance = function () {
        // Placeholder for compliance checks
        return ['GDPR', 'HIPAA'];
    };
    return GovernanceService;
}());
exports.GovernanceService = GovernanceService;
