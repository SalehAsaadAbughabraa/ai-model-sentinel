// Main exports for the core package
export * from './monitoring';
export * from './governance';
export * from './types';
export * from './utils/logger';
export * from './config';
export * from './alerts';

// Export the advanced monitoring service as default
export { AdvancedMonitoringService as MonitoringService } from './monitoring/monitoring-service';
export { ConfigManager } from './config';
export { AlertManager } from './alerts';