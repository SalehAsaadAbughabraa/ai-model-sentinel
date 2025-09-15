// ==================== CORE EXPORTS ====================
export { scanProject } from './services/scanner';
export { startMonitoring } from './services/monitor';
export { generateReport } from './utils/reporter';

// ==================== UTILITIES ====================
export { logger } from './utils/logger';
export { ConfigManager } from './services/config-manager';

// ==================== CLOUD CORE ====================
export { CloudProviderFactory } from './cloud';
export type { 
  CloudProviderConfig, 
  DeploymentConfig, 
  MonitoringConfig, 
  CloudStorageConfig, 
  TrainingJobConfig,
  CloudProvider 
} from './cloud/types';

// ==================== API ====================
export { DashboardAPI } from './services/api/dashboardAPI';

// ==================== VERSION INFO ====================
export const VERSION = '0.1.0-alpha.0';
export const LICENSE = 'MIT';
export const AUTHOR = 'AI Model Sentinel Team';
// ==================== AI SERVICE ====================
export { AIService, aiService } from './services/ai-service';
// ==================== REAL-TIME WEBSOCKET ====================
export { 
  WebSocketService, 
  webSocketService, 
  type RealTimeData 
} from './services/websocket/websocket-service';