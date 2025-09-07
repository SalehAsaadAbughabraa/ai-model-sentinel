import { z } from 'zod';
import { logger } from '../utils/logger';

// Configuration schema validation
const ConfigSchema = z.object({
  modelId: z.string().min(1),
  monitoring: z.object({
    enabled: z.boolean().default(true),
    interval: z.number().min(1000).default(5000),
    driftThreshold: z.number().min(0).max(1).default(0.1),
    performanceThreshold: z.number().min(0).max(1).default(0.8)
  }),
  alerts: z.object({
    enabled: z.boolean().default(true),
    providers: z.array(z.enum(['console', 'slack', 'email', 'webhook'])).default(['console']),
    slackWebhookUrl: z.string().url().optional(),
    emailRecipients: z.array(z.string().email()).default([])
  }),
  storage: z.object({
    type: z.enum(['memory', 'redis', 'postgres']).default('memory'),
    redisUrl: z.string().url().optional(),
    postgresUrl: z.string().url().optional()
  })
});

export type AppConfig = z.infer<typeof ConfigSchema>;

export class ConfigManager {
  private config: AppConfig;

  constructor(initialConfig: Partial<AppConfig> = {}) {
    this.config = this.validateConfig(initialConfig);
  }

  private validateConfig(config: Partial<AppConfig>): AppConfig {
    try {
      return ConfigSchema.parse(config);
    } catch (error) {
      logger.error('Invalid configuration:', error);
      throw new Error('Configuration validation failed');
    }
  }

  getConfig(): AppConfig {
    return this.config;
  }

  updateConfig(newConfig: Partial<AppConfig>): void {
    this.config = this.validateConfig({ ...this.config, ...newConfig });
    logger.info('Configuration updated successfully');
  }

  static loadFromEnv(): AppConfig {
    const envConfig = {
      modelId: process.env.MODEL_ID,
      monitoring: {
        enabled: process.env.MONITORING_ENABLED !== 'false',
        interval: process.env.MONITORING_INTERVAL ? parseInt(process.env.MONITORING_INTERVAL) : undefined,
        driftThreshold: process.env.DRIFT_THRESHOLD ? parseFloat(process.env.DRIFT_THRESHOLD) : undefined,
        performanceThreshold: process.env.PERFORMANCE_THRESHOLD ? parseFloat(process.env.PERFORMANCE_THRESHOLD) : undefined
      },
      alerts: {
        enabled: process.env.ALERTS_ENABLED !== 'false',
        providers: process.env.ALERT_PROVIDERS?.split(','),
        slackWebhookUrl: process.env.SLACK_WEBHOOK_URL,
        emailRecipients: process.env.EMAIL_RECIPIENTS?.split(',')
      },
      storage: {
        type: process.env.STORAGE_TYPE as any,
        redisUrl: process.env.REDIS_URL,
        postgresUrl: process.env.POSTGRES_URL
      }
    };

    return ConfigSchema.parse(envConfig);
  }
}