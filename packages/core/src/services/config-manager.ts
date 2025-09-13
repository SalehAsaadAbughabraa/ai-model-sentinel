
export interface AppConfig {
  modelPath: string;
  checkInterval: number;
  apiKey?: string;
  alerts?: {
    email?: string;
    slackWebhook?: string;
  };
}

export class ConfigManager {
  private config: AppConfig;

  constructor() {
    this.config = this.loadDefaultConfig();
  }

  private loadDefaultConfig(): AppConfig {
    return {
      modelPath: '.',
      checkInterval: 300,
      apiKey: process.env.API_KEY
    };
  }

  static loadFromEnv(): AppConfig {
    console.log('Loading configuration from environment variables');
    return {
      modelPath: process.env.MODEL_PATH || '.',
      checkInterval: parseInt(process.env.CHECK_INTERVAL || '300'),
      apiKey: process.env.API_KEY,
      alerts: {
        email: process.env.ALERT_EMAIL,
        slackWebhook: process.env.SLACK_WEBHOOK
      }
    };
  }

  static loadFromFile(configPath: string): AppConfig {
    console.log(`Loading configuration from file: ${configPath}`);
    return {
      modelPath: '.',
      checkInterval: 300,
      apiKey: undefined
    };
  }

  loadConfig(configPath: string): AppConfig {
    console.log(`Loading config from: ${configPath}`);
    return this.config;
  }

  saveConfig(config: AppConfig, configPath: string): void {
    console.log(`Saving config to: ${configPath}`);
    this.config = config;
  }

  getConfig(): AppConfig {
    return this.config;
  }

  updateConfig(updates: Partial<AppConfig>): void {
    this.config = { ...this.config, ...updates };
    console.log('Configuration updated');
  }
}