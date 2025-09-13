import { Alert } from '../types';
import { logger } from '../utils/logger';
import { AppConfig } from '../config';

export interface AlertProvider {
  sendAlert(alert: Alert): Promise<void>;
  getName(): string;
}

export class AlertManager {
  private providers: AlertProvider[] = [];

  constructor(private config: AppConfig) {
    this.initializeProviders();
  }

  private initializeProviders(): void {
    if (this.config.alerts.providers.includes('console')) {
      this.providers.push(new ConsoleAlertProvider());
    }

    if (this.config.alerts.providers.includes('slack') && this.config.alerts.slackWebhookUrl) {
      this.providers.push(new SlackAlertProvider(this.config.alerts.slackWebhookUrl));
    }

    if (this.config.alerts.providers.includes('email') && this.config.alerts.emailRecipients.length > 0) {
      this.providers.push(new EmailAlertProvider(this.config.alerts.emailRecipients));
    }

    logger.info(`Initialized ${this.providers.length} alert providers`);
  }

  async sendAlert(alert: Alert): Promise<void> {
    if (!this.config.alerts.enabled) {
      logger.debug('Alerts are disabled, skipping alert notification');
      return;
    }

    const promises = this.providers.map(provider => 
      provider.sendAlert(alert).catch(error => {
        logger.error(`Failed to send alert via ${provider.getName()}:`, error);
      })
    );

    await Promise.all(promises);
    logger.info(`Alert sent via ${this.providers.length} providers`);
  }
}

// Concrete alert providers
class ConsoleAlertProvider implements AlertProvider {
  getName(): string { return 'console'; }

  async sendAlert(alert: Alert): Promise<void> {
    console.log(`🚨 ALERT [${alert.severity.toUpperCase()}]: ${alert.message}`);
    console.log('📊 Metadata:', alert.metadata);
  }
}

class SlackAlertProvider implements AlertProvider {
  constructor(private webhookUrl: string) {}

  getName(): string { return 'slack'; }

  async sendAlert(alert: Alert): Promise<void> {
    // Placeholder for actual Slack integration
    logger.info(`Sending Slack alert to ${this.webhookUrl}: ${alert.message}`);
  }
}

class EmailAlertProvider implements AlertProvider {
  constructor(private recipients: string[]) {}

  getName(): string { return 'email'; }

  async sendAlert(alert: Alert): Promise<void> {
    // Placeholder for actual email integration
    logger.info(`Sending email alert to ${this.recipients.join(', ')}: ${alert.message}`);
  }
}