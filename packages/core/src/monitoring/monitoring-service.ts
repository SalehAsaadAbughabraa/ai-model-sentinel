import { MonitoringService as BaseService, DriftDetector, PerformanceMonitor, AnomalyDetector } from '.';
import { AlertManager } from '../alerts';
import { ConfigManager, AppConfig } from '../config';
import { DataDriftResult, ModelMetrics, Alert } from '../types';
import { logger } from '../utils/logger';

export class AdvancedMonitoringService extends BaseService {
  private alertManager: AlertManager;
  private config: AppConfig;
  private intervalId?: NodeJS.Timeout;

  constructor(config: Partial<AppConfig> = {}) {
    super();
    const configManager = new ConfigManager(config);
    this.config = configManager.getConfig();
    this.alertManager = new AlertManager(this.config);
  }

  override startMonitoring(): void {
    if (this.getStatus()) {
      logger.warn('Monitoring is already running');
      return;
    }

    super.startMonitoring();
    this.scheduleMonitoring();
    logger.info(`Monitoring started with interval: ${this.config.monitoring.interval}ms`);
  }

  override stopMonitoring(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
    super.stopMonitoring();
    logger.info('Monitoring stopped');
  }

  private scheduleMonitoring(): void {
    this.intervalId = setInterval(() => {
      this.runMonitoringCycle().catch(error => {
        logger.error('Monitoring cycle failed:', error);
      });
    }, this.config.monitoring.interval);
  }

  private async runMonitoringCycle(): Promise<void> {
    try {
      logger.debug('Starting monitoring cycle...');

      // Check for data drift
      const driftResult = DriftDetector.detectDrift();
      if (driftResult.score > this.config.monitoring.driftThreshold) {
        await this.handleDriftDetection(driftResult);
      }

      // Check performance metrics
      const metrics = PerformanceMonitor.trackMetrics();
      if (metrics.accuracy < this.config.monitoring.performanceThreshold) {
        await this.handlePerformanceDegradation(metrics);
      }

      logger.debug('Monitoring cycle completed successfully');

    } catch (error) {
      logger.error('Monitoring cycle failed:', error);
      await this.alertManager.sendAlert({
        id: `error-${Date.now()}`,
        type: 'anomaly',
        severity: 'high',
        message: `Monitoring cycle failed: ${(error as Error).message}`,
        timestamp: new Date(),
        metadata: { error: (error as Error).message }
      });
    }
  }

  private async handleDriftDetection(driftResult: DataDriftResult): Promise<void> {
    const alert: Alert = {
      id: `drift-${Date.now()}`,
      type: 'drift',
      severity: driftResult.severity,
      message: `Data drift detected with score: ${driftResult.score.toFixed(3)}`,
      timestamp: new Date(),
      metadata: driftResult
    };

    await this.alertManager.sendAlert(alert);
    logger.warn(`Data drift alert sent: ${driftResult.score}`);
  }

  private async handlePerformanceDegradation(metrics: ModelMetrics): Promise<void> {
    const alert: Alert = {
      id: `performance-${Date.now()}`,
      type: 'performance',
      severity: 'medium',
      message: `Performance degradation detected. Accuracy: ${metrics.accuracy.toFixed(3)}`,
      timestamp: new Date(),
      metadata: metrics
    };

    await this.alertManager.sendAlert(alert);
    logger.warn(`Performance alert sent. Accuracy: ${metrics.accuracy}`);
  }

  updateConfig(newConfig: Partial<AppConfig>): void {
    const configManager = new ConfigManager({ ...this.config, ...newConfig });
    this.config = configManager.getConfig();
    this.alertManager = new AlertManager(this.config);

    // Restart monitoring with new interval if changed
    if (this.getStatus()) {
      this.stopMonitoring();
      this.startMonitoring();
    }

    logger.info('Configuration updated and monitoring restarted');
  }
}