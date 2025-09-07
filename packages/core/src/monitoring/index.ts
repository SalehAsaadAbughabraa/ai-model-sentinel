// Core monitoring functionality
export * from './drift-detector';
export * from './performance-monitor';
export * from './anomaly-detector';

// Main monitoring service
export class MonitoringService {
  private isMonitoring: boolean = false;

  startMonitoring(): void {
    this.isMonitoring = true;
    console.log('Monitoring started...');
  }

  stopMonitoring(): void {
    this.isMonitoring = false;
    console.log('Monitoring stopped...');
  }

  getStatus(): boolean {
    return this.isMonitoring;
  }
}