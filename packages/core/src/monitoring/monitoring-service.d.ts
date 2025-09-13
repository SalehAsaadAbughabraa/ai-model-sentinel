import { MonitoringService as BaseService } from '.';
import { AppConfig } from '../config';
export declare class AdvancedMonitoringService extends BaseService {
    private alertManager;
    private config;
    private intervalId?;
    constructor(config?: Partial<AppConfig>);
    startMonitoring(): void;
    stopMonitoring(): void;
    private scheduleMonitoring;
    private runMonitoringCycle;
    private handleDriftDetection;
    private handlePerformanceDegradation;
    updateConfig(newConfig: Partial<AppConfig>): void;
}
//# sourceMappingURL=monitoring-service.d.ts.map