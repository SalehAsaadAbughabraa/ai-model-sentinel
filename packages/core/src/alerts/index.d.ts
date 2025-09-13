import { Alert } from '../types';
import { AppConfig } from '../config';
export interface AlertProvider {
    sendAlert(alert: Alert): Promise<void>;
    getName(): string;
}
export declare class AlertManager {
    private config;
    private providers;
    constructor(config: AppConfig);
    private initializeProviders;
    sendAlert(alert: Alert): Promise<void>;
}
//# sourceMappingURL=index.d.ts.map