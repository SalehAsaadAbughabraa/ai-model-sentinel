import { z } from 'zod';
declare const ConfigSchema: z.ZodObject<{
    modelId: z.ZodString;
    monitoring: z.ZodObject<{
        enabled: z.ZodDefault<z.ZodBoolean>;
        interval: z.ZodDefault<z.ZodNumber>;
        driftThreshold: z.ZodDefault<z.ZodNumber>;
        performanceThreshold: z.ZodDefault<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        enabled: boolean;
        interval: number;
        driftThreshold: number;
        performanceThreshold: number;
    }, {
        enabled?: boolean | undefined;
        interval?: number | undefined;
        driftThreshold?: number | undefined;
        performanceThreshold?: number | undefined;
    }>;
    alerts: z.ZodObject<{
        enabled: z.ZodDefault<z.ZodBoolean>;
        providers: z.ZodDefault<z.ZodArray<z.ZodEnum<["console", "slack", "email", "webhook"]>, "many">>;
        slackWebhookUrl: z.ZodOptional<z.ZodString>;
        emailRecipients: z.ZodDefault<z.ZodArray<z.ZodString, "many">>;
    }, "strip", z.ZodTypeAny, {
        enabled: boolean;
        providers: ("console" | "slack" | "email" | "webhook")[];
        emailRecipients: string[];
        slackWebhookUrl?: string | undefined;
    }, {
        enabled?: boolean | undefined;
        providers?: ("console" | "slack" | "email" | "webhook")[] | undefined;
        slackWebhookUrl?: string | undefined;
        emailRecipients?: string[] | undefined;
    }>;
    storage: z.ZodObject<{
        type: z.ZodDefault<z.ZodEnum<["memory", "redis", "postgres"]>>;
        redisUrl: z.ZodOptional<z.ZodString>;
        postgresUrl: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: "memory" | "redis" | "postgres";
        redisUrl?: string | undefined;
        postgresUrl?: string | undefined;
    }, {
        type?: "memory" | "redis" | "postgres" | undefined;
        redisUrl?: string | undefined;
        postgresUrl?: string | undefined;
    }>;
}, "strip", z.ZodTypeAny, {
    modelId: string;
    monitoring: {
        enabled: boolean;
        interval: number;
        driftThreshold: number;
        performanceThreshold: number;
    };
    alerts: {
        enabled: boolean;
        providers: ("console" | "slack" | "email" | "webhook")[];
        emailRecipients: string[];
        slackWebhookUrl?: string | undefined;
    };
    storage: {
        type: "memory" | "redis" | "postgres";
        redisUrl?: string | undefined;
        postgresUrl?: string | undefined;
    };
}, {
    modelId: string;
    monitoring: {
        enabled?: boolean | undefined;
        interval?: number | undefined;
        driftThreshold?: number | undefined;
        performanceThreshold?: number | undefined;
    };
    alerts: {
        enabled?: boolean | undefined;
        providers?: ("console" | "slack" | "email" | "webhook")[] | undefined;
        slackWebhookUrl?: string | undefined;
        emailRecipients?: string[] | undefined;
    };
    storage: {
        type?: "memory" | "redis" | "postgres" | undefined;
        redisUrl?: string | undefined;
        postgresUrl?: string | undefined;
    };
}>;
export type AppConfig = z.infer<typeof ConfigSchema>;
export declare class ConfigManager {
    private config;
    constructor(initialConfig?: Partial<AppConfig>);
    private validateConfig;
    getConfig(): AppConfig;
    updateConfig(newConfig: Partial<AppConfig>): void;
    static loadFromEnv(): AppConfig;
    static loadFromFile(filePath: string): AppConfig;
}
export {};
//# sourceMappingURL=index.d.ts.map