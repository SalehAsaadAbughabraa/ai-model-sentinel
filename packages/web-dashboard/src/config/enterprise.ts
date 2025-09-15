 
export const EnterpriseConfig = {
  // Real-time configurations
  realTime: {
    refreshInterval: 1000, // 1 second
    maxDataPoints: 5000,
    websocket: {
      reconnect: true,
      timeout: 30000,
      heartbeat: 5000
    }
  },

  // AI-Powered features
  ai: {
    anomalyDetection: true,
    predictiveAnalytics: true,
    autoScaling: true,
    explainableAI: true,
    naturalLanguage: true
  },

  // Multi-cloud configurations
  clouds: {
    aws: { regions: ['us-east-1', 'eu-west-1', 'ap-southeast-1'] },
    azure: { regions: ['eastus', 'westeurope', 'southeastasia'] },
    gcp: { regions: ['us-central1', 'europe-west1', 'asia-southeast1'] },
    ibm: { regions: ['us-south', 'eu-gb', 'jp-tok'] },
    oracle: { regions: ['us-ashburn-1', 'uk-london-1', 'ap-sydney-1'] }
  },

  // Enterprise security
  security: {
    encryption: 'AES-256-GCM',
    ssl: true,
    auditLogs: true,
    compliance: ['GDPR', 'HIPAA', 'SOC2']
  }
};

export type EnterpriseFeature = keyof typeof EnterpriseConfig;