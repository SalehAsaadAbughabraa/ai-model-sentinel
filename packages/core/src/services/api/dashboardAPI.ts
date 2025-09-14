import express from 'express';
import { CloudProviderFactory } from '../../cloud';
import { logger } from '../../utils/logger';

export class DashboardAPI {
  private app: express.Application;
  private port: number;

  constructor(port: number = 3001) {
    this.app = express();
    this.port = port;
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware(): void {
    this.app.use(express.json());
    this.app.use((req, res, next) => {
      res.header('Access-Control-Allow-Origin', 'http://localhost:3000');
      res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
      next();
    });
  }

  private setupRoutes(): void {
    // Health check endpoint
    this.app.get('/api/health', (req, res) => {
      res.json({ status: 'healthy', timestamp: new Date().toISOString() });
    });

    // Get cloud providers
    this.app.get('/api/providers', (req, res) => {
      const providers = CloudProviderFactory.getAvailableProviders();
      res.json({ providers });
    });

    // Get provider status
    this.app.post('/api/provider/status', async (req, res) => {
      try {
        const { provider, credentials } = req.body;
        const cloudProvider = CloudProviderFactory.createProvider({
          provider,
          credentials
        });

        await cloudProvider.initialize();
        const status = await cloudProvider.healthCheck();
        
        res.json({ provider, status, healthy: status });
      } catch (error) {
res.status(500).json({ error: (error as Error).message });
      }
    });

    // Get metrics data
    this.app.get('/api/metrics', (req, res) => {
      // Simulated metrics data
      const metrics = {
        timestamp: new Date().toISOString(),
        throughput: Math.random() * 1000,
        latency: Math.random() * 100,
        errorRate: Math.random() * 5,
        cost: Math.random() * 50
      };
      res.json(metrics);
    });
  }

  start(): void {
    this.app.listen(this.port, () => {
      logger.info(`Dashboard API server running on port ${this.port}`);
      console.log(`🚀 API Server: http://localhost:${this.port}`);
      console.log(`📊 Health Check: http://localhost:${this.port}/api/health`);
    });
  }
}

if (require.main === module) {
  const api = new DashboardAPI();
  api.start();
}