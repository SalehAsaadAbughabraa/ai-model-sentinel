import { Server } from 'socket.io';
import { createServer } from 'http';
import { logger } from '../../utils/logger';

export interface RealTimeData {
  timestamp: Date;
  metrics: {
    throughput: number;
    latency: number;
    errorRate: number;
    memoryUsage: number;
    cpuUsage: number;
  };
  predictions: {
    nextHour: number;
    anomalyRisk: number;
    recommendedAction: string;
  };
}

export class WebSocketService {
  private static instance: WebSocketService;
  private io: Server | null = null;
  private connectedClients: Set<string> = new Set();
  private metricsHistory: RealTimeData[] = [];

  static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService();
    }
    return WebSocketService.instance;
  }

  initialize(port: number = 3002): void {
    const httpServer = createServer();
    this.io = new Server(httpServer, {
      cors: {
        origin: "http://localhost:3000",
        methods: ["GET", "POST"]
      }
    });

    this.setupEventHandlers();
    
    httpServer.listen(port, () => {
      logger.info(`🚀 WebSocket Server running on port ${port}`);
      console.log(`📡 Real-time WebSocket: ws://localhost:${port}`);
      this.startMetricsBroadcast();
    });
  }

  private setupEventHandlers(): void {
    if (!this.io) return;

    this.io.on('connection', (socket) => {
      console.log('🔗 Client connected:', socket.id);
      this.connectedClients.add(socket.id);

      // Send initial metrics history
      socket.emit('metrics-history', this.metricsHistory.slice(-100));

      socket.on('disconnect', () => {
        console.log('🔌 Client disconnected:', socket.id);
        this.connectedClients.delete(socket.id);
      });

      socket.on('subscribe-metrics', (data) => {
        console.log('📊 Client subscribed to metrics:', socket.id);
        socket.join('metrics-room');
      });

      socket.on('unsubscribe-metrics', () => {
        console.log('📊 Client unsubscribed from metrics:', socket.id);
        socket.leave('metrics-room');
      });
    });
  }

  private startMetricsBroadcast(): void {
    // Simulate real-time metrics updates
    setInterval(() => {
      const realTimeData: RealTimeData = {
        timestamp: new Date(),
        metrics: {
          throughput: Math.random() * 1000,
          latency: Math.random() * 100,
          errorRate: Math.random() * 5,
          memoryUsage: Math.random() * 100,
          cpuUsage: Math.random() * 100
        },
        predictions: {
          nextHour: Math.random() * 1000,
          anomalyRisk: Math.random() * 100,
          recommendedAction: this.getRecommendedAction()
        }
      };

      this.metricsHistory.push(realTimeData);
      
      // Keep only last 1000 data points
      if (this.metricsHistory.length > 1000) {
        this.metricsHistory = this.metricsHistory.slice(-1000);
      }

      // Broadcast to all connected clients
      if (this.io) {
        this.io.to('metrics-room').emit('metrics-update', realTimeData);
        this.io.emit('clients-count', this.connectedClients.size);
      }

    }, 2000); // Update every 2 seconds
  }

  private getRecommendedAction(): string {
    const actions = [
      'Scale up instances',
      'Scale down instances',
      'Check model performance',
      'No action needed',
      'Investigate anomalies',
      'Optimize model',
      'Increase monitoring frequency'
    ];
    return actions[Math.floor(Math.random() * actions.length)];
  }

  getConnectedClients(): number {
    return this.connectedClients.size;
  }

  broadcastAlert(alert: any): void {
    if (this.io) {
      this.io.emit('alert', {
        ...alert,
        timestamp: new Date(),
        severity: alert.severity || 'warning'
      });
    }
  }
}

export const webSocketService = WebSocketService.getInstance();