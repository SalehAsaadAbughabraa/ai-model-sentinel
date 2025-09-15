 
import { io, Socket } from 'socket.io-client';

class RealTimeService {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;

  connect(url: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.socket = io(url, {
        transports: ['websocket'],
        upgrade: true,
        forceNew: true,
        timeout: 10000,
        reconnectionAttempts: this.maxReconnectAttempts
      });

      this.socket.on('connect', () => {
        this.reconnectAttempts = 0;
        resolve();
      });

      this.socket.on('connect_error', (error) => {
        reject(error);
      });

      this.socket.on('disconnect', (reason) => {
        this.handleDisconnect(reason);
      });
    });
  }

  subscribeToMetrics(callback: (data: any) => void): void {
    this.socket?.on('metrics-update', callback);
  }

  subscribeToAlerts(callback: (alert: any) => void): void {
    this.socket?.on('alert-triggered', callback);
  }

  private handleDisconnect(reason: string): void {
    if (reason === 'io server disconnect') {
      this.socket?.connect();
    }
  }

  disconnect(): void {
    this.socket?.disconnect();
  }
}

export const realTimeService = new RealTimeService();