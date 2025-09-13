import pino from 'pino';

// Create a simple logger for CLI environment
export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: process.env.NODE_ENV === 'development' ? {
    target: 'pino-pretty',
    options: {
      colorize: true,
      translateTime: 'SYS:standard',
      ignore: 'pid,hostname'
    }
  } : undefined
});

// Logger instance with context
export const createLogger = (context: string) => {
  return logger.child({ context });
};