import pino from 'pino';

// Create a structured logger
export const logger = pino({
  level: process.env.LOG_LEVEL || 'info',
  transport: {
    target: 'pino-pretty',
    options: {
      colorize: true,
      translateTime: 'SYS:standard',
      ignore: 'pid,hostname'
    }
  }
});

// Logger instance with context
export const createLogger = (context: string) => {
  return logger.child({ context });
};