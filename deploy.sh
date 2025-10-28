#!/bin/bash

# Production Deployment Script for AI Sentinel
set -e

echo "🚀 Deploying AI Sentinel to Production..."

# Validate environment file
if [ ! -f production.env ]; then
    echo "❌ production.env file not found"
    exit 1
fi

# Load environment variables
set -a
source production.env
set +a

# Validate required secrets
if [ "$VAULT_ENABLED" = "false" ] && [ -z "$ENCRYPTION_KEY" ]; then
    echo "❌ ENCRYPTION_KEY is required when Vault is disabled"
    exit 1
fi

# Create necessary directories
mkdir -p $UPLOAD_DIR $FEATURE_CACHE_DIR $BACKUP_DIR
chmod 755 $UPLOAD_DIR $FEATURE_CACHE_DIR $BACKUP_DIR

# Database health check
echo "🔍 Checking database connection..."
python -c "
import sys
try:
    from sqlalchemy import create_engine
    engine = create_engine('${DATABASE_URL}')
    with engine.connect() as conn:
        result = conn.execute('SELECT 1')
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    sys.exit(1)
"

# Start the application
echo "🎯 Starting AI Sentinel application..."
exec gunicorn \
    --bind ${API_HOST}:${API_PORT} \
    --workers ${API_WORKERS} \
    --timeout ${API_TIMEOUT} \
    --access-logfile - \
    --error-logfile - \
    --preload \
    app:app