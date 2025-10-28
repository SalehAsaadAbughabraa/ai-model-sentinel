-- AI Model Sentinel v2.0.0 - Production Database Schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table with enhanced security
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    is_verified BOOLEAN DEFAULT FALSE,
    last_login TIMESTAMP,
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    mfa_secret VARCHAR(32),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API keys for programmatic access
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    scopes JSONB DEFAULT '[]',
    last_used TIMESTAMP,
    expires_at TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Scan history with comprehensive tracking
CREATE TABLE IF NOT EXISTS scan_history (
    id SERIAL PRIMARY KEY,
    uuid UUID DEFAULT uuid_generate_v4() UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    scan_id VARCHAR(50) UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    file_name VARCHAR(255),
    file_size BIGINT,
    file_hash_sha256 VARCHAR(64),
    file_hash_md5 VARCHAR(32),
    file_type VARCHAR(100),
    threat_level VARCHAR(20) NOT NULL,
    threat_score DECIMAL(3,2) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL,
    processing_time DECIMAL(8,3) NOT NULL,
    engines_used JSONB,
    engine_results JSONB,
    details JSONB,
    false_positive BOOLEAN DEFAULT FALSE,
    reviewed_by INTEGER REFERENCES users(id),
    reviewed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Security events audit trail
CREATE TABLE IF NOT EXISTS security_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) DEFAULT 'INFO',
    user_id INTEGER REFERENCES users(id),
    ip_address INET,
    user_agent TEXT,
    request_method VARCHAR(10),
    request_path TEXT,
    status_code INTEGER,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Threat intelligence feed
CREATE TABLE IF NOT EXISTS threat_intelligence (
    id SERIAL PRIMARY KEY,
    ioc_type VARCHAR(50) NOT NULL,
    ioc_value TEXT NOT NULL,
    threat_type VARCHAR(100),
    severity VARCHAR(20),
    confidence DECIMAL(3,2),
    source VARCHAR(255),
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System performance metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(12,4) NOT NULL,
    labels JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scan_history_created_at ON scan_history(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scan_history_user_id ON scan_history(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scan_history_threat_level ON scan_history(threat_level);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_scan_history_file_hash ON scan_history(file_hash_sha256);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_events_created_at ON security_events(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_security_events_user_id ON security_events(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_threat_intelligence_ioc ON threat_intelligence(ioc_type, ioc_value);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- Create admin user (password: AdminSecure123!)
INSERT INTO users (username, email, hashed_password, full_name, is_superuser, is_verified) 
VALUES (
    'admin', 
    'admin@aimodelsentinel.com', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhf8/3EgTQ7/7Vl5o8M6R2',
    'System Administrator', 
    TRUE,
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- Insert sample threat intelligence
INSERT INTO threat_intelligence (ioc_type, ioc_value, threat_type, severity, confidence, source) VALUES
('hash', 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'benign', 'LOW', 0.95, 'internal'),
('hash', 'd41d8cd98f00b204e9800998ecf8427e', 'benign', 'LOW', 0.95, 'internal'),
('domain', 'malicious-domain.com', 'phishing', 'HIGH', 0.85, 'external'),
('ip', '192.168.1.100', 'c2_server', 'CRITICAL', 0.90, 'external')
ON CONFLICT DO NOTHING;