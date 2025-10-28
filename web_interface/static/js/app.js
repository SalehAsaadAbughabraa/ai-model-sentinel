// AI Model Sentinel Enterprise - Frontend Application
class EnterpriseSentinelDashboard {
    constructor() {
        this.metricsInterval = null;
        this.performanceInterval = null;
        this.securityInterval = null;
        this.enginesInterval = null;
        this.startTime = new Date();
        this.isAnalyzing = false;
        this.enginesManager = null;

        this.init();
    }

    init() {
        this.updateDateTime();
        this.startRealTimeUpdates();
        this.setupEventListeners();
        this.loadInitialData();

        // Initialize Engines Manager
        this.enginesManager = new EnginesManager(this);
        this.enginesManager.init();

        // Update datetime every second
        setInterval(() => this.updateDateTime(), 1000);

        console.log('AI Model Sentinel Enterprise Dashboard Initialized');
    }

    updateDateTime() {
        const now = new Date();
        const dateTimeString = now.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        });

        document.getElementById('currentDateTime').textContent = dateTimeString;
        document.getElementById('lastUpdateTime').textContent = `Last updated: ${now.toLocaleTimeString()}`;

        // Update uptime
        const uptime = Math.floor((new Date() - this.startTime) / 1000);
        const hours = Math.floor(uptime / 3600);
        const minutes = Math.floor((uptime % 3600) / 60);
        const seconds = uptime % 60;
        document.getElementById('systemUptime').textContent =
            `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    startRealTimeUpdates() {
        // System metrics every 3 seconds
        this.metricsInterval = setInterval(() => this.fetchSystemMetrics(), 3000);

        // Performance metrics every 5 seconds
        this.performanceInterval = setInterval(() => this.fetchPerformanceMetrics(), 5000);

        // Security status every 10 seconds
        this.securityInterval = setInterval(() => this.fetchSecurityStatus(), 10000);

        // Backup status every 15 seconds
        setInterval(() => this.fetchBackupStatus(), 15000);

        // System stats every 20 seconds
        setInterval(() => this.fetchSystemStats(), 20000);

        // Engines data every 30 seconds
        this.enginesInterval = setInterval(() => this.enginesManager.loadEnginesData(), 30000);

        // Initial fetch
        this.fetchSystemMetrics();
        this.fetchPerformanceMetrics();
        this.fetchSecurityStatus();
        this.fetchBackupStatus();
        this.fetchSystemStats();
    }

    setupEventListeners() {
        // Analysis button
        document.getElementById('analyzeBtn').addEventListener('click', () => this.runSecurityAnalysis());

        // Manual backup button
        document.getElementById('manualBackupBtn').addEventListener('click', () => this.triggerManualBackup());

        // Enter key in model ID field
        document.getElementById('modelId').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.runSecurityAnalysis();
            }
        });
    }

    async fetchSystemMetrics() {
        try {
            const response = await fetch('/api/v1/enterprise/metrics');
            if (!response.ok) throw new Error('Network response was not ok');
            const metrics = await response.json();

            this.updateSystemMetrics(metrics);
        } catch (error) {
            console.error('Error fetching system metrics:', error);
            this.showError('Failed to fetch system metrics');
        }
    }

    async fetchPerformanceMetrics() {
        try {
            const response = await fetch('/api/v1/enterprise/performance');
            if (!response.ok) throw new Error('Network response was not ok');
            const performance = await response.json();

            this.updatePerformanceMetrics(performance);
        } catch (error) {
            console.error('Error fetching performance metrics:', error);
        }
    }

    async fetchSecurityStatus() {
        try {
            const response = await fetch('/api/v1/enterprise/health');
            if (!response.ok) throw new Error('Network response was not ok');
            const health = await response.json();

            this.updateSecurityStatus(health);
        } catch (error) {
            console.error('Error fetching security status:', error);
        }
    }

    async fetchBackupStatus() {
        try {
            const response = await fetch('/api/v1/enterprise/backup/status');
            if (!response.ok) throw new Error('Network response was not ok');
            const backupStatus = await response.json();

            this.updateBackupStatus(backupStatus);
        } catch (error) {
            console.error('Error fetching backup status:', error);
        }
    }

    async fetchSystemStats() {
        try {
            const response = await fetch('/api/v1/enterprise/statistics');
            if (!response.ok) throw new Error('Network response was not ok');
            const stats = await response.json();

            this.updateSystemStats(stats);
        } catch (error) {
            console.error('Error fetching system stats:', error);
        }
    }

    updateSystemMetrics(metrics) {
        // Update main metrics
        document.getElementById('systemHealth').textContent = `${metrics.system_health}%`;
        document.getElementById('healthProgress').style.width = `${metrics.system_health}%`;
        document.getElementById('healthStatus').textContent = this.getHealthStatus(metrics.system_health);

        document.getElementById('securityScore').textContent = `${metrics.security_score}%`;
        document.getElementById('securityProgress').style.width = `${metrics.security_score}%`;
        document.getElementById('securityStatus').textContent = this.getSecurityStatus(metrics.security_score);

        // Update detailed metrics
        document.getElementById('cpuUsage').textContent = `${metrics.cpu_usage}%`;
        document.getElementById('cpuProgress').style.width = `${metrics.cpu_usage}%`;

        document.getElementById('memoryUsage').textContent = `${metrics.memory_usage}%`;
        document.getElementById('memoryProgress').style.width = `${metrics.memory_usage}%`;

        // Calculate memory values (approximate)
        const memoryUsed = (metrics.memory_usage * 16 / 100).toFixed(1); // Assuming 16GB total
        const memoryAvailable = (16 - memoryUsed).toFixed(1);
        document.getElementById('memoryUsed').textContent = memoryUsed;
        document.getElementById('memoryAvailable').textContent = memoryAvailable;

        document.getElementById('diskUsage').textContent = `${metrics.disk_usage}%`;
        document.getElementById('diskProgress').style.width = `${metrics.disk_usage}%`;

        // Calculate disk values (approximate)
        const diskUsed = (metrics.disk_usage * 500 / 100).toFixed(1); // Assuming 500GB total
        const diskFree = (500 - diskUsed).toFixed(1);
        document.getElementById('diskUsed').textContent = diskUsed;
        document.getElementById('diskFree').textContent = diskFree;

        document.getElementById('networkActivity').textContent = `${Math.min(100, metrics.network_activity * 100).toFixed(1)}%`;
        document.getElementById('networkProgress').style.width = `${Math.min(100, metrics.network_activity * 100)}%`;

        // Update performance index
        document.getElementById('performanceValue').textContent = `${metrics.performance_index}%`;
        document.getElementById('performanceProgress').style.width = `${metrics.performance_index}%`;
        document.getElementById('performanceStatus').textContent = this.getPerformanceStatus(metrics.performance_index);

        // Update threat level based on system health and security score
        this.updateThreatLevel(metrics.system_health, metrics.security_score, metrics.threat_level);
    }

    updatePerformanceMetrics(performance) {
        // Already handled in updateSystemMetrics
    }

    updateSecurityStatus(health) {
        const statusElement = document.getElementById('healthStatus');
        if (health.status === 'HEALTHY') {
            statusElement.innerHTML = '<i class="fas fa-check-circle"></i> System Healthy';
            statusElement.style.color = '#27ae60';
        } else {
            statusElement.innerHTML = '<i class="fas fa-exclamation-triangle"></i> System Degraded';
            statusElement.style.color = '#e74c3c';
        }
    }

    updateBackupStatus(backupStatus) {
        const statusElement = document.getElementById('backupStatus');
        const backupInfo = backupStatus.backup_info;

        if (backupStatus.status === 'CONNECTED') {
            statusElement.textContent = 'CONNECTED';
            statusElement.className = 'status connected';
            document.getElementById('backupFiles').textContent = backupInfo.total_files || 0;
            document.getElementById('backupSize').textContent = backupInfo.total_size_mb || '0';
        } else {
            statusElement.textContent = 'DISCONNECTED';
            statusElement.className = 'status disconnected';
            document.getElementById('backupFiles').textContent = '--';
            document.getElementById('backupSize').textContent = '-- MB';
        }
    }

    updateSystemStats(stats) {
        document.getElementById('totalAnalyses').textContent = stats.total_analyses || 0;
        document.getElementById('totalEngines').textContent = stats.total_engines || 0;
    }

    updateThreatLevel(systemHealth, securityScore, threatLevel = null) {
        let threatPercent, statusText;

        if (threatLevel) {
            // Use provided threat level from API
            switch (threatLevel) {
                case 'NEGLIGIBLE':
                    threatPercent = 10;
                    statusText = 'No immediate threats detected';
                    break;
                case 'LOW':
                    threatPercent = 30;
                    statusText = 'Low risk - monitoring recommended';
                    break;
                case 'MEDIUM':
                    threatPercent = 50;
                    statusText = 'Medium risk - review recommended';
                    break;
                case 'HIGH':
                    threatPercent = 75;
                    statusText = 'High risk - immediate action suggested';
                    break;
                case 'CRITICAL':
                    threatPercent = 95;
                    statusText = 'CRITICAL RISK - IMMEDIATE ACTION REQUIRED';
                    break;
                default:
                    threatPercent = 30;
                    statusText = 'Analyzing...';
            }
        } else {
            // Fallback calculation
            const combinedScore = (systemHealth + securityScore) / 2;

            if (combinedScore >= 90) {
                threatLevel = 'NEGLIGIBLE';
                threatPercent = 10;
                statusText = 'No immediate threats detected';
            } else if (combinedScore >= 75) {
                threatLevel = 'LOW';
                threatPercent = 30;
                statusText = 'Low risk - monitoring recommended';
            } else if (combinedScore >= 60) {
                threatLevel = 'MEDIUM';
                threatPercent = 50;
                statusText = 'Medium risk - review recommended';
            } else if (combinedScore >= 40) {
                threatLevel = 'HIGH';
                threatPercent = 75;
                statusText = 'High risk - immediate action suggested';
            } else {
                threatLevel = 'CRITICAL';
                threatPercent = 95;
                statusText = 'CRITICAL RISK - IMMEDIATE ACTION REQUIRED';
            }
        }

        document.getElementById('threatLevel').textContent = threatLevel;
        document.getElementById('threatFill').style.width = `${threatPercent}%`;
        document.getElementById('threatStatus').textContent = statusText;

        // Update threat level class for styling
        const threatElement = document.getElementById('threatLevel');
        threatElement.className = 'threat-level ' + threatLevel;
    }

    async runSecurityAnalysis() {
        if (this.isAnalyzing) return;

        const modelId = document.getElementById('modelId').value.trim() || 'production_model_001';
        const dataSize = parseInt(document.getElementById('dataSize').value) || 1024;

        if (dataSize < 10) {
            this.showError('Data size must be at least 10 samples');
            return;
        }

        this.isAnalyzing = true;
        const analyzeBtn = document.getElementById('analyzeBtn');
        const originalText = analyzeBtn.innerHTML;

        // Show loading state
        analyzeBtn.innerHTML = '<span class="spinner"></span> Analyzing...';
        analyzeBtn.disabled = true;

        try {
            // Generate sample model data
            const modelData = this.generateSampleData(dataSize);

            const response = await fetch('/api/v1/enterprise/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model_id: modelId,
                    model_data: modelData
                })
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const result = await response.json();

            if (result.status === 'SUCCESS') {
                this.displayAnalysisResults(result.analysis);
                this.showSuccess('Security analysis completed successfully');
            } else {
                this.showError(result.message || 'Analysis failed');
            }

        } catch (error) {
            console.error('Error running security analysis:', error);
            this.showError('Failed to run security analysis');
        } finally {
            // Restore button state
            analyzeBtn.innerHTML = originalText;
            analyzeBtn.disabled = false;
            this.isAnalyzing = false;
        }
    }

    generateSampleData(size) {
        // Generate realistic sample data for analysis
        const data = [];
        for (let i = 0; i < size; i++) {
            // Normal distribution around 0.5 with some noise
            const value = 0.5 + (Math.random() - 0.5) * 0.3 + Math.sin(i * 0.1) * 0.1;
            data.push(parseFloat(value.toFixed(4)));
        }
        return data;
    }

    displayAnalysisResults(analysis) {
        const resultsElement = document.getElementById('analysisResults');
        resultsElement.style.display = 'block';

        // Update basic info
        document.getElementById('analysisId').textContent = analysis.analysis_id;
        document.getElementById('healthScore').textContent = (analysis.health_score * 100).toFixed(1) + '%';
        document.getElementById('resultThreatLevel').textContent = analysis.threat_level;
        document.getElementById('analysisTime').textContent = (analysis.analysis_time * 1000).toFixed(0) + ' ms';

        // Update threat level styling
        const threatElement = document.getElementById('resultThreatLevel');
        threatElement.className = 'result-value threat-level ' + analysis.threat_level;

        // Update risk factors
        const riskFactorsList = document.getElementById('riskFactorsList');
        riskFactorsList.innerHTML = '';
        analysis.risk_factors.forEach(factor => {
            const li = document.createElement('li');
            li.textContent = factor;
            riskFactorsList.appendChild(li);
        });

        // Update recommendations
        const recommendationsList = document.getElementById('recommendationsList');
        recommendationsList.innerHTML = '';
        analysis.recommendations.forEach(recommendation => {
            const li = document.createElement('li');
            li.textContent = recommendation;
            recommendationsList.appendChild(li);
        });

        // Scroll to results
        resultsElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    async triggerManualBackup() {
        const backupBtn = document.getElementById('manualBackupBtn');
        const originalText = backupBtn.innerHTML;

        // Show loading state
        backupBtn.innerHTML = '<span class="spinner"></span> Backing up...';
        backupBtn.disabled = true;

        try {
            const response = await fetch('/api/v1/enterprise/backup/now', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            if (!response.ok) throw new Error('Network response was not ok');

            const result = await response.json();

            if (result.status === 'SUCCESS') {
                this.showSuccess('Manual backup completed successfully');
                // Refresh backup status
                this.fetchBackupStatus();
            } else {
                this.showError(result.message || 'Backup failed');
            }

        } catch (error) {
            console.error('Error triggering manual backup:', error);
            this.showError('Failed to trigger manual backup');
        } finally {
            // Restore button state
            backupBtn.innerHTML = originalText;
            backupBtn.disabled = false;
        }
    }

    getHealthStatus(health) {
        if (health >= 90) return 'Excellent';
        if (health >= 75) return 'Good';
        if (health >= 60) return 'Fair';
        return 'Poor';
    }

    getSecurityStatus(score) {
        if (score >= 90) return 'Highly Secure';
        if (score >= 75) return 'Secure';
        if (score >= 60) return 'Moderate';
        return 'Vulnerable';
    }

    getPerformanceStatus(performance) {
        if (performance >= 90) return 'Optimal';
        if (performance >= 75) return 'Good';
        if (performance >= 60) return 'Acceptable';
        return 'Needs Attention';
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : type === 'success' ? 'check-circle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;

        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#e74c3c' : type === 'success' ? '#27ae60' : '#3498db'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            z-index: 1000;
            animation: slideIn 0.3s ease;
            max-width: 400px;
        `;

        document.body.appendChild(notification);

        // Remove after 5 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }

    loadInitialData() {
        // Load system info
        fetch('/api/v1/enterprise/system/info')
            .then(response => response.json())
            .then(data => {
                console.log('System Info:', data);
            })
            .catch(error => {
                console.error('Error loading system info:', error);
            });

        // Load analysis history
        fetch('/api/v1/enterprise/analysis/history')
            .then(response => response.json())
            .then(data => {
                console.log('Analysis History:', data);
            })
            .catch(error => {
                console.error('Error loading analysis history:', error);
            });
    }
}

// 🚀 ENTERPRISE ENGINES MANAGER
class EnginesManager {
    constructor(dashboard) {
        this.dashboard = dashboard;
        this.engines = [];
        this.filteredEngines = [];
    }

    async init() {
        await this.loadEnginesData();
        this.setupEventListeners();
        this.startRealTimeUpdates();
    }

    setupEventListeners() {
        document.getElementById('refreshEngines').addEventListener('click', () => this.loadEnginesData());
        document.getElementById('categoryFilter').addEventListener('change', () => this.filterEngines());
        document.getElementById('statusFilter').addEventListener('change', () => this.filterEngines());
    }

    startRealTimeUpdates() {
        // Update engines stats every 45 seconds
        setInterval(() => this.loadEnginesStats(), 45000);
    }

    async loadEnginesData() {
        try {
            const response = await fetch('/api/v1/enterprise/engines');
            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            
            if (data.status === 'SUCCESS') {
                this.engines = data.engines;
                this.filteredEngines = [...this.engines];
                this.displayEnginesData();
                this.updateEnginesStats();
                this.updateEnginesCount();
                
                // Load detailed stats
                this.loadEnginesStats();
            }
        } catch (error) {
            console.error('Error loading engines data:', error);
            this.dashboard.showError('Failed to load engines data');
        }
    }

    async loadEnginesStats() {
        try {
            const response = await fetch('/api/v1/enterprise/engines/stats');
            if (!response.ok) throw new Error('Network response was not ok');
            
            const stats = await response.json();
            this.updateEnginesStats(stats);
        } catch (error) {
            console.error('Error loading engines stats:', error);
        }
    }

    displayEnginesData() {
        const enginesGrid = document.getElementById('enginesGrid');
        
        if (!this.filteredEngines.length) {
            enginesGrid.innerHTML = '<div class="no-engines">No engines match the current filters</div>';
            return;
        }

        let html = '';
        this.filteredEngines.forEach(engine => {
            const healthPercent = (engine.health * 100).toFixed(1);
            const performancePercent = (engine.performance * 100).toFixed(1);
            const healthClass = this.getHealthClass(engine.health);
            const stars = this.getPerformanceStars(engine.performance);
            
            html += `
                <div class="engine-card" data-category="${engine.category}" data-status="${engine.status}">
                    <div class="engine-header">
                        <div class="engine-name">${engine.name}</div>
                        <div class="engine-status ${engine.status}">${engine.status}</div>
                    </div>
                    <div class="engine-category">${this.formatCategory(engine.category)}</div>
                    <div class="engine-source">${engine.source}</div>
                    
                    <div class="engine-health">
                        <div class="health-label">
                            <span>Health</span>
                            <span>${healthPercent}%</span>
                        </div>
                        <div class="health-bar">
                            <div class="health-fill ${healthClass}" style="width: ${healthPercent}%"></div>
                        </div>
                    </div>

                    <div class="engine-performance">
                        <span class="performance-label">Performance:</span>
                        <div class="performance-stars">${stars}</div>
                        <span style="color: #b8b8b8; font-size: 0.8rem;">${performancePercent}%</span>
                    </div>

                    ${engine.dependencies && engine.dependencies.length ? `
                        <div class="engine-dependencies">
                            ${engine.dependencies.map(dep => `<span class="dependency-tag">${dep}</span>`).join('')}
                        </div>
                    ` : ''}

                    <div class="engine-last-seen">Last seen: ${new Date(engine.last_seen).toLocaleTimeString()}</div>
                </div>
            `;
        });
        
        enginesGrid.innerHTML = html;
    }

    updateEnginesStats(stats = null) {
        const statsElement = document.getElementById('enginesStats');
        
        if (stats) {
            // Use API stats if provided
            let statsHtml = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">${stats.total_engines || 0}</div>
                        <div class="stat-label">Total Engines</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${stats.active_engines || 0}</div>
                        <div class="stat-label">Active Engines</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${Math.round((stats.average_health || 0) * 100)}%</div>
                        <div class="stat-label">Avg Health</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${Math.round((stats.average_performance || 0) * 100)}%</div>
                        <div class="stat-label">Avg Performance</div>
                    </div>
            `;
            
            if (stats.categories) {
                Object.entries(stats.categories).forEach(([category, count]) => {
                    if (count > 0) {
                        statsHtml += `
                            <div class="stat-card">
                                <div class="stat-value">${count}</div>
                                <div class="stat-label">${this.formatCategory(category)}</div>
                            </div>
                        `;
                    }
                });
            }
            
            statsHtml += '</div>';
            statsElement.innerHTML = statsHtml;
        } else {
            // Calculate stats from filtered engines
            const totalEngines = this.filteredEngines.length;
            const activeEngines = this.filteredEngines.filter(e => e.status === 'active').length;
            const avgHealth = totalEngines > 0 ? 
                this.filteredEngines.reduce((sum, e) => sum + e.health, 0) / totalEngines : 0;
            const avgPerformance = totalEngines > 0 ? 
                this.filteredEngines.reduce((sum, e) => sum + e.performance, 0) / totalEngines : 0;

            const categories = {};
            this.filteredEngines.forEach(engine => {
                categories[engine.category] = (categories[engine.category] || 0) + 1;
            });

            let statsHtml = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">${totalEngines}</div>
                        <div class="stat-label">Total Engines</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${activeEngines}</div>
                        <div class="stat-label">Active Engines</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${Math.round(avgHealth * 100)}%</div>
                        <div class="stat-label">Avg Health</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${Math.round(avgPerformance * 100)}%</div>
                        <div class="stat-label">Avg Performance</div>
                    </div>
            `;

            Object.entries(categories).forEach(([category, count]) => {
                statsHtml += `
                    <div class="stat-card">
                        <div class="stat-value">${count}</div>
                        <div class="stat-label">${this.formatCategory(category)}</div>
                    </div>
                `;
            });

            statsHtml += '</div>';
            statsElement.innerHTML = statsHtml;
        }

        // Update categories display
        this.updateCategoriesDisplay();
    }

    updateCategoriesDisplay() {
        const categoriesElement = document.getElementById('enginesCategories');
        const categories = {};

        this.filteredEngines.forEach(engine => {
            categories[engine.category] = (categories[engine.category] || 0) + 1;
        });

        let categoriesHtml = '';
        Object.entries(categories).forEach(([category, count]) => {
            categoriesHtml += `
                <div class="category-item">
                    <div class="category-name">${this.formatCategory(category)}</div>
                    <div class="category-count">${count}</div>
                </div>
            `;
        });

        categoriesElement.innerHTML = categoriesHtml;
    }

    updateEnginesCount() {
        const enginesCount = document.getElementById('enginesCount');
        const totalEngines = document.getElementById('totalEngines');
        
        enginesCount.textContent = `${this.filteredEngines.length} Engines Displayed`;
        if (totalEngines) {
            totalEngines.textContent = this.engines.length;
        }
    }

    filterEngines() {
        const categoryFilter = document.getElementById('categoryFilter').value;
        const statusFilter = document.getElementById('statusFilter').value;

        this.filteredEngines = this.engines.filter(engine => {
            const categoryMatch = categoryFilter === 'all' || engine.category === categoryFilter;
            const statusMatch = statusFilter === 'all' || engine.status === statusFilter;
            return categoryMatch && statusMatch;
        });

        this.displayEnginesData();
        this.updateEnginesStats();
        this.updateEnginesCount();
    }

    getHealthClass(health) {
        if (health >= 0.9) return 'excellent';
        if (health >= 0.7) return 'good';
        if (health >= 0.5) return 'fair';
        return 'poor';
    }

    getPerformanceStars(performance) {
        const starCount = Math.ceil(performance * 5);
        let stars = '';
        for (let i = 0; i < 5; i++) {
            stars += `<i class="fas fa-star star" style="${i < starCount ? 'color: #f1c40f' : 'color: #555'}"></i>`;
        }
        return stars;
    }

    formatCategory(category) {
        const categoryMap = {
            'ai_ml': 'AI/ML Engine',
            'security': 'Security Engine',
            'quantum': 'Quantum Engine',
            'data': 'Data Engine',
            'monitoring': 'Monitoring Engine',
            'analytics': 'Analytics Engine',
            'fusion': 'Fusion Engine',
            'testing': 'Testing Engine'
        };
        return categoryMap[category] || category;
    }
}

// Add CSS for notifications and spinner
const notificationStyles = `
@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

@keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
}

.notification-content {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.notification-content i {
    font-size: 1.2rem;
}

.spinner {
    border: 2px solid #f3f3f3;
    border-top: 2px solid #3498db;
    border-radius: 50%;
    width: 16px;
    height: 16px;
    animation: spin 1s linear infinite;
    display: inline-block;
    margin-right: 0.5rem;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.status.connected {
    background: rgba(39, 174, 96, 0.2);
    color: #27ae60;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    border: 1px solid #27ae60;
}

.status.disconnected {
    background: rgba(231, 76, 60, 0.2);
    color: #e74c3c;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    border: 1px solid #e74c3c;
}

.threat-level.NEGLIGIBLE {
    background: rgba(39, 174, 96, 0.2);
    color: #27ae60;
    border: 1px solid #27ae60;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
}
.threat-level.LOW {
    background: rgba(241, 196, 15, 0.2);
    color: #f1c40f;
    border: 1px solid #f1c40f;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
}
.threat-level.MEDIUM {
    background: rgba(243, 156, 18, 0.2);
    color: #f39c12;
    border: 1px solid #f39c12;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
}
.threat-level.HIGH {
    background: rgba(231, 76, 60, 0.2);
    color: #e74c3c;
    border: 1px solid #e74c3c;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
}
.threat-level.CRITICAL {
    background: rgba(192, 57, 43, 0.2);
    color: #c0392b;
    border: 1px solid #c0392b;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    animation: blink 1s infinite;
}

@keyframes blink {
    0%, 50% { opacity: 1; }
    51%, 100% { opacity: 0.5; }
}
`;

const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EnterpriseSentinelDashboard();
});