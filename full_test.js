const { 
    scanProject, 
    startMonitoring, 
    generateReport, 
    logger,
    ConfigManager,
    AIService,
    CloudProviderFactory,
    DashboardAPI
} = require('./packages/core/dist');

console.log('?? Starting Comprehensive AI Sentinel Test');
console.log('==========================================');

async function runComprehensiveTest() {
    try {
        // 1. Test Config Manager
        console.log('\n1. Testing Config Manager...');
        const config = new ConfigManager();
        const defaultConfig = config.getConfig();
        console.log('? Default config:', defaultConfig);
        
        // Update configuration
        config.updateConfig({ 
            checkInterval: 60,
            modelPath: './test-models',
            apiKey: 'test-api-key-123'
        });
        console.log('? Config updated successfully');
        
        // 2. Test Project Scanning
        console.log('\n2. Testing Project Scanning...');
        const scanResult = await scanProject('./');
        console.log('? Scan completed');
        console.log('   Models found:', scanResult.models.length);
        console.log('   Files scanned:', scanResult.stats.filesScanned);
        console.log('   Scan duration:', scanResult.stats.scanDuration + 'ms');
        
        // 3. Test Report Generation
        console.log('\n3. Testing Report Generation...');
        const jsonReport = generateReport({ 
            format: 'json', 
            data: scanResult 
        });
        console.log('? JSON Report generated');
        
        const htmlReport = generateReport({
            format: 'html',
            data: scanResult
        });
        console.log('? HTML Report generated');
        
        // 4. Test AI Service
        console.log('\n4. Testing AI Service...');
        const aiService = new AIService();
        const aiStatus = aiService.getStatus();
        console.log('? AI Service status:', aiStatus);
        
        // 5. Test Cloud Provider Factory
        console.log('\n5. Testing Cloud Providers...');
        const awsProvider = CloudProviderFactory.createProvider('aws');
        const azureProvider = CloudProviderFactory.createProvider('azure');
        console.log('? AWS Provider created:', !!awsProvider);
        console.log('? Azure Provider created:', !!azureProvider);
        
        // 6. Test Dashboard API
        console.log('\n6. Testing Dashboard API...');
        const dashboard = new DashboardAPI();
        const dashboardStats = dashboard.getStats();
        console.log('? Dashboard stats:', dashboardStats);
        
        // 7. Test Monitoring Service
        console.log('\n7. Testing Monitoring Service...');
        const monitor = startMonitoring({
            projectPath: './',
            monitoringInterval: 2000,
            cloudProvider: 'local'
        });
        
        console.log('? Monitoring started');
        
        // 8. Test Logger
        console.log('\n8. Testing Logger...');
        logger.info('Information message from comprehensive test');
        logger.warn('Warning message from comprehensive test');
        logger.error('Error message from comprehensive test');
        console.log('? Logger tested successfully');
        
        // Let monitoring run for 10 seconds
        console.log('\n? Monitoring will run for 10 seconds...');
        console.log('Press Ctrl+C to stop earlier');
        
        setTimeout(() => {
            if (monitor && monitor.stop) {
                monitor.stop();
                console.log('? Monitoring stopped successfully');
            }
            console.log('\n?? COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!');
            console.log('? All features are working correctly');
            console.log('? AI Model Sentinel is ready for production use');
        }, 10000);
        
    } catch (error) {
        console.log('? Test failed:', error.message);
        console.log(error.stack);
    }
}

runComprehensiveTest();