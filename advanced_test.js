const { startMonitoring, ConfigManager, AIService } = require('./packages/core/dist');
console.log('=== Advanced AI Sentinel Test ===');

async function runAdvancedTest() {
    try {
        // 1. Test Config Manager
        console.log('1. Testing Config Manager...');
        const config = new ConfigManager();
        const settings = config.getConfig();
        console.log('Config loaded:', settings);
        
        // 2. Test AI Service
        console.log('2. Testing AI Service...');
        const aiService = new AIService();
        console.log('AI Service initialized');
        
        // 3. Test Monitoring
        console.log('3. Testing Monitoring Service...');
        const monitor = startMonitoring({
            projectPath: './',
            monitoringInterval: 3000,
            cloudProvider: 'local'
        });
        
        console.log('Monitoring started. Will stop in 5 seconds...');
        
        // Stop after 5 seconds
        setTimeout(() => {
            if (monitor && monitor.stop) {
                monitor.stop();
                console.log('Monitoring stopped');
            }
            console.log('=== Test completed successfully ===');
        }, 5000);
        
    } catch (error) {
        console.log('Test failed:', error.message);
    }
}

runAdvancedTest();
