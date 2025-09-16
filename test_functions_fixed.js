const { scanProject, generateReport, logger } = require('./packages/core/dist');
console.log('Testing core functions...');

// Test scanProject
scanProject('./').then(result => {
    console.log('Scan result:', result);
}).catch(e => {
    console.log('Scan error:', e.message);
});

// Test generateReport
try {
    const report = generateReport({ format: 'json' });
    console.log('Report type:', typeof report);
} catch (e) {
    console.log('Report error:', e.message);
}

// Test logger
try {
    logger.info('Test log message');
    console.log('Logger working');
} catch (e) {
    console.log('Logger error:', e.message);
}