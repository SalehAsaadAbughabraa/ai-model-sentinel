// test_functions.js 
const { scanProject, startMonitoring, generateReport, logger } = require('./packages/core/dist'); 
console.log('Testing core functions...'); 
try { 
    const scanResult = scanProject('./'); 
    console.log('? scanProject result:', scanResult); 
} catch (e) { 
    console.log('? scanProject error:', e.message); 
} 
try { 
    const report = generateReport(); 
    console.log('? generateReport result:', typeof report); 
} catch (e) { 
    console.log('? generateReport error:', e.message); 
} 
try { 
    logger.info('Test log message from AI Sentinel'); 
    console.log('? Logger working'); 
} catch (e) { 
    console.log('? Logger error:', e.message); 
} 
