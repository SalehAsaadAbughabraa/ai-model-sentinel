const { startMonitoring, scanProject } = require('./packages/core/dist');

console.log('?? Starting Production Service');
console.log('==============================');

const monitor = startMonitoring({
    projectPath: './',
    monitoringInterval: 3000,
    cloudProvider: 'local'
});

scanProject('./').then(result => {
    console.log('? Models detected:', result.models.length);
    console.log('?? Files scanned:', result.stats.filesScanned);
    console.log('? Performance:', result.stats.scanDuration + 'ms');
    console.log('?? Dashboard: http://localhost:3000');
    console.log('?? API: http://localhost:8080');
    console.log('');
    console.log('??? Service running successfully!');
});

setInterval(() => {}, 1000);