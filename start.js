const { startMonitoring, scanProject } = require('./packages/core/dist');

console.log('?? STARTING AI MODEL SENTINEL PRODUCTION SERVICE');
console.log('=================================================');

const monitor = startMonitoring({
  projectPath: './',
  monitoringInterval: 3000,
  cloudProvider: 'local'
});

scanProject('./').then(result => {
  console.log('');
  console.log('? PRODUCTION STATUS: OPERATIONAL');
  console.log('?? AI Models Detected:', result.models.length);
  console.log('?? Files Protected:', result.stats.filesScanned);
  console.log('? Performance:', result.stats.scanDuration + 'ms');
  console.log('?? Dashboard: http://localhost:3000');
  console.log('?? API: http://localhost:8080');
  console.log('');
  console.log('???  REAL-TIME MODEL PROTECTION ACTIVE');
  console.log('? Monitoring interval: 3 seconds');
  console.log('');
  console.log('Press CTRL + C to stop service');
});