console.log('Testing Core Package...'); 
try { 
    const core = require('./packages/core/dist'); 
    console.log('? Core package loaded successfully'); 
    console.log('Available exports:', Object.keys(core)); 
} catch (error) { 
    console.log('? Error loading core:', error.message); 
} 
