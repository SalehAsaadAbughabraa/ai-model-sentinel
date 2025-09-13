# AI Model Sentinel 🛡️

ai-model-sentinel: Lightweight open-source toolkit for monitoring AI models in production. Detects data drift, performance issues, and provides actionable retraining recommendations. Features smart alerts, dashboard visualization, and works with any ML framework.

## Features ✨

- 🔍 **Scan Projects** – Detect AI models in your codebase
- 📊 **Multi-Format Support** – Works with TensorFlow (.pb, .h5), PyTorch (.pt, .pth), and ONNX (.onnx)
- ⚡ **Fast & Lightweight** – Minimal dependencies, fast scanning
- 🛡️ **Security Focus** – Identify potentially unsafe model files
- 📁 **Flexible Pathing** – Scan specific directories or exclude patterns
- 📈 **Dashboard Visualization** - Monitor model performance and data drift
- 🔔 **Smart Alerts** - Get notified of issues in production
- 🎯 **Retraining Recommendations** - Actionable insights for model improvement

## Installation 📦

```bash
npm install -g ai-model-sentinel

Quick Start 🚀
bash
# Scan your project for AI models
ai-sentinel scan ./your-project

# Start monitoring dashboard
ai-sentinel dashboard

# Set up monitoring for production
ai-sentinel monitor --path ./models --api-key your-key
Usage Examples 📝
Basic Scanning
bash
# Scan current directory
ai-sentinel scan

# Scan specific path
ai-sentinel scan ./src/models

# Scan with exclude pattern
ai-sentinel scan ./project --exclude "**/node_modules/**"
Advanced Monitoring
bash
# Start monitoring service
ai-sentinel monitor --path ./production-models

# With custom configuration
ai-sentinel monitor --config ./sentinel-config.json
Dashboard
bash
# Start web dashboard on default port (3000)
ai-sentinel dashboard

# Custom port
ai-sentinel dashboard --port 8080
Configuration ⚙️
Create a sentinel.config.json file:

json
{
  "scanPaths": ["./src", "./models"],
  "excludePatterns": ["**/node_modules/**", "**/test/**"],
  "modelFormats": [".pb", ".h5", ".pt", ".pth", ".onnx"],
  "monitoring": {
    "enabled": true,
    "checkInterval": 3600,
    "apiEndpoint": "https://api.your-service.com"
  },
  "alerts": {
    "email": "team@your-company.com",
    "slackWebhook": "https://hooks.slack.com/your-webhook"
  }
}
Supported Model Formats 🧩
TensorFlow: .pb, .h5, .keras

PyTorch: .pt, .pth

ONNX: .onnx

SavedModel directories

TensorFlow.js models

API Reference 🔌
JavaScript API
javascript
const { scanProject, startMonitoring } = require('ai-model-sentinel');

// Scan for models
const results = await scanProject('./project-path');

// Start monitoring
const monitor = await startMonitoring({
  path: './models',
  onDriftDetected: (data) => console.log('Drift detected:', data)
});
REST API
bash
# Scan endpoint
curl -X POST http://localhost:3000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"path": "./project"}'

# Monitoring status
curl http://localhost:3000/api/status
Architecture 🏗️
text
ai-model-sentinel/
├── packages/
│   ├── cli/           # Command-line interface
│   ├── core/          # Core scanning & monitoring logic
│   └── web-dashboard/ # React-based dashboard
├── plugins/           # Format-specific detectors
└── docs/             # Documentation
Development 🛠️
bash
# Clone the repository
git clone https://github.com/your-username/ai-model-sentinel.git

# Install dependencies
npm install

# Build all packages
npm run build

# Run tests
npm test

# Develop with hot-reload
npm run dev
Contributing 🤝
We welcome contributions! Please see our Contributing Guide for details.

Fork the repository

Create a feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

License 📄
This project is licensed under the MIT License - see the LICENSE file for details.

Support 💬
📖 Documentation

🐛 Issue Tracker

💬 Discussions

📧 Email Support
