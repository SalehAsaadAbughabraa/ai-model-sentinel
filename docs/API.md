# API Reference 
 
## API Endpoints 
 
### Main Application 
 
- \`POST /predict\` - Process input with security protection 
- \`GET /api/status\` - System status information 
- \`GET /api/metrics\` - Prometheus metrics 
 
### Dashboard Endpoints 
 
- \`GET /\` - Web dashboard 
- \`GET /api/alerts\` - Recent security alerts 
- \`GET /api/threats\` - Threat intelligence data 
 
## Python API 
 
\`\`\`python 
from core.sentinel import Sentinel 
from community import AnonymousClient 
from simulation import AttackSimulator 
 
sentinel = Sentinel() 
result = sentinel.process_input(your_data) 
\`\`\` 
