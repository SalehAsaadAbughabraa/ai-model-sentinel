# Deployment Guide 
 
## Production Deployment 
 
### Prerequisites 
- Python 3.8+ 
- Redis (for session management) 
- Prometheus (for metrics) 
 
### Installation 
 
\`\`\`bash 
git clone https://github.com/SalehAsaadAbughabraa/ai-model-sentinel.git 
cd ai-model-sentinel 
pip install -r requirements.txt 
\`\`\` 
 
### Docker Deployment 
 
\`\`\`bash 
docker build -t ai-sentinel . 
docker run -p 5000:5000 -p 5001:5001 -p 9090:9090 ai-sentinel 
\`\`\` 
 
### Kubernetes Deployment 
 
See \`deployment/kubernetes.yaml\` for Kubernetes configuration. 
