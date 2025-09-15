from flask import Flask 
from core.sentinel import Sentinel 
from dashboard import WebDashboard, AlertSystem, MonitoringSystem 
import threading 
 
app = Flask(__name__) 
sentinel = Sentinel() 
dashboard = WebDashboard() 
alerts = AlertSystem() 
monitoring = MonitoringSystem() 
 
@app.route('/predict', methods=['POST']) 
def predict(): 
    # Main prediction endpoint with full protection 
    monitoring.record_request() 
    # Add your prediction logic here 
    return {'status': 'protected'} 
 
def start_services(): 
    dashboard.start() 
    monitoring.start_metrics_server() 
 
if __name__ == '__main__': 
    start_services() 
    app.run(host='0.0.0.0', port=5000, debug=False) 
