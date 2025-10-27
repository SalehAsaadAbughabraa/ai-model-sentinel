# monitoring/realtime_monitor.py
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque
import psutil

logger = logging.getLogger(__name__)

class MonitorStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    OFFLINE = "offline"

@dataclass
class MonitoringConfig:
    check_interval: int = 30
    alert_threshold: int = 3
    metrics_retention: int = 3600
    enable_alerts: bool = True
    performance_threshold: float = 0.8
    memory_threshold: float = 0.85
    cpu_threshold: float = 0.9

class PerformanceMetrics:
    def __init__(self):
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.disk_io = deque(maxlen=100)
        self.network_io = deque(maxlen=100)
        
    def record_cpu(self, usage: float):
        self.cpu_usage.append(usage)
        
    def record_memory(self, usage: float):
        self.memory_usage.append(usage)
        
    def record_disk_io(self, read_bytes: int, write_bytes: int):
        self.disk_io.append((read_bytes, write_bytes))
        
    def record_network_io(self, bytes_sent: int, bytes_recv: int):
        self.network_io.append((bytes_sent, bytes_recv))
        
    def get_stats(self) -> Dict[str, Any]:
        return {
            'cpu_avg': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0,
            'cpu_max': max(self.cpu_usage) if self.cpu_usage else 0,
            'memory_avg': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'total_metrics': len(self.cpu_usage)
        }

class RealTimeMonitor:
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        self.metrics = PerformanceMetrics()
        self.alert_count = 0
        self.status = MonitorStatus.HEALTHY
        self.is_monitoring = False
        self.monitor_thread = None
        self.custom_metrics = {}
        self.alert_handlers = []
        
    def start_monitoring(self):
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.status = MonitorStatus.HEALTHY
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Real-time monitoring started")
        
    def stop_monitoring(self):
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.status = MonitorStatus.OFFLINE
        logger.info("Real-time monitoring stopped")
        
    def _monitoring_loop(self):
        while self.is_monitoring:
            try:
                self._collect_system_metrics()
                self._check_thresholds()
                time.sleep(self.config.check_interval)
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)
                
    def _collect_system_metrics(self):
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.record_cpu(cpu_percent / 100.0)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.metrics.record_memory(memory.percent / 100.0)
            
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            self.metrics.record_disk_io(disk_io.read_bytes, disk_io.write_bytes)
            
        # Network I/O
        net_io = psutil.net_io_counters()
        if net_io:
            self.metrics.record_network_io(net_io.bytes_sent, net_io.bytes_recv)
            
    def _check_thresholds(self):
        alerts = []
        
        stats = self.metrics.get_stats()
        
        # Check CPU threshold
        if stats['cpu_avg'] > self.config.cpu_threshold:
            alerts.append(f"High CPU usage: {stats['cpu_avg']:.1%}")
            
        # Check memory threshold
        if stats['memory_avg'] > self.config.memory_threshold:
            alerts.append(f"High memory usage: {stats['memory_avg']:.1%}")
            
        # Trigger alerts
        for alert in alerts:
            self._trigger_alert(alert)
            
    def record_metric(self, metric_name: str, value: float, tags: Dict[str, Any] = None):
        metric_data = {
            "timestamp": time.time(),
            "metric": metric_name,
            "value": value,
            "tags": tags or {}
        }
        
        if metric_name not in self.custom_metrics:
            self.custom_metrics[metric_name] = deque(maxlen=100)
            
        self.custom_metrics[metric_name].append(metric_data)
        
        # Check for custom metric alerts
        self._check_custom_metric_alerts(metric_name, value)
        
    def _check_custom_metric_alerts(self, metric_name: str, value: float):
        if "error" in metric_name.lower() and value > 0.1:
            self._trigger_alert(f"High error rate in {metric_name}: {value:.1%}")
        elif "latency" in metric_name.lower() and value > 1.0:
            self._trigger_alert(f"High latency in {metric_name}: {value:.2f}s")
            
    def _trigger_alert(self, message: str):
        self.alert_count += 1
        logger.warning(f"ALERT: {message}")
        
        # Execute alert handlers
        for handler in self.alert_handlers:
            try:
                handler(message, self.alert_count)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Update status based on alert count
        if self.alert_count >= self.config.alert_threshold:
            self.status = MonitorStatus.CRITICAL
        elif self.alert_count > 0:
            self.status = MonitorStatus.WARNING
        else:
            self.status = MonitorStatus.HEALTHY
            
    def add_alert_handler(self, handler):
        self.alert_handlers.append(handler)
        
    def get_status(self) -> Dict[str, Any]:
        stats = self.metrics.get_stats()
        
        return {
            "status": self.status.value,
            "alert_count": self.alert_count,
            "is_monitoring": self.is_monitoring,
            "system_metrics": stats,
            "custom_metrics_count": len(self.custom_metrics),
            "config": {
                "check_interval": self.config.check_interval,
                "alert_threshold": self.config.alert_threshold
            }
        }
        
    def get_metrics(self, metric_name: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        if metric_name:
            return list(self.custom_metrics.get(metric_name, []))[-limit:]
        else:
            all_metrics = []
            for metrics in self.custom_metrics.values():
                all_metrics.extend(metrics[-limit:])
            return sorted(all_metrics, key=lambda x: x['timestamp'])[-limit:]
    
    def reset_alerts(self):
        self.alert_count = 0
        self.status = MonitorStatus.HEALTHY
        logger.info("Alert counters reset")
        
    def get_performance_report(self) -> Dict[str, Any]:
        stats = self.metrics.get_stats()
        
        return {
            "timestamp": time.time(),
            "status": self.status.value,
            "performance_metrics": stats,
            "alerts_total": self.alert_count,
            "recommendations": self._generate_recommendations(stats)
        }
        
    def _generate_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        if stats['cpu_avg'] > 0.8:
            recommendations.append("Consider optimizing CPU-intensive operations")
            
        if stats['memory_avg'] > 0.8:
            recommendations.append("Monitor memory usage and consider optimization")
            
        if not recommendations:
            recommendations.append("System performance is within normal parameters")
            
        return recommendations

# Utility functions
def create_default_monitor() -> RealTimeMonitor:
    return RealTimeMonitor()

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if args and hasattr(args[0], 'monitor') and isinstance(args[0].monitor, RealTimeMonitor):
                args[0].monitor.record_metric(
                    f"function_{func.__name__}_time", 
                    execution_time, 
                    {"function": func.__name__}
                )
                
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            
            if args and hasattr(args[0], 'monitor') and isinstance(args[0].monitor, RealTimeMonitor):
                args[0].monitor.record_metric(
                    f"function_{func.__name__}_error", 
                    1, 
                    {"function": func.__name__, "error": str(e)}
                )
            raise e
    return wrapper