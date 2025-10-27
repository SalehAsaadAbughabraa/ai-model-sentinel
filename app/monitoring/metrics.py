"""
AI Model Sentinel v2.0.0 - Advanced Monitoring & Metrics System
Production-Grade Metrics Collection and Monitoring
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from prometheus_client import Counter, Histogram, Gauge, Summary, Info

@dataclass
class MetricData:
    """Metric data structure"""
    value: float
    timestamp: datetime
    labels: Dict[str, str]

class MetricsCollector:
    """
    Advanced metrics collector for monitoring system performance
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        self._lock = threading.RLock()
        
        # Prometheus metrics
        self._init_prometheus_metrics()
        
        # Custom metrics storage
        self._custom_metrics = defaultdict(deque)
        self._engine_metrics = defaultdict(lambda: defaultdict(deque))
        
        # Performance tracking
        self._performance_data = {
            "response_times": deque(maxlen=1000),
            "error_rates": deque(maxlen=100),
            "throughput": deque(maxlen=100)
        }
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        
        # Request metrics
        self.requests_total = Counter(
            'sentinel_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = Histogram(
            'sentinel_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            'sentinel_active_requests',
            'Number of active requests'
        )
        
        # Scan metrics
        self.scans_total = Counter(
            'sentinel_scans_total',
            'Total number of file scans',
            ['threat_level', 'engine']
        )
        
        self.scan_duration = Histogram(
            'sentinel_scan_duration_seconds',
            'Scan duration in seconds',
            ['engine']
        )
        
        self.active_scans = Gauge(
            'sentinel_active_scans',
            'Number of active scans'
        )
        
        # Engine metrics
        self.engine_requests = Counter(
            'sentinel_engine_requests_total',
            'Total engine analysis requests',
            ['engine']
        )
        
        self.engine_errors = Counter(
            'sentinel_engine_errors_total',
            'Total engine errors',
            ['engine']
        )
        
        self.engine_duration = Histogram(
            'sentinel_engine_duration_seconds',
            'Engine analysis duration',
            ['engine']
        )
        
        # System metrics
        self.database_queries = Counter(
            'sentinel_database_queries_total',
            'Total database queries',
            ['operation']
        )
        
        self.database_errors = Counter(
            'sentinel_database_errors_total',
            'Total database errors',
            ['operation']
        )
        
        self.database_duration = Histogram(
            'sentinel_database_duration_seconds',
            'Database operation duration',
            ['operation']
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'sentinel_cache_hits_total',
            'Total cache hits',
            ['type']
        )
        
        self.cache_misses = Counter(
            'sentinel_cache_misses_total',
            'Total cache misses',
            ['type']
        )
        
        # Security metrics
        self.auth_attempts = Counter(
            'sentinel_auth_attempts_total',
            'Total authentication attempts',
            ['result']
        )
        
        self.security_events = Counter(
            'sentinel_security_events_total',
            'Total security events',
            ['type', 'severity']
        )
        
        # Performance gauges
        self.memory_usage = Gauge(
            'sentinel_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'sentinel_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # System info
        self.system_info = Info(
            'sentinel_system_info',
            'System information'
        )
        self.system_info.info({
            'version': '2.0.0',
            'environment': 'production'
        })
    
    def initialize(self):
        """Initialize metrics system"""
        self.start_time = datetime.now()
    
    # Request monitoring
    def record_request(self, method: str = "GET", endpoint: str = "/"):
        """Record incoming request"""
        self.requests_total.labels(method=method, endpoint=endpoint, status_code="").inc()
        self.active_requests.inc()
    
    def record_response(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record request response"""
        self.requests_total.labels(
            method=method, 
            endpoint=endpoint, 
            status_code=str(status_code)
        ).inc()
        
        self.request_duration.labels(
            method=method, 
            endpoint=endpoint
        ).observe(duration)
        
        self.active_requests.dec()
        
        # Store for internal analytics
        with self._lock:
            self._performance_data["response_times"].append(duration)
    
    def record_error(self, error_type: str = "unknown"):
        """Record system error"""
        self.security_events.labels(type=error_type, severity="error").inc()
    
    # Scan monitoring
    def record_scan_start(self):
        """Record scan start"""
        self.active_scans.inc()
    
    def record_scan_completion(self, threat_level: str, engine: str, duration: float):
        """Record scan completion"""
        self.scans_total.labels(
            threat_level=threat_level,
            engine=engine
        ).inc()
        
        self.scan_duration.labels(engine=engine).observe(duration)
        self.active_scans.dec()
    
    # Engine monitoring
    def record_engine_request(self, engine: str):
        """Record engine analysis request"""
        self.engine_requests.labels(engine=engine).inc()
    
    def record_engine_completion(self, engine: str, duration: float):
        """Record engine analysis completion"""
        self.engine_duration.labels(engine=engine).observe(duration)
    
    def record_engine_error(self, engine: str):
        """Record engine error"""
        self.engine_errors.labels(engine=engine).inc()
    
    def record_engine_initialization(self, engine: str):
        """Record engine initialization"""
        self.engine_requests.labels(engine=engine).inc()
    
    def record_engine_analysis(self, engine: str):
        """Record engine analysis"""
        self.engine_requests.labels(engine=engine).inc()
    
    # Database monitoring
    def record_db_query(self, operation: str = "query"):
        """Record database query"""
        self.database_queries.labels(operation=operation).inc()
    
    def record_db_success(self, operation: str = "query", duration: float = 0.0):
        """Record successful database operation"""
        if duration > 0:
            self.database_duration.labels(operation=operation).observe(duration)
    
    def record_db_error(self, operation: str = "query"):
        """Record database error"""
        self.database_errors.labels(operation=operation).inc()
    
    def record_db_connection(self):
        """Record database connection"""
        self.database_queries.labels(operation="connection").inc()
    
    def record_db_duration(self, duration: float, operation: str = "query"):
        """Record database operation duration"""
        self.database_duration.labels(operation=operation).observe(duration)
    
    # Cache monitoring
    def record_cache_hit(self, cache_type: str = "redis"):
        """Record cache hit"""
        self.cache_hits.labels(type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str = "redis"):
        """Record cache miss"""
        self.cache_misses.labels(type=cache_type).inc()
    
    # Authentication monitoring
    def record_auth_attempt(self, result: str = "unknown"):
        """Record authentication attempt"""
        self.auth_attempts.labels(result=result).inc()
    
    def record_security_event(self, event_type: str, severity: str = "info"):
        """Record security event"""
        self.security_events.labels(type=event_type, severity=severity).inc()
    
    # Performance monitoring
    def record_startup_time(self, duration: float):
        """Record system startup time"""
        with self._lock:
            self._performance_data["startup_time"] = duration
    
    def record_memory_usage(self, usage_bytes: int):
        """Record memory usage"""
        self.memory_usage.set(usage_bytes)
    
    def record_cpu_usage(self, usage_percent: float):
        """Record CPU usage"""
        self.cpu_usage.set(usage_percent)
    
    # Custom metrics
    def record_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record custom metric"""
        with self._lock:
            metric_data = MetricData(
                value=value,
                timestamp=datetime.now(),
                labels=labels or {}
            )
            self._custom_metrics[name].append(metric_data)
    
    def get_custom_metric(self, name: str, window_minutes: int = 60) -> List[MetricData]:
        """Get custom metric data within time window"""
        cutoff = datetime.now() - timedelta(minutes=window_minutes)
        
        with self._lock:
            if name not in self._custom_metrics:
                return []
            
            return [
                data for data in self._custom_metrics[name]
                if data.timestamp >= cutoff
            ]
    
    # Analytics and reporting
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        with self._lock:
            response_times = list(self._performance_data["response_times"])
            
            summary = {
                "uptime_seconds": self.get_uptime(),
                "total_requests": self._get_counter_value(self.requests_total),
                "total_scans": self._get_counter_value(self.scans_total),
                "total_errors": self._get_counter_value(self.engine_errors),
                "average_response_time": self._calculate_average(response_times),
                "p95_response_time": self._calculate_percentile(response_times, 95),
                "p99_response_time": self._calculate_percentile(response_times, 99),
                "throughput_rps": self._calculate_throughput(),
                "error_rate": self._calculate_error_rate(),
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "timestamp": datetime.now().isoformat()
            }
            
            return summary
    
    def get_engine_performance(self) -> Dict[str, Any]:
        """Get engine performance metrics"""
        engines = ["quantum", "ml", "behavioral", "signature"]
        performance = {}
        
        for engine in engines:
            performance[engine] = {
                "total_requests": self._get_counter_value(self.engine_requests, {"engine": engine}),
                "total_errors": self._get_counter_value(self.engine_errors, {"engine": engine}),
                "error_rate": self._calculate_engine_error_rate(engine),
                "average_duration": self._get_histogram_avg(self.engine_duration, {"engine": engine})
            }
        
        return performance
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        return {
            "status": "healthy",
            "uptime": self.get_uptime(),
            "active_requests": self._get_gauge_value(self.active_requests),
            "active_scans": self._get_gauge_value(self.active_scans),
            "memory_usage": self._get_gauge_value(self.memory_usage),
            "cpu_usage": self._get_gauge_value(self.cpu_usage),
            "timestamp": datetime.now().isoformat()
        }
    
    # Utility methods
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_total_requests(self) -> int:
        """Get total number of requests"""
        return self._get_counter_value(self.requests_total)
    
    def get_error_rate(self) -> float:
        """Get current error rate"""
        return self._calculate_error_rate()
    
    def _get_counter_value(self, counter, labels: Dict[str, str] = None) -> int:
        """Get counter value (simplified implementation)"""
        # In a real implementation, this would query the actual counter value
        return 0
    
    def _get_gauge_value(self, gauge) -> float:
        """Get gauge value (simplified implementation)"""
        # In a real implementation, this would query the actual gauge value
        return 0.0
    
    def _get_histogram_avg(self, histogram, labels: Dict[str, str] = None) -> float:
        """Get histogram average (simplified implementation)"""
        # In a real implementation, this would query the actual histogram
        return 0.0
    
    def _calculate_average(self, values: List[float]) -> float:
        """Calculate average of values"""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile of values"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = sorted_values[int(index)]
            upper = sorted_values[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _calculate_throughput(self) -> float:
        """Calculate requests per second"""
        uptime = self.get_uptime()
        if uptime == 0:
            return 0.0
        
        total_requests = self.get_total_requests()
        return total_requests / uptime
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate"""
        total_requests = self.get_total_requests()
        if total_requests == 0:
            return 0.0
        
        total_errors = self._get_counter_value(self.engine_errors)
        return total_errors / total_requests
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        hits = self._get_counter_value(self.cache_hits)
        misses = self._get_counter_value(self.cache_misses)
        
        total = hits + misses
        if total == 0:
            return 0.0
        
        return hits / total
    
    def _calculate_engine_error_rate(self, engine: str) -> float:
        """Calculate engine-specific error rate"""
        requests = self._get_counter_value(self.engine_requests, {"engine": engine})
        errors = self._get_counter_value(self.engine_errors, {"engine": engine})
        
        if requests == 0:
            return 0.0
        
        return errors / requests

# Global metrics instance
metrics_collector = MetricsCollector()