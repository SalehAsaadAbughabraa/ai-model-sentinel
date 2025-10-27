import sqlite3
import duckdb
import threading
from contextlib import contextmanager

class AdvancedMultiDBConnector:
    def __init__(self):
        self.sqlite_conn = sqlite3.connect('enterprise_sentinel_2025.db', check_same_thread=False)
        self.duckdb_conn = duckdb.connect('analytics.duckdb')
        self._lock = threading.RLock()
        self._is_connected = True
        self._init_schemas()
    
    def _init_schemas(self):
        self.sqlite_conn.execute('CREATE TABLE IF NOT EXISTS system_metrics (id INTEGER PRIMARY KEY, timestamp DATETIME, engine_name TEXT, metric_name TEXT, metric_value REAL, tags TEXT)')
        self.duckdb_conn.execute('CREATE TABLE IF NOT EXISTS analytics_data (timestamp TIMESTAMP, engine_name VARCHAR, operation VARCHAR, duration_ms DOUBLE, success BOOLEAN)')
    
    def is_connected(self):
        return self._is_connected
    
    @contextmanager
    def get_cursor(self, db_type='sqlite'):
        with self._lock:
            if db_type == 'sqlite':
                cursor = self.sqlite_conn.cursor()
                try:
                    yield cursor
                    self.sqlite_conn.commit()
                except:
                    self.sqlite_conn.rollback()
                    raise
                finally:
                    cursor.close()
            else:
                yield self.duckdb_conn.cursor()
    
    def execute(self, query, params=None, db_type='sqlite'):
        with self.get_cursor(db_type) as cursor:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor
    
    def close(self):
        with self._lock:
            self.sqlite_conn.close()
            self.duckdb_conn.close()
            self._is_connected = False

db_connector = AdvancedMultiDBConnector()
print('Advanced database system ready')
