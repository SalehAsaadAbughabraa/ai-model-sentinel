import sqlite3
import duckdb

class SimpleDBConnector:
    def __init__(self):
        self.sqlite_conn = sqlite3.connect('enterprise_sentinel_2025.db', check_same_thread=False)
        self.duckdb_conn = duckdb.connect('analytics.duckdb')
        self._is_connected = True
    
    def is_connected(self):
        return self._is_connected
    
    def execute(self, query, params=None, db_type='sqlite'):
        if db_type == 'sqlite':
            cursor = self.sqlite_conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.sqlite_conn.commit()
            return cursor
        else:
            if params:
                return self.duckdb_conn.execute(query, params)
            else:
                return self.duckdb_conn.execute(query)
    
    def close(self):
        self.sqlite_conn.close()
        self.duckdb_conn.close()
        self._is_connected = False

db_connector = SimpleDBConnector()
print('Simple database system ready')
