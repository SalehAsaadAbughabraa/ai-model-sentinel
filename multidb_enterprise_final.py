import sqlite3
import duckdb

class MultiDBConnector:
    def __init__(self):
        self.sqlite_conn = sqlite3.connect('enterprise_sentinel_2025.db')
        self.duckdb_conn = duckdb.connect('analytics.duckdb')
        self.connection = True
    
    def is_connected(self):
        try:
            return self.connection is not None
        except AttributeError:
            return False
    
    def execute(self, query, params=None):
        cursor = self.sqlite_conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor
    
    def close(self):
        self.sqlite_conn.close()
        self.duckdb_conn.close()
        self.connection = False

print('MultiDBConnector with is_connected method ready')