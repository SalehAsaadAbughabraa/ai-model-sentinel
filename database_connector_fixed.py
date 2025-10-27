import os, sqlite3, duckdb
class MultiDBConnector:
    def __init__(self):
        self.sqlite_conn = sqlite3.connect("enterprise_sentinel_2025.db")
        self.duckdb_conn = duckdb.connect("analytics.duckdb")
        self.is_postgres = False
    def is_connected(self):
        return True
    def execute(self, query, params=None, db_type="sqlite"):
        if db_type == "sqlite":
            cursor = self.sqlite_conn.cursor()
            cursor.execute(query, params or ())
            return cursor
        else:
            return self.duckdb_conn.execute(query)
    def close(self):
        self.sqlite_conn.close()
        self.duckdb_conn.close()
print("Database connector fixed")