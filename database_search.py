"""
Azure SQL Database searcher for querying structured data
"""
import pyodbc
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class DatabaseSearcher:
    """Search Azure SQL Database using natural language queries converted to SQL"""
    
    def __init__(self, server: str, database: str, username: str, password: str, driver: str = "{ODBC Driver 18 for SQL Server}"):
        """
        Initialize database connection
        
        Args:
            server: Azure SQL server name (e.g., 'myserver.database.windows.net')
            database: Database name
            username: Database username
            password: Database password
            driver: ODBC driver (default: ODBC Driver 18 for SQL Server)
        """
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.driver = driver
        self.connection = None
        
        try:
            self._connect()
            logger.info(f"Successfully connected to Azure SQL Database: {database}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _connect(self):
        """Establish database connection"""
        connection_string = (
            f"DRIVER={self.driver};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
            f"Connection Timeout=30;"
        )
        self.connection = pyodbc.connect(connection_string)
    
    def get_schema_info(self) -> str:
        """Get database schema information for context"""
        try:
            cursor = self.connection.cursor()
            
            # Get table names and their columns
            schema_info = []
            
            # Query to get all tables and their columns
            query = """
            SELECT 
                t.TABLE_NAME,
                c.COLUMN_NAME,
                c.DATA_TYPE
            FROM 
                INFORMATION_SCHEMA.TABLES t
                INNER JOIN INFORMATION_SCHEMA.COLUMNS c 
                    ON t.TABLE_NAME = c.TABLE_NAME
            WHERE 
                t.TABLE_TYPE = 'BASE TABLE'
            ORDER BY 
                t.TABLE_NAME, c.ORDINAL_POSITION
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            current_table = None
            for row in rows:
                table_name, column_name, data_type = row
                
                if current_table != table_name:
                    if current_table is not None:
                        schema_info.append("")
                    schema_info.append(f"Table: {table_name}")
                    current_table = table_name
                
                schema_info.append(f"  - {column_name} ({data_type})")
            
            cursor.close()
            return "\n".join(schema_info)
            
        except Exception as e:
            logger.error(f"Error getting schema info: {e}")
            return "Schema information unavailable"
    
    def execute_query(self, sql_query: str, max_rows: int = 100) -> Dict:
        """
        Execute SQL query and return results
        
        Args:
            sql_query: SQL query to execute
            max_rows: Maximum number of rows to return
            
        Returns:
            Dict with columns and rows
        """
        try:
            # Security: Basic SQL injection prevention
            # Only allow SELECT statements
            if not sql_query.strip().upper().startswith('SELECT'):
                return {
                    'error': 'Only SELECT queries are allowed',
                    'columns': [],
                    'rows': []
                }
            
            cursor = self.connection.cursor()
            cursor.execute(sql_query)
            
            # Get column names
            columns = [column[0] for column in cursor.description]
            
            # Fetch results
            rows = cursor.fetchmany(max_rows)
            
            cursor.close()
            
            logger.info(f"Query executed successfully. Returned {len(rows)} rows.")
            
            return {
                'columns': columns,
                'rows': [list(row) for row in rows],
                'row_count': len(rows)
            }
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return {
                'error': str(e),
                'columns': [],
                'rows': []
            }
    
    def search(self, sql_query: str) -> str:
        """
        Execute SQL query and format results as string
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Formatted string with query results
        """
        result = self.execute_query(sql_query)
        
        if 'error' in result:
            return f"Database query error: {result['error']}"
        
        if not result['rows']:
            return "Query executed successfully but returned no results."
        
        # Format results as a table
        output = f"Found {result['row_count']} result(s):\n\n"
        
        # Add column headers
        output += " | ".join(result['columns']) + "\n"
        output += "-" * (len(" | ".join(result['columns']))) + "\n"
        
        # Add rows
        for row in result['rows'][:10]:  # Limit to 10 rows in output
            output += " | ".join(str(val) if val is not None else "NULL" for val in row) + "\n"
        
        if result['row_count'] > 10:
            output += f"\n... and {result['row_count'] - 10} more rows"
        
        return output
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
