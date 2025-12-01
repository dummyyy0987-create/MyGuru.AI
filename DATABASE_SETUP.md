# Database Integration Setup

## Overview
The AI Assistant now includes a Database Agent that can query Azure SQL Database using natural language. When information is not found in Confluence or GitHub, the system automatically queries the database.

## Prerequisites

### 1. Install ODBC Driver
The database connection requires ODBC Driver 18 for SQL Server.

**Windows:**
```bash
# Download and install from Microsoft:
# https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server
```

**macOS:**
```bash
brew install unixodbc
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew install msodbcsql18 mssql-tools18
```

**Linux (Ubuntu/Debian):**
```bash
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql18
```

### 2. Install Python Package
```bash
pip install pyodbc
```

### 3. Configure Database Credentials
Add to your `.env` file:
```env
AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=your_database_name
AZURE_SQL_USERNAME=your_db_username
AZURE_SQL_PASSWORD=your_db_password
```

## How It Works

1. **User asks a question** → Supervisor Agent receives query
2. **Confluence Agent searches** documentation
   - ✅ Found relevant info → Return answer
   - ❌ No relevant info → Go to step 3
3. **GitHub Agent searches** repositories
   - ✅ Found relevant info → Return answer
   - ❌ No relevant info → Go to step 4
4. **Database Agent converts** natural language to SQL and queries database
   - Retrieves structured data from tables
   - Formats and returns results

## Features

- **Text-to-SQL Conversion**: Automatically converts natural language questions to SQL queries
- **Schema Awareness**: Agent knows your database schema and constructs appropriate queries
- **Read-Only Access**: Only SELECT queries are allowed (security)
- **Formatted Results**: Query results are displayed in readable table format
- **Fallback Chain**: Database is only queried when docs/code don't have the answer

## Example Queries

The Database Agent can handle questions like:
- "How many active users do we have?"
- "Show me recent orders from the past week"
- "What are the top 5 products by sales?"
- "List all customers from California"

## Troubleshooting

**Connection Failed:**
- Verify server firewall allows your IP
- Check credentials in `.env` file
- Ensure ODBC driver is installed correctly

**Query Errors:**
- Database agent will explain if tables don't exist
- Check that your database schema is accessible
- Verify SQL Server authentication is enabled

**Agent Not Triggered:**
- Database agent only activates when Confluence AND GitHub don't find answers
- Make sure database credentials are configured in `.env`
- Check logs for "Querying Database agent..." message
