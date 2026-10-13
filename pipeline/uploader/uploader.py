import os
import sys
from pathlib import Path
import pyodbc
from dotenv import load_dotenv

# Ensure parent directory is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from transformer.main import TransEngine

load_dotenv()


def get_connection():
    driver = os.getenv("MSSQL_DRIVER", "ODBC Driver 18 for SQL Server")
    server = os.getenv("MSSQL_HOST", "localhost")
    port = os.getenv("MSSQL_PORT", "1433")
    database = os.getenv("MSSQL_DATABASE", "test")
    user = os.getenv("MSSQL_USER", "")
    password = os.getenv("MSSQL_PASSWORD", "")
    trusted = os.getenv("MSSQL_TRUSTED_CONNECTION", "yes").lower() in ("yes", "true", "1")

    conn_str = f"DRIVER={{{driver}}};SERVER={server},{port};DATABASE={database};TrustServerCertificate=yes;"
    if trusted or not user:
        conn_str += "Trusted_Connection=yes;"
    else:
        conn_str += f"UID={user};PWD={password};"
    return pyodbc.connect(conn_str)


def upload_data(df=None, table_name: str = None):
    """Fetch transformed df from TransEngine and push to SSMS without recreating table."""
    if df is None:
        engine = TransEngine()
        df = engine.transform()

    table = table_name or os.getenv("MSSQL_TABLE", "sales_analytics")
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Create table only if it doesn't exist yet (keeps table on future runs)
    type_map = {
        "String": "NVARCHAR(255)",
        "Float64": "FLOAT",
        "Int64": "BIGINT",
        "Int32": "INT",
        "Int8": "INT",
        "Boolean": "BIT",
        "Date": "DATE",
        "Datetime": "DATETIME2",
    }
    cols = ", ".join(
        f"[{col}] {type_map.get(str(dtype), 'NVARCHAR(255)')}"
        for col, dtype in zip(df.columns, df.dtypes)
    )
    cursor.execute(f"IF OBJECT_ID('{table}', 'U') IS NULL CREATE TABLE [{table}] ({cols});")

    # 2. Update data in the table without recreating table schema
    cursor.execute(f"TRUNCATE TABLE [{table}];")

    # 3. Batch insert new data
    placeholders = ", ".join(["?"] * len(df.columns))
    insert_sql = f"INSERT INTO [{table}] ({', '.join(f'[{c}]' for c in df.columns)}) VALUES ({placeholders})"

    cursor.fast_executemany = True
    rows = [
        tuple(int(v) if isinstance(v, bool) else v for v in row)
        for row in df.iter_rows()
    ]
    cursor.executemany(insert_sql, rows)
    conn.commit()

    cursor.close()
    conn.close()
    print(f"Successfully updated {len(df)} rows in [{table}].")
    return len(df)


class SSMSUploader:
    """Minimal wrapper for compatibility with DAG/scripts."""
    def upload_file(self, df=None, table_name=None, **kwargs):
        return upload_data(df=df, table_name=table_name)

