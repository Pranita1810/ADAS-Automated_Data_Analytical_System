"""
Apache Airflow DAG for ADAS Sales Pipeline.

Pipeline Stages:
1. Extract & Transform: Ingests API data and transforms it using TransEngine.
2. Parquet Backup: Creates a timestamped Parquet backup file in the backup directory.
3. SSMS Upload: Uploads the Parquet backup file into Microsoft SQL Server (SSMS).

Docker Considerations:
- Paths are resolved dynamically using relative paths or environment variables (BACKUP_DIR).
- Database host defaults to 'host.docker.internal' to reach SSMS on the Windows host from Docker containers.
"""

from datetime import datetime, timedelta
from pathlib import Path
import os
import sys

# Add project root to sys.path so modules can be imported inside Docker or locally
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer.main import TransEngine
from backups.backup import BackupEngine
from uploader.uploader import SSMSUploader

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False


def task_transform_data(**context):
    """Stage 1: Ingest & Transform data using TransEngine."""
    print("Starting Data Transformation Stage...")
    engine = TransEngine()
    df = engine.transform(drop_unrecoverable=True)


def task_backup_to_parquet(**context):
    """Stage 2: Save transformed data as a timestamped Parquet backup file."""
    print("Starting Parquet Backup Stage...")
    engine = TransEngine()
    df = engine.transform(drop_unrecoverable=True)

    backup_engine = BackupEngine()
    backup_path = backup_engine.backup_data(df=df, prefix="sales_backup")
    print(f"Parquet backup saved to: {backup_path}")

    # Pass the backup path to the next task via XCom if Airflow is active
    if "ti" in context:
        context["ti"].xcom_push(key="backup_file_path", value=str(backup_path))

    return str(backup_path)


def task_upload_to_ssms(**context):
    """Stage 3: Upload the Parquet backup file into SSMS / SQL Server."""
    print("Starting SSMS Upload Stage...")
    backup_file_path = None

    if "ti" in context:
        ti = context["ti"]
        backup_file_path = ti.xcom_pull(key="backup_file_path", task_ids="backup_parquet_task")

    uploader = SSMSUploader()
    result = uploader.upload_file(
        file_path=backup_file_path,
        table_name=os.getenv("MSSQL_TABLE", "sales_analytics"),
        mode="replace"
    )
    print(f"SSMS Upload complete: {result}")
    return result


if AIRFLOW_AVAILABLE:
    default_args = {
        "owner": "airflow",
        "depends_on_past": False,
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    }

    dag = DAG(
        dag_id="adas_sales_pipeline",
        default_args=default_args,
        description="Extracts, cleans, backs up to Parquet, and uploads sales data to SSMS",
        schedule="@daily",
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["adas", "sales", "parquet", "ssms"],
    )
    t1_transform = PythonOperator(
        task_id="transform_data_task",
        python_callable=task_transform_data,
        dag=dag,
    )

    t2_backup = PythonOperator(
        task_id="backup_parquet_task",
        python_callable=task_backup_to_parquet,
        dag=dag,
    )

    t3_upload = PythonOperator(
        task_id="upload_to_ssms_task",
        python_callable=task_upload_to_ssms,
        dag=dag,
    )

    t1_transform >> t2_backup >> t3_upload


if __name__ == "__main__":
    print("Executing pipeline sequentially (Standalone / Testing Mode)...")
    task_transform_data()
    backup_file = task_backup_to_parquet()
    task_upload_to_ssms()
    print("Pipeline finished successfully!")
