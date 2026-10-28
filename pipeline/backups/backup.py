"""
Backup Engine for ADAS (Automated Data Analytical System).

This module handles:
1. Exporting clean sales data to Parquet format.
2. Generating timestamped backup file names (e.g. backup_YYYYMMDD_HHMMSS.parquet).
3. Docker-friendly path resolution and directory management.
4. Backup listing and retrieving the latest backup file for downstream tasks (SSMS Uploader / Airflow DAG).
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import os
import logging
import polars as pl
from dotenv import load_dotenv

# Load environment configuration
load_dotenv()

# Relative imports fallback for docker/package execution
try:
    from transformer.main import TransEngine
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from transformer.main import TransEngine

try:
    from mahgaurd import get_logger
    logger = get_logger("ADAS.Pipeline.Backup")
except ImportError:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)


class BackupEngine:
    """Manages Parquet backups for transformed pipeline data."""

    def __init__(self, backup_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the BackupEngine.

        Args:
            backup_dir (Optional[Union[str, Path]]): Target directory for parquet backups.
                If None, reads from BACKUP_DIR env variable, defaulting to 'parquet_backups'
                relative to this module file (Docker-friendly).
        """
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            env_dir = os.getenv("BACKUP_DIR")
            if env_dir:
                self.backup_dir = Path(env_dir)
            else:
                # Built dynamically relative to module - works seamlessly inside Docker & locally
                self.backup_dir = Path(__file__).resolve().parent / "parquet_backups"

        # Ensure backup folder exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"BackupEngine initialized. Storage directory: {self.backup_dir.resolve()}")

    def generate_filename(
        self,
        prefix: str = "sales_backup",
        timestamp_format: str = "%Y%m%d_%H%M%S",
        extension: str = "parquet"
    ) -> str:
        """
        Generate a timestamped file name.

        Args:
            prefix (str): Prefix for the file name. If empty, uses timestamp only.
            timestamp_format (str): Datetime format string.
            extension (str): File extension without dot.

        Returns:
            str: Generated filename (e.g., 'sales_backup_20261012_224000.parquet' or '20261012_224000.parquet').
        """
        timestamp = datetime.now().strftime(timestamp_format)
        if prefix:
            return f"{prefix}_{timestamp}.{extension}"
        return f"{timestamp}.{extension}"

    def backup_data(
        self,
        df: Optional[pl.DataFrame] = None,
        prefix: str = "sales_backup",
        timestamp_format: str = "%Y%m%d_%H%M%S"
    ) -> Path:
        """
        Save DataFrame as a timestamped Parquet backup file.

        Args:
            df (Optional[pl.DataFrame]): Polars DataFrame to back up.
                If None, triggers TransEngine to fetch & transform live data.
            prefix (str): Filename prefix (default: 'sales_backup'). Set to '' for time-only name.
            timestamp_format (str): Format of timestamp in filename.

        Returns:
            Path: Resolved path to the generated Parquet backup file.
        """
        if df is None:
            logger.info("No DataFrame provided. Running TransEngine transformation...")
            engine = TransEngine()
            df = engine.transform(drop_unrecoverable=True)

        filename = self.generate_filename(prefix=prefix, timestamp_format=timestamp_format)
        output_path = self.backup_dir / filename

        df.write_parquet(output_path)
        file_size_kb = output_path.stat().st_size / 1024
        logger.info(
            f"Backup successfully created at: {output_path.resolve()} "
            f"({len(df)} rows, {file_size_kb:.2f} KB)"
        )
        return output_path.resolve()

    def get_latest_backup(self) -> Optional[Path]:
        """
        Find the most recent Parquet backup file in the storage directory.

        Returns:
            Optional[Path]: Path to latest parquet file, or None if no files found.
        """
        parquet_files = list(self.backup_dir.glob("*.parquet"))
        if not parquet_files:
            logger.warning(f"No backup parquet files found in {self.backup_dir.resolve()}")
            return None

        # Sort by modification time descending
        latest_file = max(parquet_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Latest backup identified: {latest_file.resolve()}")
        return latest_file.resolve()

    def list_backups(self) -> List[Dict[str, Any]]:
        """
        List all available backups with metadata.

        Returns:
            List[Dict[str, Any]]: List of backup metadata dictionaries.
        """
        parquet_files = sorted(
            self.backup_dir.glob("*.parquet"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        backups = []
        for file in parquet_files:
            stat = file.stat()
            backups.append({
                "filename": file.name,
                "path": str(file.resolve()),
                "size_kb": round(stat.st_size / 1024, 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
        return backups


if __name__ == "__main__":
    backup_engine = BackupEngine()
    created_file = backup_engine.backup_data()
    print(f"Backup file created: {created_file}")
    print(f"Latest backup: {backup_engine.get_latest_backup()}")
    print("All backups:", backup_engine.list_backups())
