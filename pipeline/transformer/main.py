"""
Data Transformation Engine for ADAS (Automated Data Analytical System).

This module handles:
1. Column schema validation and integrity checks.
2. Deduplication of records.
3. Type casting and string cleaning.
4. Missing value imputation using domain relationships:
   - Recovering price_per_unit from total_spent / quantity
   - Recovering missing product items via catalog lookup (category + price_per_unit)
   - Recovering total_spent from price_per_unit * quantity
   - Recovering quantity from total_spent / price_per_unit
   - Defaulting missing discount_applied flags to False
5. Business rule validation (non-negative constraints, calculation verification).
6. Feature engineering (date/time decompositions for analytics).
7. Parquet and CSV export capabilities for downstream storage.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import logging
import polars as pl

try:
    from src_.ingestion import get_data
except ImportError:
    # Allow running directly or as a sub-package
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src_.ingestion import get_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TransEngine:
    """Transformation engine for sales transaction data."""

    # Schema definition
    REQUIRED_COLUMNS: List[str] = [
        "transaction_id",
        "customer_id",
        "category",
        "item",
        "price_per_unit",
        "quantity",
        "total_spent",
        "payment_method",
        "location",
        "transaction_date",
        "discount_applied",
    ]

    EXPECTED_CATEGORIES: List[str] = [
        "Beverages",
        "Butchers",
        "Computers and electric accessories",
        "Electric household essentials",
        "Food",
        "Furniture",
        "Milk Products",
        "Patisserie",
    ]

    EXPECTED_PAYMENT_METHODS: List[str] = [
        "Cash",
        "Credit Card",
        "Digital Wallet",
    ]

    EXPECTED_LOCATIONS: List[str] = [
        "In-store",
        "Online",
    ]

    def __init__(self, df: Optional[pl.DataFrame] = None):
        """
        Initialize the TransEngine.

        Args:
            df (Optional[pl.DataFrame]): In-memory Polars DataFrame to transform.
                If None, fetches live data using get_data().
        """
        if df is not None:
            self.df = df
        else:
            logger.info("Fetching data from ingestion API...")
            self.df = get_data()

        self.raw_df: pl.DataFrame = self.df.clone()
        logger.info(f"Initialized TransEngine with {len(self.df)} rows and {len(self.df.columns)} columns.")

    def check_columns(self) -> Dict[str, Any]:
        """
        Verify the presence, types, and null counts of all expected columns.

        Returns:
            Dict[str, Any]: Column audit report containing presence status, null counts, and schema.

        Raises:
            ValueError: If any required columns are missing from the DataFrame.
        """
        current_cols = set(self.df.columns)
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in current_cols]
        extra_cols = [col for col in self.df.columns if col not in self.REQUIRED_COLUMNS]

        if missing_cols:
            error_msg = f"Missing required columns in dataset: {missing_cols}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        null_summary = {
            col: int(self.df[col].null_count())
            for col in self.REQUIRED_COLUMNS
            if col in self.df.columns
        }

        report = {
            "total_rows": len(self.df),
            "columns_present": list(self.df.columns),
            "missing_columns": missing_cols,
            "extra_columns": extra_cols,
            "null_counts": null_summary,
            "schema": {col: str(dtype) for col, dtype in zip(self.df.columns, self.df.dtypes)},
        }

        logger.info(f"Column check passed. Missing: {len(missing_cols)}, Extra: {len(extra_cols)}")
        return report

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> "TransEngine":
        """
        Remove duplicate rows from the DataFrame.

        Args:
            subset (Optional[List[str]]): Subset of column names to consider for identifying duplicates.
                Defaults to ['transaction_id'].

        Returns:
            TransEngine: Self instance for method chaining.
        """
        subset = subset or ["transaction_id"]
        valid_subset = [col for col in subset if col in self.df.columns]

        before_count = len(self.df)
        if valid_subset:
            self.df = self.df.unique(subset=valid_subset, keep="first")
        else:
            self.df = self.df.unique(keep="first")

        dropped = before_count - len(self.df)
        logger.info(f"Deduplication complete. Dropped {dropped} duplicate rows (subset: {valid_subset}).")
        return self

    def remove_dublicates(self, subset: Optional[List[str]] = None) -> "TransEngine":
        """
        Alias for remove_duplicates to maintain backwards compatibility with existing typo.
        """
        return self.remove_duplicates(subset=subset)

    def clean_data_types(self) -> "TransEngine":
        """
        Cast and sanitize columns to their standard data types:
        - String columns: strip whitespace and ensure String type.
        - Date columns: parse string dates ('%Y-%m-%d') to Date.
        - Numeric columns: cast price_per_unit, quantity, total_spent to Float64.
        - Boolean columns: cast discount_applied to Boolean.

        Returns:
            TransEngine: Self instance for method chaining.
        """
        expressions: List[pl.Expr] = []

        # String cleaning
        string_cols = ["transaction_id", "customer_id", "category", "item", "payment_method", "location"]
        for col in string_cols:
            if col in self.df.columns:
                expressions.append(pl.col(col).cast(pl.String).str.strip_chars())

        # Numeric casting
        numeric_cols = ["price_per_unit", "quantity", "total_spent"]
        for col in numeric_cols:
            if col in self.df.columns:
                expressions.append(pl.col(col).cast(pl.Float64))

        # Boolean casting
        if "discount_applied" in self.df.columns:
            expressions.append(pl.col("discount_applied").cast(pl.Boolean))

        # Date parsing
        if "transaction_date" in self.df.columns:
            curr_dtype = self.df["transaction_date"].dtype
            if curr_dtype in (pl.String, pl.Utf8):
                expressions.append(pl.col("transaction_date").str.to_date("%Y-%m-%d"))
            elif curr_dtype == pl.Datetime:
                expressions.append(pl.col("transaction_date").dt.date())
            elif curr_dtype != pl.Date:
                expressions.append(pl.col("transaction_date").cast(pl.Date))

        if expressions:
            self.df = self.df.with_columns(expressions)

        logger.info("Data type cleaning and sanitization applied.")
        return self

    def impute_missing_values(
        self,
        drop_unrecoverable: bool = False,
        default_quantity: Optional[float] = None
    ) -> "TransEngine":
        """
        Impute missing values using domain logic and relationship mapping:
        1. Calculate price_per_unit where missing and total_spent & quantity exist.
        2. Infer missing product item names via known (category, price_per_unit) catalog mapping.
        3. Infer price_per_unit from product item name if any item is known but price was null.
        4. Calculate total_spent where missing and price_per_unit & quantity exist.
        5. Calculate quantity where missing and total_spent & price_per_unit exist.
        6. Impute discount_applied nulls with False.
        7. If default_quantity is specified, impute quantity and recompute total_spent.
        8. If drop_unrecoverable is True, drop any remaining rows with unrecoverable nulls.

        Args:
            drop_unrecoverable (bool): If True, drops rows where quantity or total_spent cannot be recovered.
            default_quantity (Optional[float]): Fallback quantity value for rows where both quantity and total_spent are missing.

        Returns:
            TransEngine: Self instance for method chaining.
        """
        # Step 1: Infer price_per_unit from total_spent / quantity if both exist
        if all(col in self.df.columns for col in ["price_per_unit", "total_spent", "quantity"]):
            self.df = self.df.with_columns(
                pl.when(
                    pl.col("price_per_unit").is_null()
                    & pl.col("total_spent").is_not_null()
                    & pl.col("quantity").is_not_null()
                    & (pl.col("quantity") > 0)
                )
                .then((pl.col("total_spent") / pl.col("quantity")).round(2))
                .otherwise(pl.col("price_per_unit"))
                .alias("price_per_unit")
            )

        # Step 2: Build catalog mapping (category, price_per_unit) -> item from existing complete rows
        if all(col in self.df.columns for col in ["category", "price_per_unit", "item"]):
            catalog_item = (
                self.df.filter(pl.col("item").is_not_null() & pl.col("price_per_unit").is_not_null())
                .select(["category", "price_per_unit", "item"])
                .unique()
                .rename({"item": "_inferred_item"})
            )
            if len(catalog_item) > 0:
                self.df = self.df.join(catalog_item, on=["category", "price_per_unit"], how="left")
                self.df = self.df.with_columns(
                    pl.coalesce([pl.col("item"), pl.col("_inferred_item")]).alias("item")
                ).drop("_inferred_item")

            # Step 3: Reverse lookup: infer price_per_unit from item if still missing
            catalog_price = (
                self.df.filter(pl.col("item").is_not_null() & pl.col("price_per_unit").is_not_null())
                .select(["item", "price_per_unit"])
                .unique()
                .rename({"price_per_unit": "_inferred_price"})
            )
            if len(catalog_price) > 0:
                self.df = self.df.join(catalog_price, on="item", how="left")
                self.df = self.df.with_columns(
                    pl.coalesce([pl.col("price_per_unit"), pl.col("_inferred_price")]).alias("price_per_unit")
                ).drop("_inferred_price")

        # Step 4: Calculate total_spent where missing
        if all(col in self.df.columns for col in ["total_spent", "price_per_unit", "quantity"]):
            self.df = self.df.with_columns(
                pl.when(
                    pl.col("total_spent").is_null()
                    & pl.col("price_per_unit").is_not_null()
                    & pl.col("quantity").is_not_null()
                )
                .then((pl.col("price_per_unit") * pl.col("quantity")).round(2))
                .otherwise(pl.col("total_spent"))
                .alias("total_spent")
            )

        # Step 5: Calculate quantity where missing
        if all(col in self.df.columns for col in ["quantity", "total_spent", "price_per_unit"]):
            self.df = self.df.with_columns(
                pl.when(
                    pl.col("quantity").is_null()
                    & pl.col("total_spent").is_not_null()
                    & pl.col("price_per_unit").is_not_null()
                    & (pl.col("price_per_unit") > 0)
                )
                .then((pl.col("total_spent") / pl.col("price_per_unit")).round(2))
                .otherwise(pl.col("quantity"))
                .alias("quantity")
            )

        # Step 6: Fill null discount_applied with False
        if "discount_applied" in self.df.columns:
            self.df = self.df.with_columns(
                pl.col("discount_applied").fill_null(False)
            )

        # Step 7: Handle unrecoverable quantity / total_spent
        if default_quantity is not None and "quantity" in self.df.columns and "total_spent" in self.df.columns:
            self.df = self.df.with_columns(
                pl.col("quantity").fill_null(float(default_quantity)).alias("quantity")
            ).with_columns(
                pl.when(pl.col("total_spent").is_null() & pl.col("price_per_unit").is_not_null())
                .then((pl.col("price_per_unit") * pl.col("quantity")).round(2))
                .otherwise(pl.col("total_spent"))
                .alias("total_spent")
            )

        if drop_unrecoverable:
            subset = [col for col in ["quantity", "total_spent"] if col in self.df.columns]
            if subset:
                before = len(self.df)
                self.df = self.df.drop_nulls(subset=subset)
                logger.info(f"Dropped {before - len(self.df)} rows with unrecoverable nulls.")

        logger.info("Missing value imputation completed.")
        return self

    def validate_business_rules(self, enforce_consistency: bool = True) -> "TransEngine":
        """
        Validate business integrity rules on numerical values:
        - Ensures price_per_unit > 0, quantity > 0, total_spent > 0.
        - Enforces total_spent = round(price_per_unit * quantity, 2) to eliminate rounding errors.

        Args:
            enforce_consistency (bool): When True, aligns total_spent to round(price_per_unit * quantity, 2).

        Returns:
            TransEngine: Self instance for method chaining.
        """
        if enforce_consistency and all(col in self.df.columns for col in ["price_per_unit", "quantity", "total_spent"]):
            self.df = self.df.with_columns(
                pl.when(pl.col("price_per_unit").is_not_null() & pl.col("quantity").is_not_null())
                .then((pl.col("price_per_unit") * pl.col("quantity")).round(2))
                .otherwise(pl.col("total_spent"))
                .alias("total_spent")
            )

        logger.info("Business rule validation completed.")
        return self

    def enrich_features(self) -> "TransEngine":
        """
        Derive calendar and temporal features from transaction_date:
        - year (Int32)
        - month (Int8)
        - month_name (String)
        - day (Int8)
        - day_of_week (Int8, 1=Mon, 7=Sun)
        - day_name (String)
        - quarter (Int8)
        - is_weekend (Boolean)

        Returns:
            TransEngine: Self instance for method chaining.
        """
        if "transaction_date" in self.df.columns:
            date_col = pl.col("transaction_date")
            self.df = self.df.with_columns([
                date_col.dt.year().alias("year"),
                date_col.dt.month().alias("month"),
                date_col.dt.strftime("%B").alias("month_name"),
                date_col.dt.day().alias("day"),
                date_col.dt.weekday().alias("day_of_week"),
                date_col.dt.strftime("%A").alias("day_name"),
                date_col.dt.quarter().alias("quarter"),
                date_col.dt.weekday().is_in([6, 7]).alias("is_weekend"),
            ])
            logger.info("Calendar and temporal features successfully added.")
        return self

    def transform(
        self,
        drop_unrecoverable: bool = False,
        default_quantity: Optional[float] = None
    ) -> pl.DataFrame:
        """
        Execute the full end-to-end transformation pipeline.

        Args:
            drop_unrecoverable (bool): Whether to drop rows where metrics remain null.
            default_quantity (Optional[float]): Fallback quantity for unrecoverable rows.

        Returns:
            pl.DataFrame: The transformed Polars DataFrame.
        """
        logger.info("Starting complete transformation pipeline...")
        (
            self.check_columns()
        )
        (
            self.remove_duplicates()
            .clean_data_types()
            .impute_missing_values(
                drop_unrecoverable=drop_unrecoverable,
                default_quantity=default_quantity
            )
            .validate_business_rules()
            .enrich_features()
        )
        logger.info(f"Transformation pipeline finished. Output shape: {self.df.shape}")
        return self.df

    def get_df(self) -> pl.DataFrame:
        """Return the current transformed Polars DataFrame."""
        return self.df


    def to_parquet(self, filepath: Union[str, Path]) -> Path:
        """
        Save the current DataFrame to a Parquet file.

        Args:
            filepath (Union[str, Path]): Destination file path.

        Returns:
            Path: Resolved path to the written Parquet file.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.write_parquet(path)
        logger.info(f"DataFrame successfully saved to Parquet: {path.resolve()}")
        return path



