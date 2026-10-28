"""
Analytical Engine for ADAS (Automated Data Analytical System).

This engine serves as the analytical foundation powering the executive
Streamlit dashboards and decision-support systems.

Features:
1. Resilient multi-tier data loading (Database -> Parquet Backups -> API -> In-Memory).
2. Executive KPI computation with period-over-period growth comparisons.
3. Pareto (ABC) Product & Category Contribution Analysis (80/20 rule).
4. Customer RFM (Recency, Frequency, Monetary) Segmentation.
5. Channel (Online vs In-store) and Payment Method performance breakdown.
6. Time-series trend decomposition and seasonality patterns.
7. What-If Scenario Decision Simulator (Price elasticity & discount modeling).
8. Inventory velocity & prescriptive re-order intelligence.
9. Rule-based automated prescriptive business recommendations.
10. Persistent Decision Logger for tracking user decisions and answers.
"""

from __future__ import annotations
import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, asdict

import polars as pl
import pandas as pd
import numpy as np

# Configure module logging via Mahgaurd
try:
    from mahgaurd import get_logger, timed
    logger = get_logger("ADAS.AnalyticalEngine")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("ADAS.AnalyticalEngine")
    def timed(name=None):
        return lambda f: f

ROOT_DIR = Path(__file__).resolve().parents[1]
DECISIONS_FILE = ROOT_DIR / "engines" / "decisions_log.json"


@dataclass
class DecisionRecord:
    """Structure for tracking strategic decisions made in the dashboard."""
    decision_id: str
    title: str
    category: str
    business_question: str
    decision_answer: str
    rationale: str
    projected_impact: str
    assigned_owner: str
    status: str  # "Proposed", "Approved", "In Progress", "Implemented"
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DecisionManager:
    """Persistent storage and management of executive decisions and answers."""

    def __init__(self, filepath: Optional[Path] = None):
        self.filepath = filepath or DECISIONS_FILE
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create initial decisions file if not present."""
        if not self.filepath.exists():
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            initial_decisions = [
                {
                    "decision_id": "DEC-001",
                    "title": "Milk Products Margin Protection",
                    "category": "Pricing & Discounts",
                    "business_question": "Should we reduce promotional discounts on Milk Products given high steady demand?",
                    "decision_answer": "Cap promotional discounts on Milk Products to maximum 5% across all stores.",
                    "rationale": "Milk Products show low price elasticity and high reorder velocity. Current 10%+ discounts cause unnecessary margin leakage.",
                    "projected_impact": "+$18,500 monthly margin improvement with <1.5% volume drop.",
                    "assigned_owner": "Pricing Operations",
                    "status": "Approved",
                    "created_at": "2026-10-15T10:00:00",
                    "updated_at": "2026-10-15T14:30:00",
                },
                {
                    "decision_id": "DEC-002",
                    "title": "Digital Wallet Incentive for In-Store Orders",
                    "category": "Channel & Payments",
                    "business_question": "How can we reduce cash handling costs in physical retail outlets?",
                    "decision_answer": "Offer a 2% instant cashback on Digital Wallet transactions over $50 in physical stores.",
                    "rationale": "Digital wallet users demonstrate a 24% higher Average Order Value compared to cash transactions.",
                    "projected_impact": "+15% increase in digital payments, lowering register settlement reconciliation overhead.",
                    "assigned_owner": "Retail Strategy Team",
                    "status": "In Progress",
                    "created_at": "2026-10-16T09:15:00",
                    "updated_at": "2026-10-16T09:15:00",
                }
            ]
            self.save_all(initial_decisions)

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded decisions."""
        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading decisions file: {e}")
            return []

    def save_all(self, decisions: List[Dict[str, Any]]) -> None:
        """Save entire decision list to JSON file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(decisions, f, indent=2, ensure_ascii=False)

    def add_decision(
        self,
        title: str,
        category: str,
        business_question: str,
        decision_answer: str,
        rationale: str,
        projected_impact: str,
        assigned_owner: str,
        status: str = "Proposed"
    ) -> DecisionRecord:
        """Add a new decision record."""
        decisions = self.get_all()
        next_id_num = len(decisions) + 1
        decision_id = f"DEC-{next_id_num:03d}"
        now = datetime.now().isoformat(timespec="seconds")

        record = DecisionRecord(
            decision_id=decision_id,
            title=title.strip(),
            category=category.strip(),
            business_question=business_question.strip(),
            decision_answer=decision_answer.strip(),
            rationale=rationale.strip(),
            projected_impact=projected_impact.strip(),
            assigned_owner=assigned_owner.strip(),
            status=status.strip(),
            created_at=now,
            updated_at=now
        )

        decisions.append(record.to_dict())
        self.save_all(decisions)
        logger.info(f"Recorded decision {decision_id}: {title}")
        return record

    def update_decision_status(self, decision_id: str, new_status: str) -> bool:
        """Update the status of an existing decision."""
        decisions = self.get_all()
        updated = False
        for d in decisions:
            if d.get("decision_id") == decision_id:
                d["status"] = new_status
                d["updated_at"] = datetime.now().isoformat(timespec="seconds")
                updated = True
                break
        if updated:
            self.save_all(decisions)
        return updated

    def delete_decision(self, decision_id: str) -> bool:
        """Delete a decision by ID."""
        decisions = self.get_all()
        filtered = [d for d in decisions if d.get("decision_id") != decision_id]
        if len(filtered) < len(decisions):
            self.save_all(filtered)
            return True
        return False

    def to_pandas(self) -> pd.DataFrame:
        """Convert decision records to Pandas DataFrame for dashboard tables."""
        decisions = self.get_all()
        if not decisions:
            return pd.DataFrame(columns=[
                "decision_id", "title", "category", "business_question",
                "decision_answer", "rationale", "projected_impact", "assigned_owner",
                "status", "created_at"
            ])
        return pd.DataFrame(decisions)


class AnalyticalEngine:
    """
    High-performance analytical computation engine.
    
    Supports data retrieval, aggregation, cohort segmentation,
    scenario forecasting, and decision support for the Streamlit dashboard.
    """

    def __init__(self, data_source: str = "auto", df: Optional[Union[pl.DataFrame, pd.DataFrame]] = None):
        """
        Initialize the AnalyticalEngine.
        
        Args:
            data_source (str): Source strategy: 'auto', 'database', 'parquet', 'api', or 'custom'.
            df (Optional[Union[pl.DataFrame, pd.DataFrame]]): Pre-loaded DataFrame.
        """
        self.data_source = data_source
        self.decision_manager = DecisionManager()
        self._raw_df: pl.DataFrame = self._load_data(df)
        self.df: pl.DataFrame = self._prepare_data(self._raw_df)
        logger.info(f"AnalyticalEngine initialized successfully with {len(self.df)} records.")

    def _load_data(self, df_input: Optional[Union[pl.DataFrame, pd.DataFrame]]) -> pl.DataFrame:
        """Load data with fallback strategy: custom -> db -> parquet -> api."""
        if df_input is not None:
            if isinstance(df_input, pd.DataFrame):
                return pl.from_pandas(df_input)
            return df_input

        # 1. Attempt Database connection via services ORM if explicit or auto
        if self.data_source in ("auto", "database"):
            urls_to_try = []
            env_url = os.getenv("DATABASE_URL")
            if env_url:
                urls_to_try.append(env_url)
                if "host.docker.internal" in env_url:
                    urls_to_try.append(env_url.replace("host.docker.internal", "localhost"))

            for url in urls_to_try:
                try:
                    from sqlalchemy import create_engine, select
                    from sqlalchemy.orm import Session
                    from services.config_.object_relations import MainTable
                    temp_engine = create_engine(url, connect_args={"connect_timeout": 2})
                    with Session(temp_engine) as session:
                        stmt = select(MainTable).limit(5)
                        probe = session.execute(stmt).scalars().first()
                        if probe is not None:
                            all_stmt = select(MainTable)
                            records = session.execute(all_stmt).scalars().all()
                            data = [
                                {
                                    "transaction_id": r.transaction_id,
                                    "customer_id": r.customer_id,
                                    "category": r.category,
                                    "item": r.item,
                                    "price_per_unit": r.price_per_unit,
                                    "quantity": r.quantity,
                                    "total_spent": r.total_spent,
                                    "payment_method": r.payment_method,
                                    "location": r.location,
                                    "transaction_date": r.transaction_date,
                                    "discount_applied": r.discount_applied,
                                }
                                for r in records
                            ]
                            logger.info(f"Loaded {len(data)} records from database warehouse at {url.split('@')[-1]}.")
                            return pl.DataFrame(data)
                except Exception as e:
                    logger.debug(f"DB probe failed on {url}: {e}")
            logger.info("Database connection unavailable or skipped, falling back to Parquet backups.")

        # 2. Attempt loading from Parquet backups in pipeline/backups/parquet_backups
        backup_dir = ROOT_DIR / "pipeline" / "backups" / "parquet_backups"
        if backup_dir.exists():
            parquet_files = sorted(backup_dir.glob("*.parquet"), key=os.path.getmtime, reverse=True)
            if parquet_files:
                target_file = parquet_files[0]
                logger.info(f"Loading latest backup parquet: {target_file.name}")
                return pl.read_parquet(target_file)

        # 3. Fallback to API if reachable
        try:
            from pipeline.src_.ingestion import get_data
            df_api = get_data()
            if len(df_api) > 0:
                logger.info(f"Loaded {len(df_api)} records from ingestion API.")
                return df_api
        except Exception as e:
            logger.warning(f"API data fetch not available: {e}")

        # 4. If all fail, create an emergency synthetic sample dataset so the engine never crashes
        logger.warning("No storage source reached. Generating bootstrap production sample dataset.")
        return self._generate_bootstrap_data()

    def _prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure consistent types and date-derived attributes."""
        # Ensure proper data types
        cast_map = {}
        if "price_per_unit" in df.columns:
            cast_map["price_per_unit"] = pl.Float64
        if "quantity" in df.columns:
            cast_map["quantity"] = pl.Float64
        if "total_spent" in df.columns:
            cast_map["total_spent"] = pl.Float64
        if "discount_applied" in df.columns:
            cast_map["discount_applied"] = pl.Boolean

        if cast_map:
            df = df.with_columns([pl.col(c).cast(t) for c, t in cast_map.items()])

        # Parse transaction_date if string
        if "transaction_date" in df.columns:
            curr_type = df["transaction_date"].dtype
            if curr_type in (pl.String, pl.Utf8):
                df = df.with_columns(pl.col("transaction_date").str.to_date("%Y-%m-%d"))
            elif curr_type == pl.Datetime:
                df = df.with_columns(pl.col("transaction_date").dt.date())

        # Enrich with temporal fields if missing
        cols = set(df.columns)
        if "transaction_date" in cols:
            exprs = []
            if "year" not in cols:
                exprs.append(pl.col("transaction_date").dt.year().alias("year"))
            if "month" not in cols:
                exprs.append(pl.col("transaction_date").dt.month().alias("month"))
            if "month_name" not in cols:
                exprs.append(pl.col("transaction_date").dt.strftime("%B").alias("month_name"))
            if "day_name" not in cols:
                exprs.append(pl.col("transaction_date").dt.strftime("%A").alias("day_name"))
            if "is_weekend" not in cols:
                exprs.append(pl.col("transaction_date").dt.weekday().is_in([6, 7]).alias("is_weekend"))
            if exprs:
                df = df.with_columns(exprs)

        # Ensure total_spent is not null or 0 if price and quantity exist
        if "total_spent" in df.columns and "price_per_unit" in df.columns and "quantity" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("total_spent").is_null() | (pl.col("total_spent") <= 0))
                .then(pl.col("price_per_unit") * pl.col("quantity"))
                .otherwise(pl.col("total_spent"))
                .alias("total_spent")
            )

        return df

    def _generate_bootstrap_data(self) -> pl.DataFrame:
        """Provide fallback operational dataset in case external systems are unreachable."""
        import random
        random.seed(42)
        n = 500
        categories = ["Beverages", "Food", "Computers and electric accessories", "Furniture", "Patisserie"]
        locations = ["In-store", "Online"]
        payments = ["Credit Card", "Digital Wallet", "Cash"]

        records = []
        for i in range(n):
            cat = random.choice(categories)
            item = f"{cat} Item {random.randint(1, 10)}"
            price = round(random.uniform(5.0, 150.0), 2)
            qty = random.randint(1, 5)
            spent = round(price * qty, 2)
            d = date(2026, random.randint(1, 10), random.randint(1, 28))
            records.append({
                "transaction_id": f"TX-{i:05d}",
                "customer_id": f"CUST-{random.randint(100, 250)}",
                "category": cat,
                "item": item,
                "price_per_unit": price,
                "quantity": float(qty),
                "total_spent": spent,
                "payment_method": random.choice(payments),
                "location": random.choice(locations),
                "transaction_date": d,
                "discount_applied": random.choice([True, False, False]),
            })
        return pl.DataFrame(records)

    # =========================================================================
    # CORE EXECUTIVE METRICS & PERIOD COMPARISONS
    # =========================================================================

    @timed("AnalyticalEngine.get_kpis")
    def get_kpis(self, df_filtered: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """Compute top-level business KPIs."""
        df = df_filtered if df_filtered is not None else self.df
        if len(df) == 0:
            return {
                "total_revenue": 0.0,
                "total_units": 0,
                "total_transactions": 0,
                "aov": 0.0,
                "unique_customers": 0,
                "discount_rate_pct": 0.0,
                "avg_price_per_unit": 0.0,
            }

        total_rev = float(df["total_spent"].sum() or 0.0)
        total_units = float(df["quantity"].sum() or 0.0)
        total_tx = len(df)
        aov = total_rev / total_tx if total_tx > 0 else 0.0
        uniq_cust = df["customer_id"].n_unique()

        discount_tx = len(df.filter(pl.col("discount_applied") == True))
        discount_rate = (discount_tx / total_tx * 100.0) if total_tx > 0 else 0.0
        avg_price = float(df["price_per_unit"].mean() or 0.0)

        return {
            "total_revenue": round(total_rev, 2),
            "total_units": int(total_units),
            "total_transactions": total_tx,
            "aov": round(aov, 2),
            "unique_customers": uniq_cust,
            "discount_rate_pct": round(discount_rate, 1),
            "avg_price_per_unit": round(avg_price, 2),
        }

    @timed("AnalyticalEngine.get_period_comparison")
    def get_period_comparison(self, df_filtered: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """Compute Month-over-Month growth metrics."""
        df = df_filtered if df_filtered is not None else self.df
        if "transaction_date" not in df.columns or len(df) == 0:
            return {"revenue_growth_pct": 0.0, "orders_growth_pct": 0.0, "aov_growth_pct": 0.0}

        monthly = (
            df.with_columns(pl.col("transaction_date").dt.strftime("%Y-%m").alias("year_month"))
            .group_by("year_month")
            .agg([
                pl.col("total_spent").sum().alias("revenue"),
                pl.len().alias("orders"),
            ])
            .sort("year_month")
        )

        if len(monthly) < 2:
            return {"revenue_growth_pct": 0.0, "orders_growth_pct": 0.0, "aov_growth_pct": 0.0}

        last_month = monthly.row(-1, named=True)
        prev_month = monthly.row(-2, named=True)

        rev_growth = ((last_month["revenue"] - prev_month["revenue"]) / prev_month["revenue"] * 100.0) if prev_month["revenue"] > 0 else 0.0
        orders_growth = ((last_month["orders"] - prev_month["orders"]) / prev_month["orders"] * 100.0) if prev_month["orders"] > 0 else 0.0

        prev_aov = prev_month["revenue"] / prev_month["orders"] if prev_month["orders"] > 0 else 0.0
        last_aov = last_month["revenue"] / last_month["orders"] if last_month["orders"] > 0 else 0.0
        aov_growth = ((last_aov - prev_aov) / prev_aov * 100.0) if prev_aov > 0 else 0.0

        return {
            "latest_period": last_month["year_month"],
            "prev_period": prev_month["year_month"],
            "revenue_growth_pct": round(rev_growth, 2),
            "orders_growth_pct": round(orders_growth, 2),
            "aov_growth_pct": round(aov_growth, 2),
            "latest_revenue": round(last_month["revenue"], 2),
            "prev_revenue": round(prev_month["revenue"], 2),
        }

    # =========================================================================
    # CATEGORY & PARETO (ABC) PRODUCT ANALYSIS
    # =========================================================================

    @timed("AnalyticalEngine.get_category_performance")
    def get_category_performance(self, df_filtered: Optional[pl.DataFrame] = None) -> pd.DataFrame:
        """Aggregate performance metrics across product categories."""
        df = df_filtered if df_filtered is not None else self.df
        total_rev = df["total_spent"].sum() or 1.0

        agg = (
            df.group_by("category")
            .agg([
                pl.col("total_spent").sum().alias("total_revenue"),
                pl.col("quantity").sum().alias("units_sold"),
                pl.len().alias("transactions"),
                pl.col("price_per_unit").mean().alias("avg_price"),
                (pl.col("discount_applied") == True).mean().alias("discount_share"),
            ])
            .with_columns([
                (pl.col("total_revenue") / pl.col("transactions")).round(2).alias("aov"),
                (pl.col("total_revenue") / total_rev * 100.0).round(2).alias("revenue_share_pct"),
                (pl.col("discount_share") * 100.0).round(1).alias("discount_applied_pct"),
            ])
            .sort("total_revenue", descending=True)
        )
        return agg.to_pandas()

    @timed("AnalyticalEngine.get_pareto_analysis")
    def get_pareto_analysis(self, df_filtered: Optional[pl.DataFrame] = None) -> pd.DataFrame:
        """
        Conduct Pareto (ABC) 80/20 analysis on product items:
        - Class A: Top 80% cumulative revenue (Critical core revenue drivers)
        - Class B: Next 15% cumulative revenue (80% to 95%)
        - Class C: Bottom 5% cumulative revenue (95% to 100% - Long tail)
        """
        df = df_filtered if df_filtered is not None else self.df
        total_rev = df["total_spent"].sum() or 1.0

        item_rev = (
            df.group_by(["category", "item"])
            .agg([
                pl.col("total_spent").sum().alias("revenue"),
                pl.col("quantity").sum().alias("units"),
                pl.len().alias("orders"),
                pl.col("price_per_unit").mean().alias("avg_price"),
            ])
            .sort("revenue", descending=True)
        )

        item_rev = item_rev.with_columns([
            pl.col("revenue").cum_sum().alias("cum_revenue"),
            (pl.col("revenue").cum_sum() / total_rev * 100.0).alias("cum_revenue_pct"),
            (pl.col("revenue") / total_rev * 100.0).alias("revenue_pct"),
        ])

        item_rev = item_rev.with_columns(
            pl.when(pl.col("cum_revenue_pct") <= 80.0)
            .then(pl.lit("A (Top 80%)"))
            .when(pl.col("cum_revenue_pct") <= 95.0)
            .then(pl.lit("B (Next 15%)"))
            .otherwise(pl.lit("C (Bottom 5%)"))
            .alias("abc_class")
        )

        return item_rev.to_pandas()

    # =========================================================================
    # CUSTOMER RFM & COHORT SEGMENTATION
    # =========================================================================

    def get_customer_rfm_segments(self, df_filtered: Optional[pl.DataFrame] = None) -> pd.DataFrame:
        """
        Segment customers into actionable cohorts using Recency, Frequency, Monetary (RFM).
        """
        df = df_filtered if df_filtered is not None else self.df
        if len(df) == 0:
            return pd.DataFrame()

        max_date = df["transaction_date"].max()

        cust_agg = (
            df.group_by("customer_id")
            .agg([
                pl.col("transaction_date").max().alias("last_order_date"),
                pl.len().alias("frequency"),
                pl.col("total_spent").sum().alias("monetary"),
                pl.col("total_spent").mean().alias("avg_order_value"),
            ])
            .with_columns(
                (pl.lit(max_date) - pl.col("last_order_date")).dt.total_days().alias("recency_days")
            )
        )

        # Quantile or rule-based scoring
        pdf = cust_agg.to_pandas()
        
        # Rank scores into 1-4 scale
        pdf["r_score"] = pd.qcut(pdf["recency_days"].rank(method="first"), q=4, labels=[4, 3, 2, 1]).astype(int)
        pdf["f_score"] = pd.qcut(pdf["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)
        pdf["m_score"] = pd.qcut(pdf["monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)

        # Segment assignment
        def assign_segment(row):
            r, f, m = row["r_score"], row["f_score"], row["m_score"]
            if r >= 3 and f >= 3 and m >= 3:
                return "Champions"
            elif f >= 3 and m >= 2:
                return "Loyal Customers"
            elif r >= 3 and f <= 2:
                return "Recent / Potential"
            elif r <= 2 and f >= 3:
                return "At Risk (High Past Value)"
            elif r == 1:
                return "Lost / Inactive"
            else:
                return "Standard Customers"

        pdf["customer_segment"] = pdf.apply(assign_segment, axis=1)
        return pdf

    # =========================================================================
    # CHANNEL & PAYMENT ANALYSIS
    # =========================================================================

    def get_channel_and_payment_insights(self, df_filtered: Optional[pl.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """Examine channel (Online vs In-store) and payment instrument dynamics."""
        df = df_filtered if df_filtered is not None else self.df

        # Channel
        channel_df = (
            df.group_by("location")
            .agg([
                pl.col("total_spent").sum().alias("total_revenue"),
                pl.len().alias("transactions"),
                pl.col("quantity").sum().alias("units_sold"),
                (pl.col("discount_applied") == True).mean().alias("discount_rate"),
            ])
            .with_columns([
                (pl.col("total_revenue") / pl.col("transactions")).round(2).alias("aov"),
                (pl.col("discount_rate") * 100.0).round(1).alias("discount_rate_pct"),
            ])
            .sort("total_revenue", descending=True)
            .to_pandas()
        )

        # Payment Method
        payment_df = (
            df.group_by("payment_method")
            .agg([
                pl.col("total_spent").sum().alias("total_revenue"),
                pl.len().alias("transactions"),
                (pl.col("total_spent").sum() / (df["total_spent"].sum() or 1.0) * 100.0).round(1).alias("revenue_share_pct"),
            ])
            .with_columns(
                (pl.col("total_revenue") / pl.col("transactions")).round(2).alias("aov")
            )
            .sort("total_revenue", descending=True)
            .to_pandas()
        )

        # Cross breakdown
        cross_df = (
            df.group_by(["location", "payment_method"])
            .agg([
                pl.col("total_spent").sum().alias("revenue"),
                pl.len().alias("orders"),
            ])
            .sort(["location", "revenue"], descending=[False, True])
            .to_pandas()
        )

        return {
            "channel": channel_df,
            "payment": payment_df,
            "cross": cross_df,
        }

    # =========================================================================
    # TIME SERIES & SEASONALITY
    # =========================================================================

    def get_timeseries_trends(self, df_filtered: Optional[pl.DataFrame] = None, freq: str = "D") -> pd.DataFrame:
        """Daily or weekly sales trends with rolling averages."""
        df = df_filtered if df_filtered is not None else self.df
        if len(df) == 0:
            return pd.DataFrame()

        daily = (
            df.group_by("transaction_date")
            .agg([
                pl.col("total_spent").sum().alias("revenue"),
                pl.len().alias("transactions"),
                pl.col("quantity").sum().alias("units"),
            ])
            .sort("transaction_date")
            .to_pandas()
        )

        daily["transaction_date"] = pd.to_datetime(daily["transaction_date"])
        daily["revenue_7d_ma"] = daily["revenue"].rolling(window=7, min_periods=1).mean().round(2)
        daily["revenue_30d_ma"] = daily["revenue"].rolling(window=30, min_periods=1).mean().round(2)
        return daily

    def get_day_of_week_distribution(self, df_filtered: Optional[pl.DataFrame] = None) -> pd.DataFrame:
        """Day of week sales velocity (Monday to Sunday)."""
        df = df_filtered if df_filtered is not None else self.df
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        dow = (
            df.group_by("day_name")
            .agg([
                pl.col("total_spent").sum().alias("revenue"),
                pl.len().alias("transactions"),
                pl.col("quantity").sum().alias("units"),
            ])
            .with_columns((pl.col("revenue") / pl.col("transactions")).round(2).alias("aov"))
            .to_pandas()
        )
        dow["day_name"] = pd.Categorical(dow["day_name"], categories=order, ordered=True)
        return dow.sort_values("day_name")

    # =========================================================================
    # WHAT-IF DECISION SCENARIO SIMULATOR
    # =========================================================================

    def simulate_what_if_scenario(
        self,
        price_change_pct: float = 0.0,
        discount_reduction_pct: float = 0.0,
        elasticity_factor: float = -1.2,
        target_categories: Optional[List[str]] = None,
        base_df: Optional[pl.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Simulate impact of pricing & discount policy decisions before execution.
        
        Args:
            price_change_pct: % price increase (positive) or discount (negative). E.g. +5.0%
            discount_reduction_pct: % reduction in promotional discount frequency. E.g. 20.0%
            elasticity_factor: Price elasticity of demand (e.g. -1.2 means 10% price rise causes 12% demand drop).
            target_categories: List of categories to apply simulation to (None = all).
        """
        df = base_df if base_df is not None else self.df
        if target_categories:
            sim_df = df.filter(pl.col("category").is_in(target_categories))
        else:
            sim_df = df

        base_revenue = float(sim_df["total_spent"].sum() or 0.0)
        base_units = float(sim_df["quantity"].sum() or 0.0)
        base_tx = len(sim_df)

        # Baseline estimated gross margin (assume standard retail gross margin benchmark ~35%)
        base_cogs_ratio = 0.65
        base_gross_margin = base_revenue * (1 - base_cogs_ratio)

        # Demand change from price
        demand_change_pct = price_change_pct * elasticity_factor
        projected_units = max(0.0, base_units * (1.0 + demand_change_pct / 100.0))

        # New effective price
        avg_price = (base_revenue / base_units) if base_units > 0 else 1.0
        new_price = avg_price * (1.0 + price_change_pct / 100.0)

        # Margin saved from reduced discounting (discounted orders spend ~15% less margin)
        discount_margin_boost = base_revenue * 0.15 * (discount_reduction_pct / 100.0)

        projected_revenue = projected_units * new_price + (discount_margin_boost * 0.5)
        revenue_delta = projected_revenue - base_revenue
        revenue_delta_pct = (revenue_delta / base_revenue * 100.0) if base_revenue > 0 else 0.0

        # Projected Gross Margin
        projected_gross_margin = (projected_revenue * (1 - base_cogs_ratio)) + discount_margin_boost
        margin_delta = projected_gross_margin - base_gross_margin
        margin_delta_pct = (margin_delta / base_gross_margin * 100.0) if base_gross_margin > 0 else 0.0

        return {
            "baseline_revenue": round(base_revenue, 2),
            "projected_revenue": round(projected_revenue, 2),
            "revenue_delta": round(revenue_delta, 2),
            "revenue_delta_pct": round(revenue_delta_pct, 2),
            "baseline_units": int(base_units),
            "projected_units": int(projected_units),
            "units_delta_pct": round(demand_change_pct, 2),
            "baseline_margin": round(base_gross_margin, 2),
            "projected_margin": round(projected_gross_margin, 2),
            "margin_delta": round(margin_delta, 2),
            "margin_delta_pct": round(margin_delta_pct, 2),
        }

    # =========================================================================
    # INVENTORY VELOCITY & PRESCRIPTIVE REORDER INTELLIGENCE
    # =========================================================================

    def get_inventory_and_reorder_recommendations(
        self,
        lead_time_days: int = 7,
        target_buffer_days: int = 30,
        df_filtered: Optional[pl.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute daily sales run-rate velocity and stockout risk recommendations per item.
        """
        df = df_filtered if df_filtered is not None else self.df
        total_days = max(1, (df["transaction_date"].max() - df["transaction_date"].min()).days)

        velocity = (
            df.group_by(["category", "item"])
            .agg([
                pl.col("quantity").sum().alias("total_units_sold"),
                pl.col("total_spent").sum().alias("total_revenue"),
                pl.col("price_per_unit").mean().alias("avg_price"),
                pl.len().alias("order_count"),
            ])
            .with_columns([
                (pl.col("total_units_sold") / float(total_days)).round(2).alias("daily_sales_velocity"),
            ])
            .to_pandas()
        )

        # Simulate current stock based on item turnover distribution for decision-making
        np.random.seed(42)
        velocity["assumed_current_stock"] = (
            velocity["daily_sales_velocity"] * np.random.uniform(5.0, 35.0, len(velocity))
        ).round(0).astype(int)

        # Estimated days of stock remaining
        velocity["days_of_inventory"] = (
            velocity["assumed_current_stock"] / np.maximum(0.1, velocity["daily_sales_velocity"])
        ).round(1)

        # Risk Classification
        def classify_stock_risk(days):
            if days <= 7:
                return "CRITICAL (Stockout <7d)"
            elif days <= 14:
                return "WARNING (Reorder Needed <14d)"
            elif days <= 30:
                return "OPTIMAL (15-30d)"
            else:
                return "OVERSTOCKED (>30d)"

        velocity["stock_status"] = velocity["days_of_inventory"].apply(classify_stock_risk)

        # Suggested Reorder Quantity
        velocity["recommended_reorder_qty"] = np.maximum(
            0,
            (velocity["daily_sales_velocity"] * (target_buffer_days + lead_time_days) - velocity["assumed_current_stock"])
        ).round(0).astype(int)

        velocity["reorder_capital_required"] = (velocity["recommended_reorder_qty"] * velocity["avg_price"]).round(2)

        return velocity.sort_values(by=["days_of_inventory", "total_revenue"], ascending=[True, False])

    # =========================================================================
    # RULE-BASED PRESCRIPTIVE EXECUTIVE ACTION CARDS
    # =========================================================================

    def get_prescriptive_insights(self) -> List[Dict[str, str]]:
        """
        Generate proactive actionable insights for executives based on engine analytics.
        """
        kpis = self.get_kpis()
        categories = self.get_category_performance()
        pareto = self.get_pareto_analysis()
        channel_data = self.get_channel_and_payment_insights()

        insights = []

        # 1. Discount leakage alert
        high_discount_cats = categories[categories["discount_applied_pct"] > 30.0]
        if not high_discount_cats.empty:
            top_leak = high_discount_cats.iloc[0]
            insights.append({
                "type": "Margin Protection",
                "severity": "High",
                "headline": f"Heavy Discounting in {top_leak['category']} ({top_leak['discount_applied_pct']}% of orders)",
                "action": f"Review discount eligibility for {top_leak['category']}. A 5% discount reduction could recover estimated ${round(top_leak['total_revenue'] * 0.05, 0):,} in net margin.",
                "category_tag": "Pricing & Discounts"
            })

        # 2. Pareto concentration insight
        class_a_count = len(pareto[pareto["abc_class"].str.startswith("A")])
        total_items = len(pareto)
        pct_items = round(class_a_count / total_items * 100, 1) if total_items > 0 else 0
        insights.append({
            "type": "Product Focus",
            "severity": "Medium",
            "headline": f"Top {class_a_count} products ({pct_items}% of catalog) drive 80% of revenue",
            "action": "Ensure supply chain priority & guaranteed in-stock SLA for these core Class A revenue drivers.",
            "category_tag": "Inventory & Restock"
        })

        # 3. Channel AOV Divergence
        ch_df = channel_data["channel"]
        if len(ch_df) >= 2:
            online_row = ch_df[ch_df["location"] == "Online"]
            instore_row = ch_df[ch_df["location"] == "In-store"]
            if not online_row.empty and not instore_row.empty:
                online_aov = online_row.iloc[0]["aov"]
                instore_aov = instore_row.iloc[0]["aov"]
                diff = abs(online_aov - instore_aov)
                higher_ch = "Online" if online_aov > instore_aov else "In-store"
                insights.append({
                    "type": "Omnichannel Strategy",
                    "severity": "Informational",
                    "headline": f"{higher_ch} channel delivers higher basket size (${max(online_aov, instore_aov)} vs ${min(online_aov, instore_aov)})",
                    "action": f"Deploy cross-sell product bundles in the lower-AOV channel to close the ${round(diff, 2)} basket gap.",
                    "category_tag": "Channel Strategy"
                })

        return insights


# Quick CLI test & verification
if __name__ == "__main__":
    engine = AnalyticalEngine()
    print("Engine KPIs:", engine.get_kpis())
    print("\nTop Categories:\n", engine.get_category_performance().head(3))
    print("\nPrescriptive Insights:\n", engine.get_prescriptive_insights())
