"""
ADAS AI Decision Service.
Provides conversational intelligence for Streamlit dashboard chat and generates executive HTML reports.
"""

from __future__ import annotations
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import polars as pl

# Project-level imports
ROOT_DIR = Path(__file__).resolve().parents[2]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mahgaurd import get_logger, timed, metrics
from engines.analytical_engine import AnalyticalEngine, DecisionManager
from .report_generator import generate_executive_html_report
from .llm_client import FreeLLMClient

logger = get_logger("ADAS.AI_Service")


class AIService:
    """
    Lightweight, resilient AI service for executive query answering and HTML report generation.
    Supports free cloud LLMs (Gemini, Groq, OpenRouter) with autonomous local analytical fallback.
    """

    def __init__(self, engine: Optional[AnalyticalEngine] = None, llm_client: Optional[FreeLLMClient] = None):
        self.engine = engine
        self.llm_client = llm_client or FreeLLMClient()
        self.decision_manager = DecisionManager()
        self.reports_dir = Path(__file__).resolve().parent / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        logger.info("AIService initialized with reports directory: %s (LLM configured: %s)", self.reports_dir, self.llm_client.is_configured())

    def __getattr__(self, name: str) -> Any:
        if name == "llm_client":
            self.llm_client = FreeLLMClient()
            return self.llm_client
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def set_llm_key(self, api_key: str, provider: str = "auto") -> bool:
        """Dynamically configure or switch LLM API provider at runtime."""
        self.llm_client = FreeLLMClient(api_key=api_key.strip(), provider=provider)
        logger.info(f"Updated LLM provider to '{self.llm_client.provider}' (Key set: {bool(api_key)})")
        return self.llm_client.is_configured()

    def _ensure_engine(self) -> AnalyticalEngine:
        """Lazy-load the analytical engine if not provided at instantiation."""
        if self.engine is None:
            self.engine = AnalyticalEngine(data_source="auto")
        return self.engine

    @timed("AIService.ask")
    def ask(self, question: str, filtered_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """
        Process executive user prompt, determine analytical intent, compute verified metrics,
        and formulate a structured markdown response (or generate HTML report).
        """
        q = question.strip().lower()
        engine = self._ensure_engine()
        df = filtered_df if filtered_df is not None else engine.df

        logger.info(f"AI query received: '{question}' (Evaluating against {len(df)} records)")

        # Intent 1: HTML Report Generation
        if any(w in q for w in ["report", "html", "export", "download", "pdf", "generate report", "create report"]):
            report_data = self.generate_html_report(df)
            kpis = engine.get_kpis(df)
            answer = (
                f"### 📄 Executive HTML Report Generated Successfully!\n\n"
                f"I have compiled a comprehensive executive analytics and decision report based on **{len(df):,} active records**.\n\n"
                f"**Report Highlights:**\n"
                f"- **Total Revenue Analyzed:** ${kpis['total_revenue']:,.2f}\n"
                f"- **Orders Processed:** {kpis['total_transactions']:,} | **AOV:** ${kpis['aov']:,.2f}\n"
                f"- **Included Sections:** Executive KPIs, 80/20 Pareto Classification, RFM Customer Cohorts, Critical Stockout Alerts, and Decision Audit Trails.\n\n"
                f"You can download or view the report directly using the controls below."
            )
            return {
                "answer": answer,
                "intent": "REPORT_GENERATION",
                "suggested_actions": [
                    "What are our top revenue categories?",
                    "Which products are at critical stockout risk?",
                    "Show customer RFM cohort distribution"
                ],
                "report_generated": True,
                "report_html": report_data["html"],
                "report_path": report_data["filepath"],
                "data_table": None,
            }

        # LLM Assisted Answering (if Free LLM Key is configured)
        if self.llm_client.is_configured():
            try:
                kpis = engine.get_kpis(df)
                growth = engine.get_period_comparison(df)
                cats = engine.get_category_performance(df)
                top_cat_str = f"{cats.iloc[0]['category']} (${cats.iloc[0]['total_revenue']:,.2f})" if not cats.empty else "N/A"
                
                context_str = (
                    f"Active Warehouse Context ({len(df):,} transactions):\n"
                    f"- Net Revenue: ${kpis['total_revenue']:,.2f}\n"
                    f"- Transactions: {kpis['total_transactions']:,} | AOV: ${kpis['aov']:,.2f}\n"
                    f"- MoM Growth: {growth.get('revenue_growth_pct', 0):+.1f}%\n"
                    f"- Leading Category: {top_cat_str}\n"
                    f"- Discounted Orders: {kpis['discount_rate_pct']}%\n"
                    f"- Unique Customers: {kpis['unique_customers']:,}"
                )
                
                llm_prompt = f"{context_str}\n\nExecutive Question: {question}\n\nDeliver a crisp, data-backed executive response with bullet points and concrete action recommendations."
                llm_reply, err = self.llm_client.generate(llm_prompt)
                if llm_reply:
                    prov = self.llm_client.provider.upper()
                    return {
                        "answer": f"*🤖 {prov} Free LLM Intelligence*\n\n{llm_reply}",
                        "intent": "FREE_LLM_ASSISTED",
                        "suggested_actions": [
                            "Generate executive HTML report",
                            "What are our top revenue categories?",
                            "Which products are at critical stockout risk?"
                        ],
                        "report_generated": False,
                        "report_html": None,
                        "report_path": None,
                        "data_table": None,
                    }
            except Exception as e:
                logger.warning(f"Free LLM generation error, proceeding with analytical fallback: {e}")

        # Intent 2: Revenue, Sales & MoM Growth
        if any(w in q for w in ["revenue", "sales", "growth", "aov", "spend", "financial", "margin", "mom"]):
            kpis = engine.get_kpis(df)
            growth = engine.get_period_comparison(df)
            growth_str = f"{growth['revenue_growth_pct']:+.1f}% MoM" if growth["revenue_growth_pct"] != 0 else "Stable"
            
            answer = (
                f"### 💰 Financial & Revenue Performance Summary\n\n"
                f"- **Total Net Revenue:** **${kpis['total_revenue']:,.2f}**\n"
                f"- **Month-over-Month Growth:** **{growth_str}** (Latest Period: {growth.get('latest_period', 'Current')})\n"
                f"- **Average Order Value (AOV):** **${kpis['aov']:,.2f}**\n"
                f"- **Total Transactions:** **{kpis['total_transactions']:,}** orders across **{kpis['total_units']:,}** units\n"
                f"- **Promotional Discount Rate:** **{kpis['discount_rate_pct']:.1f}%** of all orders received discounts.\n\n"
                f"💡 *Strategic Takeaway:* Average basket size sits at ${kpis['aov']:.2f}. "
                f"If you'd like to test the margin impact of raising prices or capping discounts, you can ask for a pricing simulation."
            )
            return {
                "answer": answer,
                "intent": "REVENUE_ANALYSIS",
                "suggested_actions": [
                    "Generate executive HTML report",
                    "What if we reduce discounts by 5%?",
                    "What are our top product categories?"
                ],
                "report_generated": False,
                "report_html": None,
                "report_path": None,
                "data_table": None,
            }

        # Intent 3: Inventory Velocity & Stockout Risk
        if any(w in q for w in ["inventory", "stock", "stockout", "reorder", "supply", "depletion", "lead time"]):
            inv = engine.get_inventory_and_reorder_recommendations(df_filtered=df)
            critical = inv[inv["stock_status"].str.contains("CRITICAL")]
            warning = inv[inv["stock_status"].str.contains("WARNING")]
            total_reorder_capital = inv["reorder_capital_required"].sum()

            answer = (
                f"### ⚠️ Inventory Run-Rate & Stockout Velocity Alerts\n\n"
                f"- **🚨 Critical Stockout Risk (<7 Days):** **{len(critical)} items**\n"
                f"- **🟡 Warning Depletion (<14 Days):** **{len(warning)} items**\n"
                f"- **Total Working Capital to Reorder:** **${total_reorder_capital:,.2f}**\n\n"
                f"**Most Urgent Items Requiring PO Issuance:**\n"
            )
            for idx, (_, row) in enumerate(critical.head(5).iterrows(), 1):
                answer += f"{idx}. **{row['item']}** ({row['category']}) — Only **{row['days_of_inventory']:.1f} days** left. Reorder: **+{row['recommended_reorder_qty']} units** (${row['reorder_capital_required']:,.2f})\n"

            answer += "\n💡 *Supply Chain Action:* Prioritize supplier expedites on critical items to safeguard core sales velocity."

            return {
                "answer": answer,
                "intent": "INVENTORY_RISK",
                "suggested_actions": [
                    "Generate executive HTML report",
                    "Show revenue and financial summary",
                    "Show active executive decisions"
                ],
                "report_generated": False,
                "report_html": None,
                "report_path": None,
                "data_table": critical.head(5),
            }

        # Intent 4: Product, Category & 80/20 Pareto
        if any(w in q for w in ["product", "category", "pareto", "80/20", "abc", "items", "best selling", "top"]):
            cat_df = engine.get_category_performance(df)
            pareto_df = engine.get_pareto_analysis(df)
            top_cat = cat_df.iloc[0] if not cat_df.empty else None
            class_a = pareto_df[pareto_df["abc_class"].str.startswith("A")]
            class_a_count = len(class_a)
            total_items = len(pareto_df)
            a_share = round((class_a_count / total_items * 100), 1) if total_items > 0 else 0

            top_cat_text = f"**{top_cat['category']}** (${top_cat['total_revenue']:,.2f}, {top_cat['revenue_share_pct']:.1f}% share)" if top_cat is not None else "N/A"

            answer = (
                f"### 🎯 Product & Pareto (80/20) Contribution Breakdown\n\n"
                f"- **#1 Leading Category:** {top_cat_text}\n"
                f"- **Pareto Core Drivers:** Just **{class_a_count} items** ({a_share}% of the catalog) generate **80% of total revenue** (Class A).\n"
                f"- **Total Catalog SKUs:** {total_items} distinct products evaluated.\n\n"
                f"**Top Class A Products:**\n"
            )
            for idx, (_, row) in enumerate(class_a.head(5).iterrows(), 1):
                answer += f"{idx}. **{row['item']}** ({row['category']}) — ${row['revenue']:,.2f} ({row['revenue_pct']:.1f}% share)\n"

            answer += "\n💡 *Prescriptive Recommendation:* Maintain uninterrupted supply contracts for Class A SKUs to avoid catastrophic revenue loss from stockouts."
            
            return {
                "answer": answer,
                "intent": "PRODUCT_PARETO",
                "suggested_actions": [
                    "Which products are at stockout risk?",
                    "Generate executive HTML report",
                    "How are our customer segments doing?"
                ],
                "report_generated": False,
                "report_html": None,
                "report_path": None,
                "data_table": cat_df.head(5),
            }

        # Intent 5: Customer RFM & Cohorts
        if any(w in q for w in ["customer", "rfm", "cohort", "churn", "loyalty", "retention", "champions"]):
            rfm = engine.get_customer_rfm_segments(df)
            kpis = engine.get_kpis(df)
            if not rfm.empty and "customer_segment" in rfm.columns:
                counts = rfm["customer_segment"].value_counts()
                champs = counts.get("Champions", 0)
                loyal = counts.get("Loyal Customers", 0)
                at_risk = counts.get("At Risk (High Past Value)", 0)
                lost = counts.get("Lost / Inactive", 0)
                total = len(rfm)

                answer = (
                    f"### 👥 Customer Cohorts & RFM Segmentation\n\n"
                    f"- **Total Unique Customers:** **{kpis['unique_customers']:,}**\n"
                    f"- **🏆 Champions:** **{champs:,}** ({champs/total*100:.1f}%) — High recency, frequency, and order volume.\n"
                    f"- **💎 Loyal Customers:** **{loyal:,}** ({loyal/total*100:.1f}%) — Steady repeat buyers.\n"
                    f"- **⚠️ At-Risk Accounts:** **{at_risk:,}** ({at_risk/total*100:.1f}%) — High lifetime spenders who haven't ordered recently.\n"
                    f"- **💤 Inactive / Lost:** **{lost:,}** ({lost/total*100:.1f}%).\n\n"
                    f"💡 *Retention Action:* Trigger immediate win-back campaigns for the **{at_risk} At-Risk customers** with personalized re-engagement offers before complete churn."
                )
            else:
                answer = f"Total unique customer base: **{kpis['unique_customers']:,}** accounts active in the dataset."

            return {
                "answer": answer,
                "intent": "CUSTOMER_RFM",
                "suggested_actions": [
                    "What are our top revenue categories?",
                    "Generate executive HTML report",
                    "What is our stockout risk?"
                ],
                "report_generated": False,
                "report_html": None,
                "report_path": None,
                "data_table": None,
            }

        # Intent 6: What-If Pricing & Margin Simulation
        if any(w in q for w in ["what if", "price change", "elasticity", "margin", "simulate", "discount reduction"]):
            sim = engine.simulate_what_if_scenario(price_change_pct=5.0, discount_reduction_pct=15.0, base_df=df)
            answer = (
                f"### 🔮 What-If Decision Simulation: +5% Price & -15% Discounts\n\n"
                f"- **Baseline Net Revenue:** ${sim['baseline_revenue']:,.2f}\n"
                f"- **Projected Net Revenue:** **${sim['projected_revenue']:,.2f}** ({sim['revenue_delta_pct']:+.2f}% / ${sim['revenue_delta']:+,.2f})\n"
                f"- **Gross Margin Delta:** **${sim['margin_delta']:+,.2f}** ({sim['margin_delta_pct']:+.2f}%)\n"
                f"- **Projected Volume Shift:** {sim['units_delta_pct']:.1f}% unit volume change due to price elasticity (-1.2 factor).\n\n"
                f"💡 *Executive Verdict:* Trimming promotional discount frequency by 15% recovers substantial gross profit margin without hurting top-line velocity."
            )
            return {
                "answer": answer,
                "intent": "WHAT_IF_SIMULATION",
                "suggested_actions": [
                    "Show active executive decisions",
                    "Generate executive HTML report",
                    "Which items drive 80% of revenue?"
                ],
                "report_generated": False,
                "report_html": None,
                "report_path": None,
                "data_table": None,
            }

        # Intent 7: Strategic Decisions Audit Log
        if any(w in q for w in ["decision", "decisions", "strategic", "audit", "status", "approved", "owner"]):
            decisions = self.decision_manager.get_all()
            answer = f"### ✍️ Executive Strategic Decisions Audit Log\n\n"
            if decisions:
                answer += f"Currently tracking **{len(decisions)} formal executive decisions**:\n\n"
                for d in decisions:
                    answer += (
                        f"- **[{d.get('decision_id')}] {d.get('title')}** (`{d.get('status')}`)\n"
                        f"  - **Question:** {d.get('business_question')}\n"
                        f"  - **Decision:** **{d.get('decision_answer')}**\n"
                        f"  - **Impact:** {d.get('projected_impact')} | *Owner:* {d.get('assigned_owner')}\n\n"
                    )
            else:
                answer += "No strategic decisions have been recorded yet. You can log one in the Strategic Decision Hub tab.\n"

            return {
                "answer": answer,
                "intent": "DECISIONS_LOG",
                "suggested_actions": [
                    "Generate executive HTML report",
                    "What are our top revenue categories?",
                    "Check inventory stockout risk"
                ],
                "report_generated": False,
                "report_html": None,
                "report_path": None,
                "data_table": None,
            }

        # Intent 8: General / Executive Overview
        kpis = engine.get_kpis(df)
        insights = engine.get_prescriptive_insights()
        answer = (
            f"### 👋 Hello! I am your ADAS AI Decision Copilot\n\n"
            f"I continuously monitor your transactional data warehouse (**{len(df):,} active records**).\n\n"
            f"**Current Business Snapshot:**\n"
            f"- **Net Revenue:** ${kpis['total_revenue']:,.2f} across {kpis['total_transactions']:,} transactions\n"
            f"- **Average Order Value (AOV):** ${kpis['aov']:,.2f}\n"
            f"- **Unique Customers:** {kpis['unique_customers']:,}\n\n"
            f"**Prescriptive Alerts Detected:**\n"
        )
        for ins in insights:
            answer += f"- **{ins.get('headline')}**: {ins.get('action')}\n"

        answer += (
            f"\n\n**How can I assist you today?**\n"
            f"- Type *'Generate HTML Report'* to create a full styled download\n"
            f"- Ask *'What are our top categories?'* or *'Pareto 80/20'* for SKU breakdowns\n"
            f"- Ask *'Show stockout risk'* to view urgent replenishment needs\n"
            f"- Ask *'Customer RFM'* to analyze retention and churn"
        )

        return {
            "answer": answer,
            "intent": "GENERAL_OVERVIEW",
            "suggested_actions": [
                "Generate executive HTML report",
                "What are our top revenue categories?",
                "Which products are at stockout risk?",
                "What if we raise prices by 5%?"
            ],
            "report_generated": False,
            "report_html": None,
            "report_path": None,
            "data_table": None,
        }

    @timed("AIService.generate_html_report")
    def generate_html_report(
        self,
        filtered_df: Optional[pl.DataFrame] = None,
        title: str = "ADAS Executive Analytical & Strategic Decision Report"
    ) -> Dict[str, Any]:
        """
        Pull all analytical slices and compile a self-contained executive HTML report.
        """
        engine = self._ensure_engine()
        df = filtered_df if filtered_df is not None else engine.df

        # Gather engine metrics
        kpis = engine.get_kpis(df)
        growth = engine.get_period_comparison(df)
        cat_df = engine.get_category_performance(df)
        pareto_df = engine.get_pareto_analysis(df)
        rfm_df = engine.get_customer_rfm_segments(df)
        inv_df = engine.get_inventory_and_reorder_recommendations(df_filtered=df)
        decisions = self.decision_manager.get_all()
        insights = engine.get_prescriptive_insights()

        # Render HTML string
        html_content = generate_executive_html_report(
            kpis=kpis,
            growth=growth,
            category_df=cat_df,
            pareto_df=pareto_df,
            rfm_df=rfm_df,
            inventory_df=inv_df,
            decisions_list=decisions,
            insights_list=insights,
            report_title=title,
        )

        # Save to disk
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"executive_report_{timestamp}.html"
        report_path = self.reports_dir / report_filename
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Also maintain a symlink/copy to latest_report.html
        latest_path = self.reports_dir / "latest_report.html"
        with open(latest_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated executive HTML report at {report_path.resolve()} ({len(html_content):,} bytes)")

        return {
            "html": html_content,
            "filepath": str(report_path),
            "filename": report_filename,
        }


# Global singleton helper
_ai_service_instance: Optional[AIService] = None


def get_ai_service(engine: Optional[AnalyticalEngine] = None, force_refresh: bool = False) -> AIService:
    """Retrieve or create the singleton AIService instance."""
    global _ai_service_instance
    if _ai_service_instance is None or force_refresh or not hasattr(_ai_service_instance, "llm_client"):
        _ai_service_instance = AIService(engine=engine)
    elif engine is not None:
        _ai_service_instance.engine = engine

    if not hasattr(_ai_service_instance, "llm_client"):
        _ai_service_instance.llm_client = FreeLLMClient()

    return _ai_service_instance
