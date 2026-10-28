"""
ADAS - Executive Analytics & Decision Support Dashboard
Production-grade Streamlit application for Automated Data Analytical System.
"""

from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, date

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from engines.analytical_engine import AnalyticalEngine, DecisionManager
try:
    from service.AI_ import get_ai_service
except ImportError:
    from services.AI_ import get_ai_service
from mahgaurd import get_logger, get_system_health, get_metrics_summary, get_recent_logs

logger = get_logger("ADAS.Frontend")

# =============================================================================
# PAGE SETUP & CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="ADAS | Executive Analytics & Decision Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern executive aesthetic
st.markdown("""
<style>
    /* Metric Card styling */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 14px 18px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="metric-container"] label {
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #94a3b8;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    /* Action alert cards */
    .action-card {
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid #3b82f6;
        background: rgba(59, 130, 246, 0.08);
    }
    .action-card-high {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.08);
    }
    .action-card-medium {
        border-left-color: #f59e0b;
        background: rgba(245, 158, 11, 0.08);
    }
    .card-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 4px;
    }
    .card-body {
        font-size: 0.92rem;
        color: #cbd5e1;
    }
    /* Tab headers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA ENGINE CACHING & INITIALIZATION
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_engine() -> AnalyticalEngine:
    """Instantiate and cache the analytical engine."""
    return AnalyticalEngine(data_source="auto")


engine = get_engine()
decision_mgr = DecisionManager()
ai_service = get_ai_service(engine)
if not hasattr(ai_service, "llm_client"):
    try:
        from service.AI_.llm_client import FreeLLMClient
    except ImportError:
        from services.AI_.llm_client import FreeLLMClient
    ai_service.llm_client = FreeLLMClient()


# =============================================================================
# SIDEBAR CONTROLS & GLOBAL FILTERS
# =============================================================================
with st.sidebar:
    st.title("⚡ ADAS Analytics")
    st.caption("Automated Data Analytical System & Decision Engine")
    st.divider()

    # Raw data date bounds
    all_dates = engine.df["transaction_date"].drop_nulls()
    min_date = all_dates.min()
    max_date = all_dates.max()

    st.subheader("Filter Controls")

    # Date Range Filter
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    # Category Filter
    all_categories = sorted(engine.df["category"].unique().to_list())
    selected_categories = st.multiselect(
        "Product Categories",
        options=all_categories,
        default=all_categories,
    )

    # Location / Channel Filter
    all_locations = sorted(engine.df["location"].unique().to_list())
    selected_locations = st.multiselect(
        "Sales Channels",
        options=all_locations,
        default=all_locations,
    )

    # Payment Methods
    all_payments = sorted(engine.df["payment_method"].unique().to_list())
    selected_payments = st.multiselect(
        "Payment Methods",
        options=all_payments,
        default=all_payments,
    )

    st.divider()

    # Engine status indicator
    st.caption(f"📦 Total Warehouse Records: **{len(engine.df):,}**")
    if st.button("🔄 Reload Engine Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


# =============================================================================
# FILTER DATA ENGINE
# =============================================================================
def apply_filters():
    import polars as pl
    filtered = engine.df

    # Date filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
        filtered = filtered.filter(
            (pl.col("transaction_date") >= start_d) & (pl.col("transaction_date") <= end_d)
        )

    # Category filter
    if selected_categories:
        filtered = filtered.filter(pl.col("category").is_in(selected_categories))

    # Location filter
    if selected_locations:
        filtered = filtered.filter(pl.col("location").is_in(selected_locations))

    # Payment filter
    if selected_payments:
        filtered = filtered.filter(pl.col("payment_method").is_in(selected_payments))

    return filtered


filtered_df = apply_filters()

# Guard for empty selection
if len(filtered_df) == 0:
    st.warning("⚠️ No transactions match the current filter selection. Please broaden your filters in the sidebar.")
    st.stop()


# =============================================================================
# MAIN NAVIGATION TABS
# =============================================================================
tabs = st.tabs([
    "📊 Executive Overview",
    "🎯 Product & Pareto (80/20)",
    "👥 Customer RFM Cohorts",
    "🔮 What-If Simulator",
    "📦 Inventory & Stockout Risk",
    "✍️ Strategic Decision Hub",
    "🤖 AI Assistant & Chat",
    "🛡️ Mahgaurd Observability",
])


# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =============================================================================
with tabs[0]:
    kpis = engine.get_kpis(filtered_df)
    growth = engine.get_period_comparison(filtered_df)

    st.subheader("Executive KPI Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric(
            "Total Revenue",
            f"${kpis['total_revenue']:,.2f}",
            delta=f"{growth['revenue_growth_pct']:+.1f}% MoM" if growth['revenue_growth_pct'] != 0 else None
        )
    with c2:
        st.metric(
            "Total Orders",
            f"{kpis['total_transactions']:,}",
            delta=f"{growth['orders_growth_pct']:+.1f}% MoM" if growth['orders_growth_pct'] != 0 else None
        )
    with c3:
        st.metric(
            "Average Order Value",
            f"${kpis['aov']:,.2f}",
            delta=f"{growth['aov_growth_pct']:+.1f}% MoM" if growth['aov_growth_pct'] != 0 else None
        )
    with c4:
        st.metric("Units Sold", f"{kpis['total_units']:,}")
    with c5:
        st.metric("Unique Customers", f"{kpis['unique_customers']:,}")
    with c6:
        st.metric("Discount Penetration", f"{kpis['discount_rate_pct']:.1f}%")

    st.write("")

    # Prescriptive Insight Cards
    insights = engine.get_prescriptive_insights()
    if insights:
        st.markdown("##### 💡 Prescriptive Decision Alerts")
        icols = st.columns(len(insights))
        for col, item in zip(icols, insights):
            card_class = "action-card-high" if item["severity"] == "High" else "action-card-medium"
            with col:
                st.markdown(f"""
                <div class="action-card {card_class}">
                    <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #94a3b8;">
                        {item['type']} • {item['severity']} Priority
                    </div>
                    <div class="card-title">{item['headline']}</div>
                    <div class="card-body">{item['action']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Visualizations Row 1: Time Series & Category Breakdown
    g1, g2 = st.columns([7, 5])

    with g1:
        st.markdown("##### 📈 Revenue Velocity & Moving Averages")
        ts_df = engine.get_timeseries_trends(filtered_df)
        if not ts_df.empty:
            fig_ts = go.Figure()
            fig_ts.add_trace(go.Bar(
                x=ts_df["transaction_date"],
                y=ts_df["revenue"],
                name="Daily Revenue",
                opacity=0.35,
                marker_color="#38bdf8"
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts_df["transaction_date"],
                y=ts_df["revenue_7d_ma"],
                name="7-Day Moving Avg",
                line=dict(color="#f59e0b", width=2.5)
            ))
            fig_ts.add_trace(go.Scatter(
                x=ts_df["transaction_date"],
                y=ts_df["revenue_30d_ma"],
                name="30-Day Moving Avg",
                line=dict(color="#10b981", width=3, dash="dot")
            ))
            fig_ts.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_dark",
                hovermode="x unified"
            )
            st.plotly_chart(fig_ts, use_container_width=True)

    with g2:
        st.markdown("##### 🏷️ Category Revenue Contribution")
        cat_df = engine.get_category_performance(filtered_df)
        fig_cat = px.bar(
            cat_df,
            x="total_revenue",
            y="category",
            orientation="h",
            color="revenue_share_pct",
            color_continuous_scale="Blues",
            text="total_revenue",
            labels={"total_revenue": "Revenue ($)", "category": "Category"}
        )
        fig_cat.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_cat.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
            template="plotly_dark"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # Row 2: Channel & Day of Week Dynamics
    c_row1, c_row2 = st.columns(2)
    ch_data = engine.get_channel_and_payment_insights(filtered_df)

    with c_row1:
        st.markdown("##### 🏬 Channel & Payment Mix")
        fig_pay = px.pie(
            ch_data["payment"],
            names="payment_method",
            values="total_revenue",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pay.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark")
        st.plotly_chart(fig_pay, use_container_width=True)

    with c_row2:
        st.markdown("##### 📅 Day of Week Sales Velocity")
        dow_df = engine.get_day_of_week_distribution(filtered_df)
        fig_dow = px.bar(
            dow_df,
            x="day_name",
            y="revenue",
            text="revenue",
            color="revenue",
            color_continuous_scale="Viridis",
            labels={"revenue": "Revenue ($)", "day_name": "Day"}
        )
        fig_dow.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_dow.update_layout(
            height=280,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
            template="plotly_dark"
        )
        st.plotly_chart(fig_dow, use_container_width=True)


# =============================================================================
# TAB 2: PRODUCT & PARETO (80/20) ANALYSIS
# =============================================================================
with tabs[1]:
    st.subheader("Product Performance & Pareto (ABC) Classification")
    st.caption("Identify core revenue contributors (Class A: 80% revenue), steady products (Class B), and long-tail items (Class C).")

    pareto_df = engine.get_pareto_analysis(filtered_df)

    # Class summary counts
    class_summary = pareto_df.groupby("abc_class").agg(
        item_count=("item", "count"),
        total_rev=("revenue", "sum"),
        avg_price=("avg_price", "mean")
    ).reset_index()

    p1, p2, p3 = st.columns(3)
    with p1:
        a_row = class_summary[class_summary["abc_class"].str.startswith("A")]
        a_items = a_row["item_count"].values[0] if not a_row.empty else 0
        a_rev = a_row["total_rev"].values[0] if not a_row.empty else 0.0
        st.metric("Class A Products (Core 80%)", f"{a_items} items", f"${a_rev:,.0f} Revenue")
    with p2:
        b_row = class_summary[class_summary["abc_class"].str.startswith("B")]
        b_items = b_row["item_count"].values[0] if not b_row.empty else 0
        b_rev = b_row["total_rev"].values[0] if not b_row.empty else 0.0
        st.metric("Class B Products (Next 15%)", f"{b_items} items", f"${b_rev:,.0f} Revenue")
    with p3:
        c_row = class_summary[class_summary["abc_class"].str.startswith("C")]
        c_items = c_row["item_count"].values[0] if not c_row.empty else 0
        c_rev = c_row["total_rev"].values[0] if not c_row.empty else 0.0
        st.metric("Class C Products (Long Tail 5%)", f"{c_items} items", f"${c_rev:,.0f} Revenue")

    st.write("")

    # Pareto Cumulative Revenue Chart
    fig_pareto = go.Figure()
    fig_pareto.add_trace(go.Bar(
        x=list(range(1, len(pareto_df) + 1)),
        y=pareto_df["revenue"],
        name="Item Revenue ($)",
        marker_color="#38bdf8",
        opacity=0.6
    ))
    fig_pareto.add_trace(go.Scatter(
        x=list(range(1, len(pareto_df) + 1)),
        y=pareto_df["cum_revenue_pct"],
        name="Cumulative Revenue %",
        yaxis="y2",
        line=dict(color="#f43f5e", width=3)
    ))
    # 80% threshold line
    fig_pareto.add_hline(y=80, line_dash="dash", line_color="#eab308", yref="y2", annotation_text="80% Cutoff (Class A)")

    fig_pareto.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Products Ranked by Revenue",
        yaxis_title="Revenue ($)",
        yaxis2=dict(
            title="Cumulative Revenue %",
            overlaying="y",
            side="right",
            range=[0, 105],
            ticksuffix="%"
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark"
    )
    st.plotly_chart(fig_pareto, use_container_width=True)

    # Detailed Table with Filters
    st.markdown("##### 📋 Product Catalog ABC Performance Table")
    selected_abc = st.multiselect(
        "Filter by ABC Class",
        options=pareto_df["abc_class"].unique().tolist(),
        default=pareto_df["abc_class"].unique().tolist()
    )
    display_pareto = pareto_df[pareto_df["abc_class"].isin(selected_abc)].copy()
    display_pareto["revenue"] = display_pareto["revenue"].map("${:,.2f}".format)
    display_pareto["avg_price"] = display_pareto["avg_price"].map("${:,.2f}".format)
    display_pareto["cum_revenue_pct"] = display_pareto["cum_revenue_pct"].map("{:.1f}%".format)
    display_pareto["revenue_pct"] = display_pareto["revenue_pct"].map("{:.2f}%".format)

    st.dataframe(
        display_pareto[[
            "abc_class", "category", "item", "revenue", "revenue_pct",
            "cum_revenue_pct", "units", "orders", "avg_price"
        ]],
        use_container_width=True,
        hide_index=True,
        height=320
    )


# =============================================================================
# TAB 3: CUSTOMER RFM COHORTS
# =============================================================================
with tabs[2]:
    st.subheader("Customer Segmentation & Cohort Value Analysis")
    st.caption("Recency (R), Frequency (F), and Monetary (M) segmentation driving customer retention strategies.")

    rfm_df = engine.get_customer_rfm_segments(filtered_df)

    if not rfm_df.empty:
        col_r1, col_r2 = st.columns([5, 7])

        with col_r1:
            st.markdown("##### Cohort Distribution")
            seg_counts = rfm_df["customer_segment"].value_counts().reset_index()
            fig_rfm = px.pie(
                seg_counts,
                names="customer_segment",
                values="count",
                color_discrete_sequence=px.colors.qualitative.Bold,
                hole=0.4
            )
            fig_rfm.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark")
            st.plotly_chart(fig_rfm, use_container_width=True)

        with col_r2:
            st.markdown("##### Cohort Value & Activity Matrix")
            seg_perf = rfm_df.groupby("customer_segment").agg(
                customer_count=("customer_id", "count"),
                total_spend=("monetary", "sum"),
                avg_spend=("monetary", "mean"),
                avg_orders=("frequency", "mean"),
                avg_recency_days=("recency_days", "mean")
            ).reset_index().sort_values(by="total_spend", ascending=False)

            fig_seg_bar = px.bar(
                seg_perf,
                x="customer_segment",
                y="total_spend",
                color="avg_recency_days",
                text="total_spend",
                color_continuous_scale="Reds_r",
                labels={"total_spend": "Total Spend ($)", "avg_recency_days": "Avg Days Inactive"}
            )
            fig_seg_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
            fig_seg_bar.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), template="plotly_dark")
            st.plotly_chart(fig_seg_bar, use_container_width=True)

        st.markdown("##### 👥 Customer Target Roster")
        chosen_segment = st.selectbox("Select Cohort to Inspect / Target", options=rfm_df["customer_segment"].unique())
        cohort_table = rfm_df[rfm_df["customer_segment"] == chosen_segment].copy()
        cohort_table["monetary"] = cohort_table["monetary"].map("${:,.2f}".format)
        cohort_table["avg_order_value"] = cohort_table["avg_order_value"].map("${:,.2f}".format)
        st.dataframe(
            cohort_table[["customer_id", "customer_segment", "monetary", "frequency", "avg_order_value", "recency_days", "last_order_date"]],
            use_container_width=True,
            hide_index=True
        )


# =============================================================================
# TAB 4: WHAT-IF DECISION SCENARIO SIMULATOR
# =============================================================================
with tabs[3]:
    st.subheader("What-If Scenario Decision Simulator")
    st.caption("Forecast revenue and margin outcomes before committing strategic pricing or discount changes.")

    s1, s2 = st.columns([4, 8])

    with s1:
        st.markdown("##### 🎛️ Simulation Parameters")
        sim_price_pct = st.slider("Price Adjustment (%)", min_value=-20.0, max_value=30.0, value=5.0, step=0.5)
        sim_discount_red_pct = st.slider("Promotional Discount Reduction (%)", min_value=0.0, max_value=100.0, value=25.0, step=5.0)
        sim_elasticity = st.slider("Price Elasticity Coefficient", min_value=-2.5, max_value=-0.2, value=-1.1, step=0.1,
                                   help="Price elasticity determines demand sensitivity. -1.0 means 1% price rise reduces demand by 1%.")
        sim_target_cats = st.multiselect("Apply Scenario to Categories", options=all_categories, default=all_categories)

        st.info("💡 **Tip**: Inelastic categories (essential foods/beverages) tolerate price increases with minimal demand drop.")

    with s2:
        st.markdown("##### 🔮 Projected Strategic Outcome")
        sim_result = engine.simulate_what_if_scenario(
            price_change_pct=sim_price_pct,
            discount_reduction_pct=sim_discount_red_pct,
            elasticity_factor=sim_elasticity,
            target_categories=sim_target_cats,
            base_df=filtered_df
        )

        r_col1, r_col2, r_col3 = st.columns(3)
        with r_col1:
            st.metric(
                "Projected Revenue",
                f"${sim_result['projected_revenue']:,.2f}",
                delta=f"{sim_result['revenue_delta']:+,.2f} ({sim_result['revenue_delta_pct']:+.1f}%)"
            )
        with r_col2:
            st.metric(
                "Projected Gross Margin",
                f"${sim_result['projected_margin']:,.2f}",
                delta=f"{sim_result['margin_delta']:+,.2f} ({sim_result['margin_delta_pct']:+.1f}%)"
            )
        with r_col3:
            st.metric(
                "Projected Volume (Units)",
                f"{sim_result['projected_units']:,}",
                delta=f"{sim_result['units_delta_pct']:+.1f}% volume shift"
            )

        st.write("")

        # Visual Scenario Comparison Bar
        comp_df = pd.DataFrame({
            "Scenario": ["Current Baseline", "Projected Outcome"],
            "Revenue": [sim_result["baseline_revenue"], sim_result["projected_revenue"]],
            "Gross Margin": [sim_result["baseline_margin"], sim_result["projected_margin"]],
        })
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Revenue",
            x=comp_df["Scenario"],
            y=comp_df["Revenue"],
            marker_color="#38bdf8",
            text=comp_df["Revenue"],
            texttemplate="$%{text:,.0f}",
            textposition="inside"
        ))
        fig_comp.add_trace(go.Bar(
            name="Gross Margin",
            x=comp_df["Scenario"],
            y=comp_df["Gross Margin"],
            marker_color="#10b981",
            text=comp_df["Gross Margin"],
            texttemplate="$%{text:,.0f}",
            textposition="inside"
        ))
        fig_comp.update_layout(
            barmode="group",
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Quick Commit Button
        if st.button("📌 Convert This Scenario Into an Official Decision in the Hub", use_container_width=True):
            st.session_state["prefill_title"] = f"Price Adjustment ({sim_price_pct:+.1f}%) & Discount Cut ({sim_discount_red_pct:.0f}%)"
            st.session_state["prefill_category"] = "Pricing & Discounts"
            st.session_state["prefill_question"] = f"How should we adjust pricing on {', '.join(sim_target_cats[:2])} to maximize gross margin?"
            st.session_state["prefill_answer"] = f"Implement a {sim_price_pct:+.1f}% price adjustment and decrease discount frequency by {sim_discount_red_pct:.0f}%."
            st.session_state["prefill_impact"] = f"Projected {sim_result['revenue_delta']:+,.0f} revenue change and {sim_result['margin_delta']:+,.0f} gross margin improvement."
            st.success("Scenario details staged! Navigate to the '✍️ Strategic Decision Hub' tab to review and log it.")


# =============================================================================
# TAB 5: INVENTORY & STOCKOUT RISK
# =============================================================================
with tabs[4]:
    st.subheader("Inventory Velocity & Prescriptive Reorder Intelligence")
    st.caption("Prevent revenue loss by identifying items at imminent risk of stockout based on daily run rate velocity.")

    inv_df = engine.get_inventory_and_reorder_recommendations(df_filtered=filtered_df)

    # Risk metrics
    crit_count = len(inv_df[inv_df["stock_status"].str.startswith("CRITICAL")])
    warn_count = len(inv_df[inv_df["stock_status"].str.startswith("WARNING")])
    opt_count = len(inv_df[inv_df["stock_status"].str.startswith("OPTIMAL")])
    over_count = len(inv_df[inv_df["stock_status"].str.startswith("OVERSTOCKED")])
    total_reorder_capital = inv_df["reorder_capital_required"].sum()

    i1, i2, i3, i4, i5 = st.columns(5)
    with i1:
        st.metric("Critical Stockout Risk (<7d)", f"{crit_count} items", delta="Immediate Action", delta_color="inverse")
    with i2:
        st.metric("Reorder Warning (<14d)", f"{warn_count} items")
    with i3:
        st.metric("Optimal Stock (15-30d)", f"{opt_count} items")
    with i4:
        st.metric("Overstocked Items (>30d)", f"{over_count} items")
    with i5:
        st.metric("Capital Required to Restock", f"${total_reorder_capital:,.2f}")

    st.write("")

    # Filterable inventory table
    inv_status_filter = st.multiselect(
        "Filter by Inventory Status",
        options=inv_df["stock_status"].unique().tolist(),
        default=[s for s in inv_df["stock_status"].unique().tolist() if s.startswith(("CRITICAL", "WARNING"))]
    )

    filtered_inv = inv_df[inv_df["stock_status"].isin(inv_status_filter)].copy()
    filtered_inv["avg_price"] = filtered_inv["avg_price"].map("${:,.2f}".format)
    filtered_inv["reorder_capital_required"] = filtered_inv["reorder_capital_required"].map("${:,.2f}".format)
    filtered_inv["total_revenue"] = filtered_inv["total_revenue"].map("${:,.2f}".format)

    st.dataframe(
        filtered_inv[[
            "stock_status", "category", "item", "days_of_inventory",
            "daily_sales_velocity", "assumed_current_stock",
            "recommended_reorder_qty", "reorder_capital_required", "avg_price"
        ]],
        use_container_width=True,
        hide_index=True
    )


# =============================================================================
# TAB 6: STRATEGIC DECISION HUB (MAKE DECISIONS & ADD ANSWERS)
# =============================================================================
with tabs[5]:
    st.subheader("Executive Strategic Decision Hub")
    st.caption("Empower leadership to document real business questions, record actionable decision answers, and track status with a persistent audit trail.")

    # Top summary metrics of logged decisions
    existing_decisions = decision_mgr.get_all()
    total_decisions = len(existing_decisions)
    approved_count = sum(1 for d in existing_decisions if d.get("status") in ("Approved", "Implemented"))
    in_progress_count = sum(1 for d in existing_decisions if d.get("status") == "In Progress")
    proposed_count = sum(1 for d in existing_decisions if d.get("status") == "Proposed")

    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.metric("Total Decisions Logged", f"{total_decisions}")
    with d_col2:
        st.metric("Approved / Implemented", f"{approved_count}", delta=f"{(approved_count/total_decisions*100):.0f}%" if total_decisions else "0%")
    with d_col3:
        st.metric("In Progress", f"{in_progress_count}")
    with d_col4:
        st.metric("Proposed / Pending Review", f"{proposed_count}")

    st.divider()

    # SECTION 1: ADD NEW DECISION & ANSWER FORM
    with st.expander("➕ Log a New Executive Decision & Answer", expanded=True):
        # Handle prefill from what-if simulator if available
        def_title = st.session_state.get("prefill_title", "")
        def_cat = st.session_state.get("prefill_category", "Pricing & Discounts")
        def_q = st.session_state.get("prefill_question", "")
        def_ans = st.session_state.get("prefill_answer", "")
        def_impact = st.session_state.get("prefill_impact", "")

        with st.form("decision_entry_form", clear_on_submit=True):
            f1, f2 = st.columns([7, 5])
            with f1:
                dec_title = st.text_input("Decision Title *", value=def_title, placeholder="e.g. Optimize Discount Cap for Butchers Category")
            with f2:
                dec_category = st.selectbox(
                    "Decision Category *",
                    options=[
                        "Pricing & Discounts",
                        "Inventory & Restock",
                        "Marketing & Promotions",
                        "Channel Strategy",
                        "Product Assortment",
                        "Operations & Supply Chain"
                    ],
                    index=0 if def_cat == "Pricing & Discounts" else 1
                )

            dec_question = st.text_area(
                "Business Challenge / Question Addressed *",
                value=def_q,
                placeholder="What core analytical question or operational dilemma does this decision solve?",
                height=70
            )

            dec_answer = st.text_area(
                "Executive Decision & Direct Action (Your Answer) *",
                value=def_ans,
                placeholder="State clearly the decision answer and exact actions approved by leadership...",
                height=90
            )

            f3, f4, f5 = st.columns(3)
            with f3:
                dec_impact = st.text_input("Projected Financial / Strategic Impact", value=def_impact, placeholder="e.g. +$24,000 margin, 0 stockouts")
            with f4:
                dec_owner = st.text_input("Assigned Owner / Team *", placeholder="e.g. Sarah Jenkins (Head of Category)")
            with f5:
                dec_status = st.selectbox("Decision Status", options=["Approved", "Proposed", "In Progress", "Implemented"])

            dec_rationale = st.text_area(
                "Analytical Rationale & Evidence",
                placeholder="Reference key data points, Pareto findings, or What-If outputs supporting this decision...",
                height=70
            )

            submit_decision = st.form_submit_button("💾 Save Decision to System Record", use_container_width=True)

            if submit_decision:
                if not dec_title or not dec_question or not dec_answer or not dec_owner:
                    st.error("Please fill in all required fields (Title, Question, Decision Answer, and Owner).")
                else:
                    new_rec = decision_mgr.add_decision(
                        title=dec_title,
                        category=dec_category,
                        business_question=dec_question,
                        decision_answer=dec_answer,
                        rationale=dec_rationale,
                        projected_impact=dec_impact,
                        assigned_owner=dec_owner,
                        status=dec_status
                    )
                    # Clear session state prefill
                    for k in ["prefill_title", "prefill_category", "prefill_question", "prefill_answer", "prefill_impact"]:
                        st.session_state.pop(k, None)
                    st.success(f"✅ Decision {new_rec.decision_id} successfully logged and committed to audit log!")
                    st.rerun()

    # SECTION 2: DECISION AUDIT TRAIL & STATUS MANAGEMENT
    st.markdown("##### 📜 Active Decisions Log & Audit Trail")

    all_decs = decision_mgr.get_all()
    if not all_decs:
        st.info("No decisions logged yet. Use the form above to add your first executive decision.")
    else:
        # Filter controls for decision table
        df_col1, df_col2 = st.columns([4, 8])
        with df_col1:
            status_filter = st.multiselect(
                "Filter Log by Status",
                options=["All", "Approved", "In Progress", "Proposed", "Implemented"],
                default=["All"]
            )

        filtered_decs = all_decs
        if "All" not in status_filter and status_filter:
            filtered_decs = [d for d in all_decs if d.get("status") in status_filter]

        # Display Decisions in Card Format with Action buttons
        for item in reversed(filtered_decs):
            with st.container():
                d_head_col, d_status_col, d_btn_col = st.columns([8, 2, 2])

                with d_head_col:
                    st.markdown(f"**[{item['decision_id']}] {item['title']}** — `{item['category']}`")
                    st.caption(f"Owner: **{item['assigned_owner']}** | Logged: {item['created_at'][:10]}")

                with d_status_col:
                    status_colors = {
                        "Approved": "🟢",
                        "Implemented": "✅",
                        "In Progress": "🟡",
                        "Proposed": "🔵"
                    }
                    st.markdown(f"{status_colors.get(item['status'], '⚪')} **{item['status']}**")

                with d_btn_col:
                    new_st = st.selectbox(
                        "Change Status",
                        options=["Proposed", "Approved", "In Progress", "Implemented"],
                        index=["Proposed", "Approved", "In Progress", "Implemented"].index(item["status"]),
                        key=f"status_sel_{item['decision_id']}",
                        label_visibility="collapsed"
                    )
                    if new_st != item["status"]:
                        decision_mgr.update_decision_status(item["decision_id"], new_st)
                        st.rerun()

                # Card details body
                st.markdown(f"""
                > **Business Question:** {item['business_question']}  
                > **Executive Answer:** **{item['decision_answer']}**  
                > **Strategic Impact:** {item['projected_impact']}  
                > *Rationale:* {item['rationale']}
                """)

                # Delete button
                if st.button("🗑️ Delete", key=f"del_{item['decision_id']}"):
                    decision_mgr.delete_decision(item["decision_id"])
                    st.rerun()

                st.write("")

        # Export capabilities
        st.divider()
        csv_data = decision_mgr.to_pandas().to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Decision Log as CSV",
            data=csv_data,
            file_name=f"adas_decision_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# =============================================================================
# TAB 7: AI ASSISTANT & CHAT BOX
# =============================================================================
with tabs[6]:
    st.subheader("🤖 ADAS AI Decision Assistant & Executive Chat")
    st.caption("Ask questions about revenue, product Pareto analysis, customer churn, stockout risk, or generate an executive HTML report.")

    # Free LLM API Key Configuration Panel
    llm_obj = getattr(ai_service, "llm_client", None)
    is_llm_ready = llm_obj.is_configured() if llm_obj else False
    existing_key = getattr(llm_obj, "api_key", "") or ""
    current_prov = getattr(llm_obj, "provider", "none") or "none"

    with st.expander("🔑 Free LLM API Key Settings (Gemini / Groq / OpenRouter)", expanded=not is_llm_ready):
        cfg_c1, cfg_c2, cfg_c3 = st.columns([1.5, 2.5, 1])
        with cfg_c1:
            provider_choice = st.selectbox(
                "Free LLM Provider",
                options=["Google Gemini (Free)", "Groq Cloud (Free)", "OpenRouter (Free)"],
                index=0,
                key="free_llm_provider_select"
            )
            provider_key_map = {
                "Google Gemini (Free)": "gemini",
                "Groq Cloud (Free)": "groq",
                "OpenRouter (Free)": "openrouter"
            }
        with cfg_c2:
            input_key = st.text_input(
                "Enter Free API Key",
                type="password",
                value=existing_key,
                placeholder="e.g. AIzaSy... (Gemini) or gsk_... (Groq)",
                key="free_llm_key_input"
            )
            if provider_choice.startswith("Google"):
                st.caption("👉 Get free Gemini key (no credit card): [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)")
            elif provider_choice.startswith("Groq"):
                st.caption("👉 Get free Groq key (no credit card): [console.groq.com/keys](https://console.groq.com/keys)")
            else:
                st.caption("👉 Get free OpenRouter key: [openrouter.ai/keys](https://openrouter.ai/keys)")
        with cfg_c3:
            st.write("")
            st.write("")
            if st.button("💾 Apply Key", use_container_width=True, key="save_free_llm_btn"):
                if input_key:
                    chosen_p = provider_key_map[provider_choice]
                    if hasattr(ai_service, "set_llm_key"):
                        ai_service.set_llm_key(input_key, provider=chosen_p)
                    st.success(f"Configured {chosen_p.upper()}!")
                    st.rerun()
                else:
                    st.warning("Please enter a key.")

        status_text = (
            f"🟢 Active LLM: **{current_prov.upper()} Free Tier**"
            if is_llm_ready
            else "⚪ Active Mode: **Autonomous Local Analytics** (No external API key required)"
        )
        st.markdown(status_text)

    # Initialize chat history
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = [
            {
                "role": "assistant",
                "content": (
                    "👋 **Welcome to the ADAS AI Executive Assistant!**\n\n"
                    "I am connected directly to your analytical engine and active data warehouse.\n\n"
                    "**You can ask me questions like:**\n"
                    "- *'What is our total revenue and month-over-month growth?'*\n"
                    "- *'Which products drive 80% of our revenue (Pareto Class A)?'*\n"
                    "- *'Show customer RFM cohorts and churn risk'* \n"
                    "- *'Which inventory items are at critical stockout risk?'*\n"
                    "- *'What if we reduce promotional discounts by 15%?'*\n"
                    "- *'Generate executive HTML report'* 📄\n\n"
                    "Click any quick action below or type your question!"
                ),
                "report_html": None,
                "report_filename": None,
            }
        ]

    # Quick Action Prompt Buttons
    btn_c1, btn_c2, btn_c3, btn_c4, btn_c5, btn_c6 = st.columns(6)
    quick_prompt = None

    with btn_c1:
        if st.button("📄 Generate HTML Report", use_container_width=True):
            quick_prompt = "Generate executive HTML report"
    with btn_c2:
        if st.button("💰 Revenue & Growth", use_container_width=True):
            quick_prompt = "What is our total revenue and growth summary?"
    with btn_c3:
        if st.button("🎯 Pareto 80/20 SKUs", use_container_width=True):
            quick_prompt = "Which products generate 80% of our revenue?"
    with btn_c4:
        if st.button("👥 Customer RFM", use_container_width=True):
            quick_prompt = "How are our customer RFM cohorts performing?"
    with btn_c5:
        if st.button("⚠️ Stockout Risks", use_container_width=True):
            quick_prompt = "Which products are at stockout risk?"
    with btn_c6:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.ai_chat_history = [
                {
                    "role": "assistant",
                    "content": "Chat history cleared. How can I help you analyze your data?",
                    "report_html": None,
                    "report_filename": None,
                }
            ]
            st.rerun()

    st.write("")

    # Display Chat Messages
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.ai_chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # If this message generated an HTML report, provide download & live preview
                if msg.get("report_html"):
                    dl_col1, dl_col2 = st.columns([1, 2])
                    fname = msg.get("report_filename") or f"adas_executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    with dl_col1:
                        st.download_button(
                            label="📥 Download Executive HTML Report",
                            data=msg["report_html"],
                            file_name=fname,
                            mime="text/html",
                            key=f"dl_btn_{idx}",
                            use_container_width=True,
                        )
                    with st.expander("👁️ Preview Generated HTML Report Inside Dashboard", expanded=False):
                        import streamlit.components.v1 as components
                        components.html(msg["report_html"], height=650, scrolling=True)

    # Chat Input Box
    chat_input = st.chat_input("Ask a question about sales, inventory, customers, or request an HTML report...")
    active_prompt = quick_prompt or chat_input

    if active_prompt:
        # Add user message to state
        st.session_state.ai_chat_history.append({"role": "user", "content": active_prompt})

        # Process with AI Service
        with st.spinner("🤖 Analyzing data and formulating executive intelligence..."):
            res = ai_service.ask(active_prompt, filtered_df)

        # Append assistant message
        st.session_state.ai_chat_history.append({
            "role": "assistant",
            "content": res["answer"],
            "report_html": res.get("report_html"),
            "report_filename": Path(res["report_path"]).name if res.get("report_path") else None,
        })
        st.rerun()


# =============================================================================
# TAB 8: MAHA GAURD OBSERVABILITY & CENTRAL LOGS
# =============================================================================
with tabs[7]:
    st.subheader("🛡️ Mahgaurd Observability & Central Telemetry Hub")
    st.caption("Unified system health monitoring, operation latency metrics, and real-time project logs.")

    health = get_system_health()
    metrics_data = get_metrics_summary()

    # Subsystem Health Status Scorecards
    h_c1, h_c2, h_c3, h_c4 = st.columns(4)

    with h_c1:
        st.metric(
            "System Status",
            health.get("status", "HEALTHY"),
            delta="Operational",
            delta_color="normal" if health.get("status") == "HEALTHY" else "inverse"
        )
    with h_c2:
        backups_status = health["subsystems"].get("Backups_Storage", {})
        st.metric("Parquet Backups", backups_status.get("status", "ONLINE"), help=backups_status.get("details"))
    with h_c3:
        hub_status = health["subsystems"].get("Strategic_Decision_Hub", {})
        st.metric("Decision Hub", hub_status.get("status", "ONLINE"), help=hub_status.get("details"))
    with h_c4:
        log_status = health["subsystems"].get("Mahgaurd_Logging", {})
        st.metric("Central Logging", log_status.get("status", "ONLINE"), help=log_status.get("details"))

    st.write("")

    # Host Resources Grid (CPU & RAM)
    res_data = health.get("host_resources", {})
    if "cpu_percent" in res_data:
        st.markdown("##### 💻 Server & Host Metrics")
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("Host CPU Usage", res_data.get("cpu_percent", "N/A"))
        with rc2:
            st.metric("Host RAM Usage", res_data.get("memory_percent", "N/A"))
        with rc3:
            st.metric("Available RAM", f"{res_data.get('memory_available_mb', 0):,.1f} MB")
        with rc4:
            st.metric("Uptime", f"{metrics_data.get('uptime_seconds', 0):,.0f}s")

    st.divider()

    # Telemetry Latencies Table
    st.markdown("##### ⏱️ Subsystem Function Latencies & Call Counts")
    ops = metrics_data.get("operations", {})
    if ops:
        ops_rows = []
        for op_name, op_stats in ops.items():
            ops_rows.append({
                "Operation": op_name,
                "Invocations": op_stats["count"],
                "Avg Latency (ms)": op_stats["avg_ms"],
                "Min (ms)": op_stats["min_ms"],
                "Max (ms)": op_stats["max_ms"],
                "Last (ms)": op_stats["last_ms"],
                "Errors": op_stats["errors"],
                "Last Execution": op_stats["last_seen"],
            })
        ops_df = pd.DataFrame(ops_rows).sort_values("Invocations", ascending=False)
        st.dataframe(ops_df, use_container_width=True, hide_index=True)
    else:
        st.info("No timed operations logged in this session yet. Interact with the dashboard or AI chat to populate telemetry.")

    st.divider()

    # Central Log Stream
    st.markdown("##### 📜 Mahgaurd Central Project Log Stream")
    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        log_filter = st.selectbox("Filter Level", options=["ALL", "INFO", "WARNING", "ERROR"], index=0)
    with col_f2:
        log_lines_count = st.selectbox("Lines to show", options=[50, 100, 200, 500], index=1)
    with col_f3:
        st.write("")
        st.write("")
        if st.button("🔄 Refresh Logs", use_container_width=True):
            st.rerun()

    raw_logs = get_recent_logs(lines=log_lines_count)
    if log_filter != "ALL":
        filtered_logs = [l for l in raw_logs if f"| {log_filter}" in l or log_filter in l]
    else:
        filtered_logs = raw_logs

    log_text = "".join(filtered_logs) if filtered_logs else f"No logs matching level '{log_filter}'."
    st.code(log_text, language="log")

    # Download Log File
    st.download_button(
        label="📥 Download Complete adas.log File",
        data=log_text.encode("utf-8"),
        file_name=f"adas_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        mime="text/plain",
        use_container_width=True
    )

