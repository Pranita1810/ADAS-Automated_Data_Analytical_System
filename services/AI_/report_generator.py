"""
ADAS AI Service - Executive HTML Report Generator.
Generates self-contained, beautifully styled, print-ready HTML analytical reports.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


def generate_executive_html_report(
    kpis: Dict[str, Any],
    growth: Dict[str, Any],
    category_df: Optional[pd.DataFrame] = None,
    pareto_df: Optional[pd.DataFrame] = None,
    rfm_df: Optional[pd.DataFrame] = None,
    inventory_df: Optional[pd.DataFrame] = None,
    decisions_list: Optional[List[Dict[str, Any]]] = None,
    insights_list: Optional[List[Dict[str, Any]]] = None,
    report_title: str = "ADAS Executive Analytical & Strategic Decision Report",
) -> str:
    """
    Renders an executive-grade HTML report with responsive styling, KPI scorecards,
    Pareto product tables, RFM cohort distribution, stockout risk alerts, and strategic decisions.
    """
    generated_at = datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    # Format KPI values
    rev = f"${kpis.get('total_revenue', 0.0):,.2f}"
    orders = f"{kpis.get('total_transactions', 0):,}"
    aov = f"${kpis.get('aov', 0.0):,.2f}"
    units = f"{kpis.get('total_units', 0):,}"
    customers = f"{kpis.get('unique_customers', 0):,}"
    discount_pct = f"{kpis.get('discount_rate_pct', 0.0):.1f}%"

    rev_growth = growth.get("revenue_growth_pct", 0.0)
    rev_delta_badge = (
        f"<span class='badge badge-{'pos' if rev_growth >= 0 else 'neg'}'>{rev_growth:+.1f}% MoM</span>"
        if rev_growth != 0 else ""
    )

    # 1. Categories Table HTML
    cat_rows = ""
    if category_df is not None and not category_df.empty:
        for _, row in category_df.head(6).iterrows():
            cat_rows += f"""
            <tr>
                <td><strong>{row.get('category', 'N/A')}</strong></td>
                <td>${row.get('total_revenue', 0.0):,.2f}</td>
                <td>{row.get('revenue_share_pct', 0.0):.1f}%</td>
                <td>{int(row.get('units_sold', 0)):,}</td>
                <td>${row.get('aov', 0.0):,.2f}</td>
                <td>{row.get('discount_applied_pct', 0.0):.1f}%</td>
            </tr>
            """

    # 2. Pareto Class A Items Table HTML
    pareto_rows = ""
    if pareto_df is not None and not pareto_df.empty:
        class_a = pareto_df[pareto_df["abc_class"].str.startswith("A")].head(8)
        for _, row in class_a.iterrows():
            pareto_rows += f"""
            <tr>
                <td><span class='badge badge-a'>Class A</span></td>
                <td><strong>{row.get('item', 'N/A')}</strong></td>
                <td>{row.get('category', 'N/A')}</td>
                <td>${row.get('revenue', 0.0):,.2f}</td>
                <td>{row.get('revenue_pct', 0.0):.1f}%</td>
                <td>{row.get('cum_revenue_pct', 0.0):.1f}%</td>
            </tr>
            """

    # 3. Inventory Stockout Alerts HTML
    inv_rows = ""
    if inventory_df is not None and not inventory_df.empty:
        critical_items = inventory_df[inventory_df["stock_status"].str.contains("CRITICAL|WARNING")].head(6)
        if critical_items.empty:
            critical_items = inventory_df.head(5)
        for _, row in critical_items.iterrows():
            status = row.get("stock_status", "")
            badge_class = "badge-neg" if "CRITICAL" in status else "badge-warning"
            inv_rows += f"""
            <tr>
                <td><strong>{row.get('item', 'N/A')}</strong></td>
                <td>{row.get('category', 'N/A')}</td>
                <td>{row.get('daily_sales_velocity', 0.0):.1f} /day</td>
                <td>{row.get('days_of_inventory', 0.0):.1f} days</td>
                <td><span class='badge {badge_class}'>{status}</span></td>
                <td><strong>+{int(row.get('recommended_reorder_qty', 0)):,} units</strong></td>
                <td>${row.get('reorder_capital_required', 0.0):,.2f}</td>
            </tr>
            """

    # 4. RFM Cohorts HTML
    rfm_cards = ""
    if rfm_df is not None and not rfm_df.empty and "customer_segment" in rfm_df.columns:
        seg_counts = rfm_df["customer_segment"].value_counts()
        total_cust = len(rfm_df)
        for seg, count in seg_counts.items():
            pct = (count / total_cust * 100.0) if total_cust > 0 else 0
            rfm_cards += f"""
            <div class='rfm-card'>
                <div class='rfm-title'>{seg}</div>
                <div class='rfm-count'>{count:,} <span class='rfm-pct'>({pct:.1f}%)</span></div>
            </div>
            """

    # 5. Prescriptive Insights HTML
    insights_html = ""
    if insights_list:
        for ins in insights_list:
            insights_html += f"""
            <div class='insight-box'>
                <div class='insight-badge'>{ins.get('category_tag', 'Strategic')} | {ins.get('severity', 'Notice')}</div>
                <div class='insight-headline'>{ins.get('headline', '')}</div>
                <div class='insight-action'>💡 <strong>Action:</strong> {ins.get('action', '')}</div>
            </div>
            """

    # 6. Decisions Audit HTML
    decisions_html = ""
    if decisions_list:
        for d in decisions_list:
            decisions_html += f"""
            <div class='decision-card'>
                <div class='decision-header'>
                    <span class='decision-id'>{d.get('decision_id', '')}</span>
                    <span class='decision-title'>{d.get('title', '')}</span>
                    <span class='badge badge-pos'>{d.get('status', 'Approved')}</span>
                </div>
                <div class='decision-body'>
                    <div><strong>Question:</strong> {d.get('business_question', '')}</div>
                    <div><strong>Executive Answer:</strong> {d.get('decision_answer', '')}</div>
                    <div><strong>Projected Impact:</strong> {d.get('projected_impact', '')}</div>
                    <div class='decision-meta'>Owner: {d.get('assigned_owner', 'N/A')} | Updated: {d.get('updated_at', '')}</div>
                </div>
            </div>
            """
    else:
        decisions_html = "<p style='color: #64748b; font-style: italic;'>No strategic decisions logged yet.</p>"

    # Build Complete HTML Document
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{report_title}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-color: #334155;
            --accent-blue: #3b82f6;
            --accent-teal: #06b6d4;
            --accent-emerald: #10b981;
            --accent-rose: #ef4444;
            --accent-amber: #f59e0b;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-primary);
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        /* Header Banner */
        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid var(--border-color);
            border-left: 6px solid var(--accent-blue);
            border-radius: 12px;
            padding: 24px 28px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 16px;
        }}
        .header h1 {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 6px;
        }}
        .header p {{
            color: var(--text-secondary);
            font-size: 0.95rem;
        }}
        .print-btn {{
            background-color: var(--accent-blue);
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .print-btn:hover {{
            background-color: #2563eb;
        }}
        /* KPI Grid */
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .kpi-card {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .kpi-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
        }}
        /* Sections */
        .section {{
            background-color: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 22px;
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 10px;
            color: #f1f5f9;
        }}
        /* Tables */
        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.92rem;
        }}
        th {{
            background-color: rgba(255, 255, 255, 0.03);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border-color);
        }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            color: #e2e8f0;
        }}
        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        /* Badges */
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
        }}
        .badge-pos {{
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        .badge-neg {{
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        .badge-warning {{
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        .badge-a {{
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        /* RFM Grid */
        .rfm-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 14px;
        }}
        .rfm-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px;
        }}
        .rfm-title {{
            font-size: 0.88rem;
            color: var(--text-secondary);
            font-weight: 600;
        }}
        .rfm-count {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #ffffff;
            margin-top: 4px;
        }}
        .rfm-pct {{
            font-size: 0.85rem;
            color: var(--accent-teal);
            font-weight: normal;
        }}
        /* Insights & Decisions */
        .insight-box {{
            background: rgba(59, 130, 246, 0.06);
            border-left: 4px solid var(--accent-blue);
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 12px;
        }}
        .insight-badge {{
            font-size: 0.75rem;
            color: var(--accent-blue);
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 4px;
        }}
        .insight-headline {{
            font-weight: 700;
            font-size: 1rem;
            color: #f1f5f9;
            margin-bottom: 6px;
        }}
        .insight-action {{
            font-size: 0.9rem;
            color: #cbd5e1;
        }}
        .decision-card {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
        }}
        .decision-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 8px;
        }}
        .decision-id {{
            background-color: rgba(255, 255, 255, 0.08);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 700;
        }}
        .decision-title {{
            font-weight: 700;
            font-size: 1.05rem;
            color: #ffffff;
            flex-grow: 1;
        }}
        .decision-body {{
            font-size: 0.92rem;
            color: #cbd5e1;
            line-height: 1.6;
        }}
        .decision-meta {{
            margin-top: 8px;
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}
        /* Footer */
        .footer {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid var(--border-color);
        }}
        /* Print Stylesheet */
        @media print {{
            body {{
                background-color: #ffffff !important;
                color: #000000 !important;
                padding: 0;
            }}
            .print-btn {{
                display: none;
            }}
            .section, .kpi-card, .header {{
                background-color: #ffffff !important;
                border: 1px solid #ccc !important;
                color: #000000 !important;
                box-shadow: none !important;
            }}
            th, td, .kpi-label, .kpi-value, .rfm-title, .rfm-count, .insight-headline, .decision-title {{
                color: #000000 !important;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div>
                <h1>⚡ {report_title}</h1>
                <p>Generated by <strong>ADAS AI Decision Service</strong> on {generated_at}</p>
            </div>
            <button class="print-btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
        </div>

        <!-- KPI Scorecards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-value">{rev} {rev_delta_badge}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Orders</div>
                <div class="kpi-value">{orders}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Average Order Value</div>
                <div class="kpi-value">{aov}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Units Sold</div>
                <div class="kpi-value">{units}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Unique Customers</div>
                <div class="kpi-value">{customers}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Discount Penetration</div>
                <div class="kpi-value">{discount_pct}</div>
            </div>
        </div>

        <!-- Prescriptive Action Alerts -->
        <div class="section">
            <div class="section-title">💡 Strategic Prescriptive Alerts & Opportunities</div>
            {insights_html if insights_html else "<p style='color: #64748b;'>No critical risk alerts detected.</p>"}
        </div>

        <!-- Category Performance Table -->
        <div class="section">
            <div class="section-title">📊 Product Category Contribution</div>
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th>Revenue</th>
                        <th>Revenue Share</th>
                        <th>Units Sold</th>
                        <th>AOV</th>
                        <th>Discounted Orders %</th>
                    </tr>
                </thead>
                <tbody>
                    {cat_rows}
                </tbody>
            </table>
        </div>

        <!-- Pareto 80/20 Core Drivers -->
        <div class="section">
            <div class="section-title">🎯 Pareto (80/20) Class A Revenue Drivers</div>
            <table>
                <thead>
                    <tr>
                        <th>Classification</th>
                        <th>Product Item</th>
                        <th>Category</th>
                        <th>Revenue</th>
                        <th>Revenue %</th>
                        <th>Cumulative %</th>
                    </tr>
                </thead>
                <tbody>
                    {pareto_rows}
                </tbody>
            </table>
        </div>

        <!-- Inventory Stockout Risks -->
        <div class="section">
            <div class="section-title">⚠️ Critical Inventory Velocity & Restock Alerts</div>
            <table>
                <thead>
                    <tr>
                        <th>Product Item</th>
                        <th>Category</th>
                        <th>Sales Velocity</th>
                        <th>Days of Stock</th>
                        <th>Risk Status</th>
                        <th>Recommended Reorder</th>
                        <th>Capital Required</th>
                    </tr>
                </thead>
                <tbody>
                    {inv_rows}
                </tbody>
            </table>
        </div>

        <!-- Customer RFM Segments -->
        <div class="section">
            <div class="section-title">👥 Customer RFM Cohort Segmentation</div>
            <div class="rfm-grid">
                {rfm_cards}
            </div>
        </div>

        <!-- Strategic Decisions Log -->
        <div class="section">
            <div class="section-title">✍️ Active Executive Strategic Decisions</div>
            {decisions_html}
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>ADAS Automated Data Analytical System &bull; Confidential Executive Document &bull; Mahgaurd Verified</p>
        </div>
    </div>
</body>
</html>
"""
    return html
