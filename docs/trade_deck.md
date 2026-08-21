# Madhu Siddharth Suthagar

::: {custom-style="ContactCenter"}
Chicago, IL 60616 | 312-508-1185 | [madhusiddharths1@outlook.com](mailto:madhusiddharths1@outlook.com) | [LinkedIn](https://www.linkedin.com/in/madhu-siddharth-suthagar/) | [GitHub](https://github.com/madhusiddharths) | [Portfolio](https://www.madhusiddharths.com)
:::

## Education

**Master of Science in Data Science** — Illinois Institute of Technology, Chicago, IL | GPA 3.77 | May 2026  
**B.Tech, Information Technology** — Coimbatore Institute of Technology, Coimbatore, India | GPA 3.66 | April 2024

## Certifications & Publications

First-Author IEEE Publication (ICAIC 2026) • AWS Certified Cloud Practitioner (Amazon Web Services, Jun 2025) • Google Data Analytics Certificate (Google, May 2022) • IBM Data Science Professional Certificate (IBM, Apr 2023)

## Skills

**SQL & Programming:** SQL (Advanced: joins, CTEs, temp tables, CASE WHEN, nested queries), Python, R  
**Data Platforms & Tools:** Databricks, Snowflake, BigQuery, Excel (pivot tables, charts, large data sets), MS Office  
**Visualization & Reporting:** Power BI, Tableau, Looker Studio, Streamlit, Plotly, Matplotlib, dashboard development  
**Analytics & Data Quality:** RFM/Audience Segmentation, Cohort Analysis, A/B Testing, Root Cause Analysis, Data QA  
**Data Engineering & Automation:** ETL, Python automation scripts, dbt, Pandas, Polars, Star Schema, Git/CI

## Experience

### Data Science Co-op | Labelmaster — Chicago, IL | Jan 2026 – May 2026

- Investigated a performance anomaly across 132K+ order lines and 26,290 sites, tracing $0-sales flags to a city-name merge defect ('Saint Louis' vs 'St. Louis') that hid 4,792 customers — root-cause fix recovered $40M+.
- Mapped the full commercial network for the sales team — 68.5% of 26,290 sites unserved and 83.6% without DGeo packaging — converting raw compliance and sales data into a scored, site-level prospect list.
- Delivered per-site product recommendations and a natural-language White Space Assistant under strict data-handling rules — non-identifiable fields only in LLM prompts, manual review before every external API call.
- Partnered with a Labelmaster project manager and 4–5 marketing stakeholders to translate business questions into reproducible reporting deliverables spanning 178 corporate families and 7 years of Ship-to sales history.

## Projects

### [Online Retail — Dual-Mode Analytics Pipeline (Star Schema, RFM & Cohorts)](https://www.madhusiddharths.com/work/online-retail/)

*SQL, BigQuery, Snowflake, Star Schema, RFM & Cohort Segmentation, Polars, Looker Studio, GitHub Actions CI*

- Surfaced £10.23M gross / £9.75M net revenue across 19,771 orders (£517 AOV, 4.6% return rate) and flagged an 85% UK revenue concentration and a +84% pre-Christmas seasonal peak (£1.45M vs £0.79M average).
- Segmented 4,334 customers into six RFM cohorts (Champions, Loyal, At Risk, Hibernating and more) and built monthly cohort retention that revealed a returning-gift-buyer echo resurging to 50% at month 11.
- Enforced tested data-quality rules over ~540K transactions — guest-bucketing ~135k null-customer rows preserved ~£1.5M of revenue — with GitHub Actions CI (ruff + pytest) guarding every KPI on each push.

### [E-Commerce Growth & Retention — A/B Test + Cart-Abandonment ML](https://www.madhusiddharths.com/work/growth-retention/)

*SQL (CTEs, window functions), BigQuery, dbt, A/B Testing (Chi-Square), Python, XGBoost, SHAP, Streamlit, Docker*

- Turned 20M+ clickstream events (~4.6 GB) into session- and user-grain marts in SQL — LAG() gap detection, a 30-minute inactivity timeout, and cumulative-sum session IDs — exposing lifetime value and recency.
- Validated a redesigned-checkout A/B experiment with a Chi-Square test of independence, recovering a statistically significant conversion lift of 6.73% → 10.83% (+4.1 percentage points, p < 0.001).
- Enforced data contracts in-pipeline with dbt schema tests (not_null, accepted_values) and shipped a multi-tab Streamlit app with a live BigQuery A/B dashboard of revenue and conversion metrics with deltas.

### [PulseTrade — Real-Time Financial Intelligence Platform](https://www.madhusiddharths.com/work/pulsetrade/)

*Databricks, Delta Lake, SQL, Kafka, Spark Structured Streaming, Airflow, Streamlit, Postgres, Docker, GKE*

- Designed Databricks Delta Lake gold-layer 5-minute feature windows (OHLC, volatility, mean sentiment) as the serving contract powering downstream dashboards and KPI views across bronze/silver/gold layers.
- Built a multi-page Streamlit analytics dashboard auto-refreshing every 30 seconds, surfacing live OHLC price trends, news-sentiment lines, and anomaly investigations across 5 tickers on a 5-minute cadence.
- Implemented a transparent statistical anomaly detector (2-sigma price moves, mean sentiment < -0.5, price stddev > 3x rolling average) flagging unusual market behavior for analyst triage, orchestrated in Airflow.