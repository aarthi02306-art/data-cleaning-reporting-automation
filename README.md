"""
Data Cleaning & Reporting Automation
Run:  python clean_and_report.py
      python clean_and_report.py --input my_data.csv

Steps: load -> clean (missing, duplicates, inconsistent) -> save cleaned CSV -> HTML report with charts
"""
import argparse
import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAW_FILE = "raw_data.csv"
CLEAN_FILE = "cleaned_data.csv"
REPORT_FILE = "report.html"


# ---------------------------------------------------------------- sample data
def make_sample_data(path, n=200):
    """Create a messy dataset so the script can be demoed without any input file."""
    rng = np.random.default_rng(42)
    cities = ["Chennai", "chennai ", "CHENNAI", "Chenai", "Mumbai", "mumbai",
              "Delhi", "DELHI ", "Bengaluru", "bangalore"]
    cats = ["Electronics", "electronics", "Clothing", "clothing ", "Grocery", "GROCERY", "Books"]
    dates = pd.date_range("2025-01-01", "2025-12-31", periods=n)
    fmts = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y"]  # mixed date formats

    df = pd.DataFrame({
        "Order ID": range(1001, 1001 + n),
        "Customer Name": rng.choice(["  arun kumar", "PRIYA S", "divya R", "Karthik ", "meena devi"], n),
        "City": rng.choice(cities, n),
        "Category": rng.choice(cats, n),
        "Quantity": rng.integers(1, 10, n).astype(float),
        "Price": rng.integers(100, 5000, n).astype(float),
        "Order Date": [d.strftime(fmts[i % 3]) for i, d in enumerate(dates)],
    })
    df["Price"] = df["Price"].astype(object)
    for col in ["Quantity", "Price", "City", "Customer Name"]:  # missing values
        df.loc[rng.choice(n, 10, replace=False), col] = np.nan
    df.loc[rng.choice(n, 3, replace=False), "Price"] = "N/A"  # junk text in numeric column
    df = pd.concat([df, df.sample(12, random_state=1)], ignore_index=True)  # duplicates
    df.to_csv(path, index=False)


# ------------------------------------------------------------------- cleaning
def clean_data(df):
    log = {"Rows (raw)": len(df), "Missing cells (before)": int(df.isna().sum().sum())}

    # 1. standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # 2. trim spaces + fix text case
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].str.strip()
    for c in ["customer_name", "city", "category"]:
        df[c] = df[c].str.title()

    # 3. fix known spelling variants
    df["city"] = df["city"].replace({"Chenai": "Chennai", "Bangalore": "Bengaluru"})

    # 4. fix data types (bad values like "N/A" become NaN)
    for c in ["quantity", "price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["order_date"] = pd.to_datetime(df["order_date"], format="mixed", dayfirst=True, errors="coerce")

    # 5. remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    log["Duplicates removed"] = before - len(df)

    # 6. handle missing values
    for c in ["quantity", "price"]:
        df[c] = df[c].fillna(df[c].median())
    df["quantity"] = df["quantity"].round().astype(int)
    for c in ["customer_name", "city"]:
        df[c] = df[c].fillna("Unknown")
    before = len(df)
    df = df.dropna(subset=["order_date"])
    log["Rows dropped (invalid date)"] = before - len(df)

    # 7. derived column
    df["revenue"] = df["quantity"] * df["price"]

    log["Rows (clean)"] = len(df)
    log["Missing cells (after)"] = int(df.isna().sum().sum())
    return df.reset_index(drop=True), log


# ------------------------------------------------------------------ reporting
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_charts(df):
    charts = {}

    fig, ax = plt.subplots(figsize=(6, 3.6))
    df.groupby("category")["revenue"].sum().sort_values().plot.barh(ax=ax, color="#4C78A8")
    ax.set_title("Revenue by Category")
    ax.set_xlabel("Revenue")
    charts["Revenue by Category"] = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    df.groupby("city")["revenue"].sum().sort_values().plot.barh(ax=ax, color="#F58518")
    ax.set_title("Revenue by City")
    ax.set_xlabel("Revenue")
    charts["Revenue by City"] = fig_to_base64(fig)

    fig, ax = plt.subplots(figsize=(12, 3.6))
    monthly = df.set_index("order_date")["revenue"].resample("MS").sum()
    ax.plot(monthly.index, monthly.values, marker="o", color="#54A24B")
    ax.set_title("Monthly Revenue Trend")
    ax.set_ylabel("Revenue")
    ax.grid(alpha=0.3)
    charts["Monthly Revenue Trend"] = fig_to_base64(fig)

    return charts


def build_report(df, log, charts, path):
    log_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in log.items())
    kpis = {
        "Total Revenue": f"{df['revenue'].sum():,.0f}",
        "Orders": f"{len(df):,}",
        "Avg Order Value": f"{df['revenue'].mean():,.0f}",
        "Top Category": df.groupby("category")["revenue"].sum().idxmax(),
    }
    kpi_html = "".join(f"<div class='kpi'><span>{k}</span><b>{v}</b></div>" for k, v in kpis.items())
    imgs = "".join(
        f"<div class='card'><img src='data:image/png;base64,{b}' alt='{t}'></div>"
        for t, b in charts.items()
    )
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Data Cleaning Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:30px;background:#f5f6fa;color:#222}}
h1{{margin-bottom:4px}} .sub{{color:#666;margin-bottom:20px}}
.kpis{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:20px}}
.kpi{{background:#fff;padding:14px 22px;border-radius:8px;box-shadow:0 1px 4px #0002}}
.kpi span{{display:block;font-size:12px;color:#777}} .kpi b{{font-size:22px}}
.grid{{display:flex;gap:14px;flex-wrap:wrap}}
.card{{background:#fff;padding:10px;border-radius:8px;box-shadow:0 1px 4px #0002}}
.card img{{max-width:100%}}
table{{border-collapse:collapse;background:#fff;margin:10px 0 24px}}
td,th{{border:1px solid #ddd;padding:6px 12px;font-size:14px}} th{{background:#eef}}
</style></head><body>
<h1>Data Cleaning &amp; Reporting</h1>
<div class="sub">Generated automatically on {pd.Timestamp.now():%d %b %Y, %H:%M}</div>
<div class="kpis">{kpi_html}</div>
<h2>Cleaning Summary</h2>
<table><tr><th>Step</th><th>Result</th></tr>{log_rows}</table>
<h2>Visual Summary</h2>
<div class="grid">{imgs}</div>
<h2>Cleaned Data (first 10 rows)</h2>
{df.head(10).to_html(index=False)}
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=RAW_FILE, help="path to raw CSV file")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"{args.input} not found - creating sample messy data...")
        make_sample_data(args.input)

    raw = pd.read_csv(args.input)
    clean, log = clean_data(raw)
    clean.to_csv(CLEAN_FILE, index=False)
    build_report(clean, log, build_charts(clean), REPORT_FILE)

    print("Cleaning summary:")
    for k, v in log.items():
        print(f"  {k}: {v}")
    print(f"\nSaved: {CLEAN_FILE}, {REPORT_FILE}")


if __name__ == "__main__":
    main()
