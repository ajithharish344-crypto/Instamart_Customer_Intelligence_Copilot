
import sqlite3
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
DB = BASE / "data" / "instamart.db"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

con = sqlite3.connect(DB)

customers = pd.read_sql_query("SELECT * FROM customers", con)
products = pd.read_sql_query("SELECT * FROM products", con)
orders = pd.read_sql_query("SELECT * FROM orders", con)

con.close()

orders["order_amount"] = pd.to_numeric(orders["order_amount"])
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
completed = orders[
    orders["status"].str.lower() == "completed"
].copy()

# Basic database checks
assert len(customers) == 1200
assert len(products) == 40
assert len(orders) == 12309
assert len(completed) == 11110

revenue = completed["order_amount"].sum()
assert abs(revenue - 2998668.80) < 0.01

# Customer checks
assert customers["age"].notna().all()
assert customers["age"].between(18, 60).all()

gender_counts = (
    customers["gender"].str.lower().value_counts().to_dict()
)

# EDA calculations
amount = completed["order_amount"]
q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

correlation = completed[
    ["order_amount", "discount_pct"]
].corr().iloc[0, 1]

# City summary
city_data = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id"
)

city_summary = city_data.groupby("city").agg(
    revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    order_count=("order_id", "count")
).round(2)

city_summary.to_csv(OUT / "city_revenue_summary.csv")

# Create report
report = f"""
PART A - FINAL VERIFICATION REPORT

DATABASE COUNTS
Customers: {len(customers)}
Products: {len(products)}
Orders: {len(orders)}
Completed orders: {len(completed)}
Cancelled orders: {(orders['status'].str.lower() == 'cancelled').sum()}

COMPLETED REVENUE
INR {revenue:,.2f}

CUSTOMER CHECKS
Age minimum: {customers['age'].min()}
Age maximum: {customers['age'].max()}
Gender counts: {gender_counts}

ORDER AMOUNT EDA
Mean: INR {amount.mean():.2f}
Median: INR {amount.median():.2f}
Skewness: {amount.skew():.2f}
Q1: INR {q1:.2f}
Q3: INR {q3:.2f}
Lower fence: INR {lower:.2f}
Upper fence: INR {upper:.2f}
Above upper fence: {(amount > upper).sum()}
Below lower fence: {(amount < lower).sum()}

CORRELATION
Order amount vs discount percentage: {correlation:.3f}

VALIDATION
All database and revenue assertions passed.
"""

report_path = OUT / "part_a_summary.txt"
report_path.write_text(report.strip(), encoding="utf-8")

print(report)
print("\nPART A FINAL VERIFICATION PASSED")
print("Saved:", report_path)
