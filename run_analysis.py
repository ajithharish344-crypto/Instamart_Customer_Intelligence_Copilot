
import sqlite3
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "instamart.db"

con = sqlite3.connect(DB_PATH)

# Load cleaned tables
customers = pd.read_sql_query(
    "SELECT * FROM customers", con
)

products = pd.read_sql_query(
    "SELECT * FROM products", con
)

orders = pd.read_sql_query(
    "SELECT * FROM orders", con
)

con.close()

# Convert required columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"]
)

# Completed orders only
completed = orders[
    orders["status"].str.lower() == "completed"
].copy()

# --------------------------------------------------
# 1. PANDAS REVENUE CROSS-CHECK
# --------------------------------------------------

merged = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id"
).merge(
    products[["product_id", "category"]],
    on="product_id"
)

city_category = merged.pivot_table(
    index=["city", "category"],
    values="order_amount",
    aggfunc="sum"
)

print("\nCITY AND CATEGORY REVENUE")
print(city_category.sort_values(
    "order_amount", ascending=False
).head(10))

print(
    "\nTOTAL COMPLETED REVENUE:",
    round(completed["order_amount"].sum(), 2)
)

# --------------------------------------------------
# 2. CITY-WISE SUMMARY
# --------------------------------------------------

city_summary = merged.groupby("city").agg(
    revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    order_count=("order_id", "count")
).round(2)

print("\nCITY-WISE SUMMARY")
print(city_summary)

# --------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

amount = completed["order_amount"]

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
lower_fence = q1 - 1.5 * iqr

print("\nORDER AMOUNT ANALYSIS")
print("Mean:", round(amount.mean(), 2))
print("Median:", round(amount.median(), 2))
print("Skewness:", round(amount.skew(), 2))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Above upper fence:", (amount > upper_fence).sum())
print("Below lower fence:", (amount < lower_fence).sum())

# --------------------------------------------------
# 4. CORRELATION
# --------------------------------------------------

corr = completed[
    ["order_amount", "discount_pct"]
].corr()

print("\nCORRELATION")
print(corr.round(3))

# --------------------------------------------------
# 5. SAVE ANALYSIS
# --------------------------------------------------

output = BASE_DIR / "outputs"
output.mkdir(exist_ok=True)

city_summary.to_csv(
    output / "city_revenue_summary.csv"
)

print("\nPANDAS CROSS-CHECK AND EDA COMPLETED")
