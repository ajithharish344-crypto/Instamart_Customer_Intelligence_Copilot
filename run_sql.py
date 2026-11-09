
import sqlite3
import pandas as pd
import re
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "instamart.db"
SQL_PATH = BASE_DIR / "sql" / "queries.sql"

# --------------------------------------------------
# 2. CHECK REQUIRED FILES
# --------------------------------------------------

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

if not SQL_PATH.exists():
    raise FileNotFoundError(f"SQL file not found: {SQL_PATH}")

# --------------------------------------------------
# 3. READ SQL FILE
# --------------------------------------------------

sql_text = SQL_PATH.read_text(encoding="utf-8")

# Split queries using headings such as:
# -- Q1: ...
# -- Q2: ...

parts = re.split(
    r"(?im)^\s*--\s*Q([1-7])\b[^\n]*\n",
    sql_text
)

if len(parts) != 15:
    raise ValueError(
        "Expected 7 SQL query sections. "
        "Check that queries.sql contains headings "
        "such as '-- Q1:', '-- Q2:' ... '-- Q7:'."
    )

queries = {}

for i in range(1, len(parts), 2):
    query_number = int(parts[i])
    query_text = parts[i + 1].strip()

    # Remove empty lines and comment-only lines
    query_lines = [
        line for line in query_text.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]

    queries[query_number] = "\n".join(query_lines).strip()

# --------------------------------------------------
# 4. EXECUTE QUERIES USING PANDAS
# --------------------------------------------------

connection = sqlite3.connect(DB_PATH)

try:
    for number in range(1, 8):

        print("\n" + "=" * 65)
        print(f"QUERY {number} RESULTS")
        print("=" * 65)

        result = pd.read_sql_query(
            queries[number],
            connection
        )

        print(result.to_string(index=False))
        print("\nRows returned:", len(result))

        # Save each query result
        output_path = (
            BASE_DIR / "outputs" /
            f"Q{number}_result.csv"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        result.to_csv(output_path, index=False)

        print("Saved:", output_path.name)

finally:
    connection.close()

print("\nALL 7 SQL QUERIES EXECUTED SUCCESSFULLY")
