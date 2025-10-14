"""
AI Query Optimizer - Backend (Member 2)
---------------------------------------
This script connects to PostgreSQL (dvdrental),
generates random queries, extracts EXPLAIN (ANALYZE, JSON)
plans, and saves node-level features to CSV.

Output:
data/features/node_features.csv
"""

import psycopg2
import pandas as pd
import random
import json
import os
from datetime import datetime

# ---------------------- CONNECTION SETUP ----------------------
conn = psycopg2.connect(
    host="localhost",
    database="dvdrental",
    user="postgres",
    password="Harshit123",   # change this
    port=5432
)
cursor = conn.cursor()

# ---------------------- PATH SETUP ----------------------
os.makedirs("data/features", exist_ok=True)
csv_path = "data/features/node_features.csv"

# ---------------------- TABLE / COLUMN INFO ----------------------
# Common dvdrental tables & columns
schema = {
    "customer": ["customer_id", "first_name", "last_name", "email", "address_id", "active"],
    "payment": ["payment_id", "customer_id", "staff_id", "rental_id", "amount"],
    "rental": ["rental_id", "inventory_id", "customer_id", "staff_id", "return_date"],
    "film": ["film_id", "title", "rental_duration", "length", "rental_rate"],
    "inventory": ["inventory_id", "film_id", "store_id"],
    "store": ["store_id", "manager_staff_id", "address_id"],
    "category": ["category_id", "name"],
    "film_category": ["film_id", "category_id"],
    "city": ["city_id", "city", "country_id"],
    "country": ["country_id", "country"],
}

# ---------------------- RANDOM QUERY GENERATOR ----------------------
def generate_random_query():
    tables = list(schema.keys())
    t1 = random.choice(tables)

    # Simple SELECT
    if random.random() < 0.3:
        col = random.choice(schema[t1])
        query = f"SELECT {col} FROM {t1} LIMIT {random.randint(5,50)};"

    # Filter Query
    elif random.random() < 0.6:
        col = random.choice(schema[t1])
        query = f"SELECT * FROM {t1} WHERE {col} IS NOT NULL LIMIT {random.randint(10,100)};"

    # JOIN Query (2 tables)
    else:
        t2 = random.choice(tables)
        if t1 == t2:
            t2 = random.choice(tables)
        join_col = "customer_id" if "customer" in t1 or "customer" in t2 else None
        if not join_col:
            join_col = random.choice(schema[t1])
        query = f"SELECT * FROM {t1} t1 JOIN {t2} t2 ON 1=1 LIMIT {random.randint(10,200)};"
    return query

# ---------------------- EXPLAIN ANALYZE + EXTRACTION ----------------------
def extract_nodes(plan, parent=None, query_name=None):
    nodes = []
    node = {
        "node_type": plan.get("Node Type"),
        "relation": plan.get("Relation Name"),
        "filter": plan.get("Filter"),
        "join_type": plan.get("Join Type"),
        "plan_rows": plan.get("Plan Rows"),
        "actual_rows": plan.get("Actual Rows"),
        "total_cost": plan.get("Total Cost"),
        "actual_time": plan.get("Actual Total Time"),
        "parent_node": parent,
        "query_name": query_name
    }
    nodes.append(node)

    for subplan in plan.get("Plans", []):
        nodes.extend(extract_nodes(subplan, parent=plan.get("Node Type"), query_name=query_name))
    return nodes

# ---------------------- MAIN EXTRACTION ----------------------
all_nodes = []

print("🚀 Generating random queries and extracting plans...\n")

num_queries = 100   # You can increase this if you want even more data

for i in range(1, num_queries + 1):
    qname = f"q{i}"
    query = generate_random_query()

    try:
        cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
        result = cursor.fetchone()[0][0]
        nodes = extract_nodes(result["Plan"], query_name=qname)
        all_nodes.extend(nodes)
        print(f"✅ Extracted from {qname}")
    except Exception as e:
        print(f"⚠️ Skipped {qname} due to error: {e}")
        conn.rollback()

# ---------------------- SAVE TO CSV ----------------------
df = pd.DataFrame(all_nodes)
df["timestamp"] = datetime.now()

if os.path.exists(csv_path):
    df.to_csv(csv_path, mode="a", index=False, header=False)
else:
    df.to_csv(csv_path, index=False)

print(f"\n✅ Done! Extracted {len(df)} plan nodes from {num_queries} queries.")
print(f"📁 Saved CSV to: {csv_path}")

cursor.close()
conn.close()
