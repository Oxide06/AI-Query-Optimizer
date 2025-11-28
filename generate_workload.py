#!/usr/bin/env python3
"""
generate_lce_data.py - FIXED VERSION
Generates training data without data leakage
"""

import psycopg2
import pandas as pd
import random
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

DB_CONFIG = {
    "host": "localhost",
    "dbname": "DVD_rental",
    "user": "postgres",
    "password": "Octane@06"
}

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "lce_training_data_fixed.csv"
TARGET_NODES = 15_000  # Increased for better coverage
MAX_QUERIES = 4_000
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# IMPROVED NORMALIZATION
def normalize_filter(s):
    if not s or s == "None": 
        return "none"
    s = re.sub(r'[a-zA-Z_]+\.', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace('(', '').replace(')', '')
    s = s.lower()
    # Generalize numeric values
    s = re.sub(r'\b\d+\b', '#', s)
    return s

def normalize_cond(s):
    if not s or s == "None": 
        return "none"
    cols = re.findall(r'([a-zA-Z_]+)\.([a-zA-Z_]+)', s)
    if len(cols) == 2:
        return f"{cols[0][1]}={cols[1][1]}".lower()
    return "none"

# EXPANDED TEMPLATES for better variety
TEMPLATES = [
    {
        "name": "film_category_query",
        "sql": """
            SELECT f.title, f.length, c.name
            FROM film f
            JOIN film_category fc ON f.film_id = fc.film_id
            JOIN category c ON fc.category_id = c.category_id
            WHERE f.length > %s AND c.name = %s
            ORDER BY f.length DESC LIMIT %s;
        """,
        "params": lambda: (
            random.randint(60, 180),
            random.choice(["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]),
            random.randint(5, 50)
        )
    },
    {
        "name": "customer_payment_query", 
        "sql": """
            SELECT c.first_name, c.last_name, COUNT(p.payment_id)
            FROM customer c
            JOIN payment p ON c.customer_id = p.customer_id
            WHERE p.payment_date > %s
            GROUP BY c.customer_id
            HAVING COUNT(p.payment_id) > %s
            LIMIT %s;
        """,
        "params": lambda: (
            datetime(2005, 1, 1) + timedelta(days=random.randint(0, 1500)),
            random.randint(3, 30),
            random.randint(5, 100)
        )
    },
    {
        "name": "film_rental_aggregate",
        "sql": """
            SELECT f.title, c.name, COUNT(r.rental_id) as rental_count
            FROM film f
            JOIN film_category fc ON f.film_id = fc.film_id
            JOIN category c ON fc.category_id = c.category_id
            JOIN inventory i ON f.film_id = i.film_id
            JOIN rental r ON i.inventory_id = r.inventory_id
            WHERE f.length > %s AND c.name LIKE %s
            GROUP BY f.film_id, c.name
            HAVING COUNT(r.rental_id) > %s
            ORDER BY rental_count DESC LIMIT %s;
        """,
        "params": lambda: (
            random.randint(80, 160),
            f"%{random.choice(['Action', 'Comedy', 'Drama', 'Family'])}%",
            random.randint(2, 25),
            random.randint(5, 30)
        )
    },
    {
        "name": "actor_film_query",
        "sql": """
            SELECT a.first_name, a.last_name, COUNT(fa.film_id) as film_count
            FROM actor a
            JOIN film_actor fa ON a.actor_id = fa.actor_id
            JOIN film f ON fa.film_id = f.film_id
            WHERE f.rating = %s
            GROUP BY a.actor_id
            HAVING COUNT(fa.film_id) > %s
            ORDER BY film_count DESC LIMIT %s;
        """,
        "params": lambda: (
            random.choice(['PG', 'PG-13', 'R', 'G']),
            random.randint(10, 40),
            random.randint(10, 50)
        )
    }
]

# IMPROVED FLATTEN PLAN
def flatten_plan(node, parent_type="ROOT", parent_rows=None, rows=None):
    if rows is None:
        rows = []

    plan_rows = node.get("Plan Rows", 1)
    actual_rows = node.get("Actual Rows", 0)

    selectivity = 1.0
    if parent_rows and parent_rows > 0:
        selectivity = min(plan_rows / parent_rows, 1.0)
    
    startup_cost = node.get("Startup Cost", 0)
    total_cost = node.get("Total Cost", 0)
    cost_ratio = total_cost / (startup_cost or 1)

    entry = {
        "node_type": node.get("Node Type", "Unknown"),
        "parent_node": parent_type,
        "join_type": node.get("Join Type", "None"),
        "relation_name": node.get("Relation Name", "None"),
        "alias": node.get("Alias", "None"),
        "plan_rows": float(plan_rows),
        "actual_rows": float(actual_rows),
        "startup_cost": float(startup_cost),
        "total_cost": float(total_cost),
        "plan_width": float(node.get("Plan Width", 0)),
        "selectivity": selectivity,
        "cost_ratio": cost_ratio,
        "log_plan_rows": round(math.log10(max(plan_rows, 1)), 6),
        "filter": normalize_filter(node.get("Filter", "")),
        "hash_cond": normalize_cond(node.get("Hash Cond", "")),
        "index_cond": normalize_cond(node.get("Index Cond", "")),
        # REMOVED: log_actual_rows to prevent data leakage
    }
    rows.append(entry)

    for sub in node.get("Plans", []):
        flatten_plan(sub, node.get("Node Type", "Unknown"), plan_rows, rows)

    return rows

def main():
    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    all_rows = []
    seen_queries = set()
    query_count = 0
    template_counts = {t["name"]: 0 for t in TEMPLATES}

    print(f"Generating {TARGET_NODES:,} training nodes...")
    
    while len(all_rows) < TARGET_NODES and query_count < MAX_QUERIES:
        query_count += 1
        template = random.choice(TEMPLATES)
        
        try:
            params = template["params"]()
            sql = template["sql"]
            
            # Create query signature for deduplication
            query_sig = (template["name"], str(params))
            if query_sig in seen_queries:
                continue
            seen_queries.add(query_sig)

            # Execute with EXPLAIN ANALYZE
            cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {sql}", params)
            result = cur.fetchone()
            
            if result and result[0]:
                plan = result[0][0]["Plan"]
                nodes = flatten_plan(plan)
                all_rows.extend(nodes)
                template_counts[template["name"]] += 1
                
                if query_count % 100 == 0:
                    print(f"Query {query_count:4d} | +{len(nodes):3d} nodes | Total: {len(all_rows):6d} | "
                          f"Template: {template['name']}")

                # Save checkpoint
                if query_count % 500 == 0:
                    df_checkpoint = pd.DataFrame(all_rows)
                    df_checkpoint.to_csv(OUTPUT_FILE, index=False)
                    print(f"Checkpoint saved. Template distribution: {template_counts}")

        except Exception as e:
            conn.rollback()
            if "canceling statement due to statement timeout" not in str(e):
                print(f"Query {query_count} failed: {e}")
            continue

    # Final save
    df_final = pd.DataFrame(all_rows)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Total queries: {query_count}")
    print(f"Total nodes: {len(all_rows):,}")
    print(f"Template distribution: {template_counts}")
    print(f"Output: {OUTPUT_FILE}")
    
    # Data quality check
    print(f"\n=== DATA QUALITY CHECK ===")
    print(f"Node types: {df_final['node_type'].value_counts().to_dict()}")
    print(f"Unique filters: {df_final['filter'].nunique()}")
    print(f"Unique aliases: {df_final['alias'].nunique()}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()