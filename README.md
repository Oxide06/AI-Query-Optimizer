# AI Query Optimizer

AI Query Optimizer is a backend-focused system that examines SQL queries, identifies performance bottlenecks, and produces optimized query alternatives using **machine learning**, **NLP-based query parsing**, and **PostgreSQL execution plan analysis**.

The goal of this project is to reduce query execution time and resource usage while respecting **real-world database constraints**, such as avoiding schema modifications.

---

## Motivation

Database performance issues are often caused by inefficient SQL queries rather than hardware limitations.  
Optimizing such queries usually requires manual inspection of execution plans and deep database expertise.

This project explores how **machine learning models trained on execution plans** can assist in automating query optimization and performance validation.

---

## What This System Does

- Analyzes SQL queries to detect inefficiencies  
- Extracts features from PostgreSQL execution plans  
- Uses a trained ML model to identify efficient query patterns  
- Benchmarks original and optimized queries using actual runtime metrics  
- Suggests improvements without altering database schemas  

---

## Core Concepts Used

- SQL query analysis  
- PostgreSQL query planner and execution plans  
- Natural Language Processing for query parsing  
- Machine Learning–based pattern recognition  
- Performance benchmarking and validation  

---

## Technology Stack

- **Language:** Python  
- **Database:** PostgreSQL  
- **Machine Learning Model:** XGBoost  
- **Performance Analysis:** `EXPLAIN ANALYZE`  
- **Libraries:** Pandas, NumPy  

---

## System Workflow

1. **Query Input**  
   The user provides an SQL query for analysis.

2. **Query Parsing**  
   NLP-based logic extracts structural components of the query.

3. **Execution Plan Generation**  
   PostgreSQL produces detailed execution plans using `EXPLAIN ANALYZE`.

4. **ML-Based Evaluation**  
   A trained XGBoost model analyzes execution plan features to identify performance-efficient patterns.

5. **Optimization & Validation**  
   The optimized query is benchmarked against the original query to compare execution time and CPU usage.

---

## Results & Observations

- Reduced execution time for inefficient queries  
- Identified common performance bottlenecks using execution plan features  
- Demonstrated optimization without requiring schema changes  
- Highlighted the practicality of ML-assisted database tuning  

---

## Setup & Usage

### Requirements
- Python 3.x  
- PostgreSQL  
- Required Python packages listed in `requirements.txt`

### Installation
```bash
git clone https://github.com/Oxide06/AI-Query-Optimizer.git
cd AI-Query-Optimizer
pip install -r requirements.txt


