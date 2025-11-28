# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import psycopg2
import os
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AI Query Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .improvement-positive {
        color: #00aa00;
        font-weight: bold;
    }
    .improvement-negative {
        color: #ff4b4b;
        font-weight: bold;
    }
    .winner-ai {
        background-color: #d4edda;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        color: #155724;
    }
    .winner-pg {
        background-color: #f8d7da;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL_DIR = "models_improved"
DATA_FILE = "data/processed/lce_training_data_fixed.csv"

# Normalization functions
def normalize_filter(s):
    if not s or s == "None": 
        return "none"
    s = re.sub(r'[a-zA-Z_]+\.', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    s = s.replace('(', '').replace(')', '')
    s = s.lower()
    s = re.sub(r'\b\d+\b', '#', s)
    return s

def normalize_cond(s):
    if not s or s == "None": 
        return "none"
    cols = re.findall(r'([a-zA-Z_]+)\.([a-zA-Z_]+)', s)
    if len(cols) == 2:
        return f"{cols[0][1]}={cols[1][1]}".lower()
    return "none"

def flatten_plan(node, parent_type="ROOT", parent_rows=None, out=None):
    if out is None: out = []
    plan_rows = node.get("Plan Rows", 1)
    selectivity = min(plan_rows / parent_rows, 1.0) if parent_rows and parent_rows > 0 else 1.0
    startup_cost = node.get("Startup Cost", 0)
    total_cost = node.get("Total Cost", 0)
    cost_ratio = total_cost / (startup_cost or 1)

    row = {
        "node_type": node.get("Node Type", "Unknown"),
        "parent_node": parent_type,
        "join_type": node.get("Join Type", "None"),
        "relation_name": node.get("Relation Name", "None"),
        "alias": node.get("Alias", "None"),
        "plan_rows": float(plan_rows),
        "startup_cost": float(startup_cost),
        "total_cost": float(total_cost),
        "plan_width": float(node.get("Plan Width", 0)),
        "selectivity": selectivity,
        "cost_ratio": cost_ratio,
        "log_plan_rows": np.log10(max(plan_rows, 1)),
        "filter": normalize_filter(node.get("Filter", "")),
        "hash_cond": normalize_cond(node.get("Hash Cond", "")),
        "index_cond": normalize_cond(node.get("Index Cond", "")),
    }
    out.append(row)
    for sub in node.get("Plans", []):
        flatten_plan(sub, node.get("Node Type", "Unknown"), plan_rows, out)
    return out

def extract_actual_rows(node, out=None):
    if out is None: out = []
    out.append(node.get("Actual Rows", 0))
    for sub in node.get("Plans", []):
        extract_actual_rows(sub, out)
    return out

def predict_plan(plan_json):
    """Predict AI estimates for a given plan"""
    try:
        # Load model and preprocessing objects
        model = joblib.load(f"{MODEL_DIR}/xgb_lce.pkl")
        scaler = joblib.load(f"{MODEL_DIR}/input_scaler.pkl")
        
        encoders = {}
        for col in ["node_type", "parent_node", "join_type", "relation_name", "alias",
                    "filter", "hash_cond", "index_cond"]:
            p = f"{MODEL_DIR}/le_{col}.pkl"
            if os.path.exists(p):
                encoders[col] = joblib.load(p)

        # Extract data from plan
        root_node = plan_json[0]["Plan"]
        actual_rows = extract_actual_rows(root_node)
        nodes = flatten_plan(root_node)
        df = pd.DataFrame(nodes)

        # Encode categorical features
        for col, le in encoders.items():
            if col in df.columns:
                df[col] = df[col].fillna("missing").astype(str)
                df[col] = df[col].replace("None", "none")
                # Handle unseen categories
                df[col] = df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Prepare features
        num_cols = ["startup_cost", "total_cost", "plan_width", "selectivity", "cost_ratio"]
        cat_cols = ["node_type", "parent_node", "join_type", "relation_name", "alias",
                   "filter", "hash_cond", "index_cond"]
        
        num_cols = [col for col in num_cols if col in df.columns]
        cat_cols = [col for col in cat_cols if col in df.columns]

        # Scale and predict
        X_num = scaler.transform(df[num_cols])
        X_cat = df[cat_cols].values
        X = np.hstack([X_cat, X_num])

        preds_log = model.predict(X)
        ai_rows = 10 ** preds_log
        ai_rows = np.clip(ai_rows, 0.1, 1000000)

        # Create results dataframe
        results = []
        for i in range(len(ai_rows)):
            pg_estimate = df["plan_rows"].iloc[i] if "plan_rows" in df.columns else 0
            ai_estimate = ai_rows[i]
            actual = actual_rows[i]
            
            if actual > 0:
                q_pg = max(pg_estimate/actual, actual/pg_estimate)
                q_ai = max(ai_estimate/actual, actual/ai_estimate)
                winner = "AI" if q_ai < q_pg else "PostgreSQL" if q_ai > q_pg else "Tie"
            else:
                q_pg = 1.0
                q_ai = 1.0
                winner = "N/A"
            
            results.append({
                "node_id": i,
                "node_type": df["node_type"].iloc[i] if i < len(df) else "Unknown",
                "pg_estimate": pg_estimate,
                "ai_estimate": ai_estimate,
                "actual_rows": actual,
                "q_error_pg": q_pg,
                "q_error_ai": q_ai,
                "winner": winner
            })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None

def generate_sample_queries():
    """Return sample queries for demonstration"""
    return {
        "Simple Join": """
            SELECT f.title, c.name 
            FROM film f 
            JOIN film_category fc ON f.film_id = fc.film_id 
            JOIN category c ON fc.category_id = c.category_id 
            WHERE f.length > 120 AND c.name = 'Action' 
            LIMIT 10;
        """,
        "Customer Payments": """
            SELECT c.first_name, c.last_name, COUNT(p.payment_id) as payment_count
            FROM customer c
            JOIN payment p ON c.customer_id = p.customer_id
            WHERE p.payment_date > '2005-01-01'
            GROUP BY c.customer_id
            HAVING COUNT(p.payment_id) > 10
            LIMIT 20;
        """,
        "Film Rentals": """
            SELECT f.title, c.name, COUNT(r.rental_id) as rental_count
            FROM film f
            JOIN film_category fc ON f.film_id = fc.film_id
            JOIN category c ON fc.category_id = c.category_id
            JOIN inventory i ON f.film_id = i.film_id
            JOIN rental r ON i.inventory_id = r.inventory_id
            WHERE f.length > 90
            GROUP BY f.film_id, c.name
            HAVING COUNT(r.rental_id) > 5
            ORDER BY rental_count DESC LIMIT 15;
        """,
        "Actor Performance": """
            SELECT a.first_name, a.last_name, COUNT(fa.film_id) as film_count
            FROM actor a
            JOIN film_actor fa ON a.actor_id = fa.actor_id
            GROUP BY a.actor_id
            HAVING COUNT(fa.film_id) > 20
            ORDER BY film_count DESC
            LIMIT 15;
        """
    }

def load_model_metrics():
    """Load actual model metrics dynamically"""
    try:
        # Load model
        model = joblib.load(f"{MODEL_DIR}/xgb_lce.pkl")
        
        # Load training data for metrics
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            df = df[df['actual_rows'] > 0]
            
            # Calculate actual performance metrics
            y = np.log10(df["actual_rows"])
            
            # For demo, we'll use the model's feature set
            feature_columns = [col for col in ['node_type', 'parent_node', 'join_type', 
                                             'relation_name', 'alias', 'filter', 'hash_cond', 
                                             'index_cond', 'startup_cost', 'total_cost', 
                                             'plan_width', 'selectivity', 'cost_ratio'] 
                             if col in df.columns]
            
            X = df[feature_columns].copy()
            
            # Encode categorical features
            encoders = {}
            cat_cols = ['node_type', 'parent_node', 'join_type', 'relation_name', 'alias', 
                       'filter', 'hash_cond', 'index_cond']
            cat_cols = [c for c in cat_cols if c in X.columns]
            
            for col in cat_cols:
                le = joblib.load(f"{MODEL_DIR}/le_{col}.pkl")
                X[col] = X[col].fillna("missing").astype(str)
                X[col] = X[col].replace("None", "none")
                X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
            
            # Scale numerical features
            scaler = joblib.load(f"{MODEL_DIR}/input_scaler.pkl")
            num_cols = ['startup_cost', 'total_cost', 'plan_width', 'selectivity', 'cost_ratio']
            num_cols = [c for c in num_cols if c in X.columns]
            X[num_cols] = scaler.transform(X[num_cols])
            
            # Make predictions
            preds_log = model.predict(X)
            preds = 10 ** preds_log
            actuals = 10 ** y
            
            # Calculate Q-errors
            q_errors = np.maximum(preds / actuals, actuals / preds)
            
            metrics = {
                'median_q_error': np.median(q_errors),
                'mean_q_error': np.mean(q_errors),
                'p90_q_error': np.percentile(q_errors, 90),
                'training_samples': len(df),
                'node_types': df['node_type'].nunique() if 'node_type' in df.columns else 'N/A',
                'features': len(feature_columns)
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                metrics['feature_importance'] = importance_df
            
            return metrics
        else:
            st.warning("Training data file not found. Using default metrics.")
            return None
            
    except Exception as e:
        st.warning(f"Could not load dynamic metrics: {e}")
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">AI Query Optimizer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Upload Plan", "Live Query", "Model Info", "Sample Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This app uses AI to improve PostgreSQL query plan estimates. "
        "Upload an EXPLAIN ANALYZE plan or run a live query to see the AI in action!"
    )

    if app_mode == "Upload Plan":
        upload_plan_mode()
    elif app_mode == "Live Query":
        live_query_mode()
    elif app_mode == "Model Info":
        model_info_mode()
    elif app_mode == "Sample Analysis":
        sample_analysis_mode()

def upload_plan_mode():
    st.header("Upload EXPLAIN ANALYZE Plan")
    
    uploaded_file = st.file_uploader("Upload JSON plan file", type=['json'])
    
    if uploaded_file is not None:
        try:
            plan_json = json.load(uploaded_file)
            st.success("Plan file loaded successfully!")
            
            # Show plan structure
            with st.expander("View Plan Structure"):
                st.json(plan_json)
            
            # Predict
            if st.button("Analyze with AI", type="primary"):
                with st.spinner("Analyzing plan with AI..."):
                    results_df = predict_plan(plan_json)
                
                if results_df is not None:
                    display_results(results_df)
                    
        except Exception as e:
            st.error(f"Error loading plan: {str(e)}")

def live_query_mode():
    st.header("Live Query Analysis")
    
    # Database connection settings
    st.subheader("Database Connection")
    col1, col2 = st.columns(2)
    
    with col1:
        host = st.text_input("Host", "localhost")
        dbname = st.text_input("Database", "DVD_rental")
    
    with col2:
        user = st.text_input("User", "postgres")
        password = st.text_input("Password", type="password")
    
    # Query input
    st.subheader("Query Input")
    sample_queries = generate_sample_queries()
    
    query_option = st.selectbox(
        "Choose a sample query or write your own:",
        ["Custom Query"] + list(sample_queries.keys())
    )
    
    if query_option == "Custom Query":
        query = st.text_area("Enter your SQL query:", height=150, 
                           placeholder="SELECT * FROM film WHERE length > 120;")
    else:
        query = sample_queries[query_option]
        st.code(query, language="sql")
    
    if st.button("Execute & Analyze", type="primary") and query:
        if not all([host, dbname, user, password]):
            st.error("Please fill all database connection fields")
            return
            
        try:
            # Connect to database and execute EXPLAIN ANALYZE
            with st.spinner("Executing query and generating plan..."):
                conn = psycopg2.connect(
                    host=host,
                    dbname=dbname,
                    user=user,
                    password=password
                )
                cur = conn.cursor()
                
                # Execute EXPLAIN ANALYZE
                cur.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
                result = cur.fetchone()
                
                if result and result[0]:
                    plan_json = result[0]
                    
                    # Predict with AI
                    results_df = predict_plan(plan_json)
                    
                    if results_df is not None:
                        display_results(results_df)
                
                cur.close()
                conn.close()
                
        except Exception as e:
            st.error(f"Database error: {str(e)}")

def model_info_mode():
    st.header("Model Information")
    
    # Load dynamic metrics
    metrics = load_model_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        if metrics:
            st.metric("Median Q-Error", f"{metrics['median_q_error']:.3f}")
            st.metric("Mean Q-Error", f"{metrics['mean_q_error']:.3f}")
            st.metric("90th Percentile Q-Error", f"{metrics['p90_q_error']:.3f}")
        else:
            # Fallback to your actual training results
            st.metric("Median Q-Error", "1.028")
            st.metric("Mean Q-Error", "1.266")
            st.metric("90th Percentile Q-Error", "1.596")
    
    with col2:
        st.subheader("Training Data")
        if metrics:
            st.metric("Training Samples", f"{metrics['training_samples']:,}")
            st.metric("Node Types", metrics['node_types'])
            st.metric("Features", metrics['features'])
        else:
            st.metric("Training Samples", "12,818")
            st.metric("Node Types", "9")
            st.metric("Features", "13")
    
    st.subheader("Feature Importance")
    
    try:
        # Load actual feature importance from model
        model = joblib.load(f"{MODEL_DIR}/xgb_lce.pkl")
        
        # Get feature names
        feature_columns = ['node_type', 'parent_node', 'join_type', 'relation_name', 'alias',
                         'filter', 'hash_cond', 'index_cond', 'startup_cost', 'total_cost',
                         'plan_width', 'selectivity', 'cost_ratio']
        
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Top 10 Feature Importances (Actual Model)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show importance table
            st.dataframe(importance_df, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
            
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
        # Fallback to known values from your training
        importance_data = {
            'Feature': ['parent_node', 'plan_width', 'total_cost', 'join_type', 'alias', 
                       'relation_name', 'selectivity', 'cost_ratio', 'hash_cond', 'startup_cost'],
            'Importance': [0.275, 0.256, 0.129, 0.082, 0.066, 0.064, 0.056, 0.027, 0.017, 0.012]
        }
        importance_df = pd.DataFrame(importance_data)
        
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title="Top 10 Feature Importances (From Training)")
        st.plotly_chart(fig, use_container_width=True)

def sample_analysis_mode():
    st.header("Sample Analysis")
    
    # Use actual recent analysis if available, otherwise show informative message
    st.info("""
    **Run a live query or upload a plan to see dynamic analysis here!**
    
    This section will display actual results from your recent queries including:
    - Real Q-error comparisons between AI and PostgreSQL
    - Dynamic performance metrics
    - Actual plan node analysis
    """)
    
    # Option to load a sample plan for demonstration
    sample_plan_path = "sample_plan.json"
    if os.path.exists(sample_plan_path):
        if st.button("Load Sample Analysis"):
            with open(sample_plan_path) as f:
                plan_json = json.load(f)
            results_df = predict_plan(plan_json)
            if results_df is not None:
                display_results(results_df)
    else:
        st.warning("Sample plan file not found. Run a query first or upload a plan.")

def display_results(results_df):
    """Display prediction results in an interactive way"""
    st.header("Analysis Results")
    
    # Calculate summary metrics
    valid_nodes = results_df[results_df['actual_rows'] > 0]
    if len(valid_nodes) == 0:
        st.warning("No nodes with actual rows data available.")
        return
    
    avg_pg_error = valid_nodes['q_error_pg'].mean()
    avg_ai_error = valid_nodes['q_error_ai'].mean()
    improvement = ((avg_pg_error - avg_ai_error) / avg_pg_error * 100) if avg_pg_error > 0 else 0
    ai_wins = (valid_nodes['winner'] == 'AI').sum()
    pg_wins = (valid_nodes['winner'] == 'PostgreSQL').sum()
    ties = (valid_nodes['winner'] == 'Tie').sum()
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg PostgreSQL Error", f"{avg_pg_error:.2f}")
    
    with col2:
        st.metric("Avg AI Error", f"{avg_ai_error:.2f}")
    
    with col3:
        improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
        st.markdown(f'<div class="metric-card">Improvement<br><span class="{improvement_class}">{improvement:+.1f}%</span></div>', 
                   unsafe_allow_html=True)
    
    with col4:
        st.metric("AI Wins vs PostgreSQL", f"{ai_wins}-{pg_wins}")
    
    # Detailed results table with styling
    st.subheader("Detailed Node Analysis")
    
    # Format the dataframe for display with better styling
    display_df = results_df.copy()
    display_df['pg_estimate'] = display_df['pg_estimate'].round(1)
    display_df['ai_estimate'] = display_df['ai_estimate'].round(1)
    display_df['q_error_pg'] = display_df['q_error_pg'].round(3)
    display_df['q_error_ai'] = display_df['q_error_ai'].round(3)
    
    # Add winner styling
    def style_winner(val):
        if val == 'AI':
            return 'color: #155724; background-color: #d4edda;'
        elif val == 'PostgreSQL':
            return 'color: #721c24; background-color: #f8d7da;'
        else:
            return ''
    
    styled_df = display_df.style.applymap(style_winner, subset=['winner'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Q-Error comparison chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results_df['node_id'], y=results_df['q_error_pg'], 
                               name='PostgreSQL', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=results_df['node_id'], y=results_df['q_error_ai'], 
                               name='AI', line=dict(color='green')))
        fig.update_layout(title="Q-Error Comparison by Node",
                         xaxis_title="Node ID",
                         yaxis_title="Q-Error",
                         showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Winner distribution
        winner_counts = results_df['winner'].value_counts()
        colors = ['#28a745' if idx == 'AI' else '#dc3545' if idx == 'PostgreSQL' else '#6c757d' 
                 for idx in winner_counts.index]
        
        fig = px.pie(values=winner_counts.values, 
                    names=winner_counts.index,
                    title="Estimation Accuracy by Winner",
                    color_discrete_sequence=colors)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance by node type
    if 'node_type' in results_df.columns:
        st.subheader("Performance by Node Type")
        node_performance = results_df.groupby('node_type').agg({
            'q_error_pg': 'mean',
            'q_error_ai': 'mean',
            'winner': lambda x: (x == 'AI').sum() / len(x) * 100  # AI win percentage
        }).round(3)
        
        node_performance.columns = ['Avg PG Error', 'Avg AI Error', 'AI Win %']
        st.dataframe(node_performance, use_container_width=True)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="ai_optimizer_results.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()