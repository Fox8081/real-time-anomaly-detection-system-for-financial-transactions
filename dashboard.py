import time
import hashlib
from datetime import datetime

import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
import streamlit as st

# ===========================
# App Config and Styling
# ===========================
st.set_page_config(page_title="Govt Scheme Anomaly Monitor", layout="wide")

# Optional: Add minimal CSS for spacing
st.markdown(
    """
    <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# Authentication
def hash_password(password: str) -> str:
    import hashlib as _hashlib
    return _hashlib.sha256(password.encode()).hexdigest()

USERS = {
    "auditor": hash_password("aegis123"),
    "auditor2": hash_password("password2"),
}

def login():
    st.title("Auditor Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Use st.rerun on 1.48.x
    if st.button("Login", type="primary", use_container_width=True):
        hashed = hash_password(password)
        if username in USERS and USERS[username] == hashed:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid username or password")

def logout_button():
    with st.sidebar:
        st.write(f"Logged in as: {st.session_state.get('username', '')}")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()


# Data Loading

@st.cache_data(ttl=5)
def load_data():
    conn = sqlite3.connect("transactions.db", timeout=30, check_same_thread=False)
    try:
        df = pd.read_sql_query("SELECT * FROM transactions", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return df

    # Parse timestamps safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    return df

# Auto Refresh (non-blocking)

def auto_refresh(seconds_default=10):
    st.sidebar.markdown("---")
    refresh_enabled = st.sidebar.checkbox("Enable auto-refresh", value=True)
    interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, seconds_default)
    if refresh_enabled:
        st.sidebar.info(f"Dashboard will refresh every {interval} seconds.")
        # Lightweight manual refresh button (no blocking sleep)
        if st.sidebar.button("Refresh now"):
            st.rerun()
        # Passive timed refresh using query param change (fallback without extra deps)
        # Simulate a soft refresh by updating a dummy key based on time bucket
        bucket = int(time.time() // interval)
        st.session_state["__refresh_bucket"] = bucket


# KPI Cards

def display_kpi_cards(df: pd.DataFrame):
    total_txns = len(df)
    if total_txns == 0:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", "0")
        col2.metric("Anomalies", "0")
        col3.metric("Anomaly %", "0.00%")
        col4.metric("Most-Affected Scheme", "N/A")
        return

    anomalies_df = df[df["status"] == "ANOMALY"] if "status" in df.columns else df.iloc[0:0]
    anomalies_count = len(anomalies_df)
    anomaly_pct = (anomalies_count / total_txns * 100.0) if total_txns else 0.0
    most_affected_scheme = (
        anomalies_df["scheme_type"].mode().iloc[0]
        if not anomalies_df.empty and "scheme_type" in anomalies_df.columns
        else "N/A"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total_txns:,}")
    col2.metric("Anomalies", f"{anomalies_count:,}")
    col3.metric("Anomaly %", f"{anomaly_pct:.2f}%")
    col4.metric("Most-Affected Scheme", most_affected_scheme)


# Filters Panel

def filters_panel(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    if df.empty:
        st.sidebar.info("No data available yet.")
        return df

    # Date Range
    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        st.sidebar.info("Timestamps are not available.")
        date_range = None
    else:
        date_range = st.sidebar.date_input(
            "Date range", [min_ts.date(), max_ts.date()]
        )

    # Scheme filter
    if "scheme_type" in df.columns:
        scheme_list = ["All"] + sorted(df["scheme_type"].dropna().unique().tolist())
    else:
        scheme_list = ["All"]
    selected_scheme = st.sidebar.selectbox("Scheme Type", scheme_list)

    # Anomaly Status filter
    anomaly_status = st.sidebar.selectbox("Anomaly Status", ["All", "ANOMALY", "NORMAL"])

    # Amount range
    if "amount" in df.columns and not df["amount"].isna().all():
        min_amount = int(df["amount"].min())
        max_amount = int(df["amount"].max())
        if max_amount < min_amount:
            min_amount, max_amount = 0, 0
    else:
        min_amount, max_amount = 0, 0
    amount_range = st.sidebar.slider(
        "Amount Range",
        min_amount,
        max_amount if max_amount >= min_amount else min_amount,
        (min_amount, max_amount) if max_amount >= min_amount else (0, 0),
    )

    # Search
    search_text = st.sidebar.text_input("Search Transaction ID or Beneficiary ID")

    # Apply filters
    filtered = df.copy()

    if isinstance(date_range, list) and len(date_range) == 2:
        start_d, end_d = date_range
        if not pd.isna(start_d) and not pd.isna(end_d):
            filtered = filtered[
                (filtered["timestamp"].dt.date >= start_d)
                & (filtered["timestamp"].dt.date <= end_d)
            ]

    if max_amount >= min_amount:
        filtered = filtered[
            (filtered["amount"] >= amount_range[0])
            & (filtered["amount"] <= amount_range[1])
        ]

    if selected_scheme != "All" and "scheme_type" in filtered.columns:
        filtered = filtered[filtered["scheme_type"] == selected_scheme]

    if anomaly_status != "All" and "status" in filtered.columns:
        filtered = filtered[filtered["status"] == anomaly_status]

    if search_text:
        txid_match = (
            filtered["transaction_id"].astype(str).str.contains(search_text, na=False)
            if "transaction_id" in filtered.columns
            else pd.Series([False] * len(filtered), index=filtered.index)
        )
        ben_match = (
            filtered["beneficiary_id"].astype(str).str.contains(search_text, na=False)
            if "beneficiary_id" in filtered.columns
            else pd.Series([False] * len(filtered), index=filtered.index)
        )
        filtered = filtered[txid_match | ben_match]

    return filtered


# Visualizations

def plot_anomaly_trends(df):
    st.subheader("Daily Transaction Trends")
    if df.empty or "status" not in df.columns or "timestamp" not in df.columns:
        st.info("No data to plot trends yet.")
        return

    # Group by date and status
    daily = (
        df.groupby([df["timestamp"].dt.date, "status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )
    daily = daily.rename(columns={daily.columns[0]: "date"})

    # Only keep dates in actual transaction range
    earliest = df["timestamp"].dt.date.min()
    latest = df["timestamp"].dt.date.max()
    daily = daily[(daily["date"] >= earliest) & (daily["date"] <= latest)]

    ycols = [c for c in daily.columns if c not in ["date"]]
    if not ycols:
        st.info("No status categories to plot.")
        return

    fig = px.area(
        daily,
        x="date",
        y=ycols,
        title="Daily Transaction Trends",
        labels={"value": "Count", "date": "Date", "variable": "Status"},
    )
    fig.update_xaxes(range=[earliest, latest])
    st.plotly_chart(fig, use_container_width=True)


def plot_geo_anomalies(df):
    st.subheader("Anomaly Locations")
    if df.empty or "status" not in df.columns:
        st.info("No anomaly locations available.")
        return

    anomalies = df[df["status"] == "ANOMALY"].copy()
    if anomalies.empty:
        st.info("No anomaly locations available.")
        return

    in_india_only = st.sidebar.checkbox("Show only India locations", value=True, help="Hide points detected outside India bounds.")
    if in_india_only and {"latitude","longitude"}.issubset(anomalies.columns):
        anomalies = anomalies[
            (anomalies["latitude"].between(8.0, 37.1, inclusive="both"))
            & (anomalies["longitude"].between(68.0, 97.4, inclusive="both"))
        ]

    if anomalies.empty or not {"latitude","longitude"}.issubset(anomalies.columns):
        st.info("No anomaly locations available.")
        return

    color_col = "fraud_type" if "fraud_type" in anomalies.columns else None
    fig = px.scatter_map(
        anomalies,
        lat="latitude",
        lon="longitude",
        color=color_col,
        zoom=4,
        hover_data=[c for c in ["transaction_id","amount","timestamp","scheme_type","fraud_type","anomaly_score"] if c in anomalies.columns],
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)


def plot_anomaly_heatmap(df: pd.DataFrame):
    st.subheader("Anomaly Occurrence Heatmap")
    if df.empty or "status" not in df.columns or "timestamp" not in df.columns:
        st.info("No anomalies for heatmap.")
        return

    anomalies = df[df["status"] == "ANOMALY"].copy()
    if anomalies.empty:
        st.info("No anomalies for heatmap.")
        return

    anomalies["hour"] = anomalies["timestamp"].dt.hour
    anomalies["weekday"] = anomalies["timestamp"].dt.dayofweek
    pivot = anomalies.pivot_table(index="weekday", columns="hour", aggfunc="size", fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Reds", ax=ax)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Week (0=Monday)")
    ax.set_title("Anomaly Occurrence Heatmap")
    st.pyplot(fig)

# ===========================
# Download
# ===========================
def download_button(df: pd.DataFrame):
    if df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ===========================
# Main
# ===========================
def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""

    # Login gate
    if not st.session_state["logged_in"]:
        login()
        st.stop()

    # Sidebar controls
    logout_button()
    auto_refresh(10)

    st.title("Real-Time Government Scheme Anomaly Dashboard")

    # Data
    df = load_data()

    # Filters
    filtered_df = filters_panel(df)

    # KPIs
    display_kpi_cards(filtered_df)

    # Visuals
    plot_anomaly_trends(filtered_df)
    plot_geo_anomalies(filtered_df)
    plot_anomaly_heatmap(filtered_df)

    # Table
    st.subheader("Filtered Transactions")

    show_all = st.checkbox("Show all rows", value=False, help="Uncheck to see the top 100 most recent.")
    view_df = filtered_df.sort_values(by="timestamp", ascending=False) if "timestamp" in filtered_df.columns else filtered_df

    if not show_all:
      view_df = view_df.head(100)

    if not view_df.empty:
        cols = [c for c in [
            "transaction_id","beneficiary_id","amount","timestamp","scheme_type",
            "status","fraud_type","anomaly_score","latitude","longitude","anomaly_alert"
        ] if c in view_df.columns]
        st.dataframe(view_df[cols], use_container_width=True)
        download_button(view_df[cols])
    else:
        st.info("No transactions to display for the current filters.")


if __name__ == "__main__":
    main()
