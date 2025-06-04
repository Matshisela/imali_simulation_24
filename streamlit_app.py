# -*- coding: utf-8 -*-
"""streamlit_app.py

Transaction Analytics Dashboard for C-Level Executives
"""

import streamlit as st
import pandas as pd
from datetime import datetime

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Required packages not found. Please ensure you have plotly installed.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Transaction Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=600)
def load_data():
    url = "https://raw.githubusercontent.com/Matshisela/imali_simulation_24/main/data/transaction_data_6_months.csv"
    try:
        df = pd.read_csv(url, parse_dates=['timestamp'])
        # Add derived datetime features
        df['date'] = df['timestamp'].dt.date
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month_name()
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
try:
    date_range = st.sidebar.date_input(
        "Date range",
        value=[df['timestamp'].min().date(), df['timestamp'].max().date()],
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.max.time())
        filtered_df = df[(df['timestamp'] >= start_datetime) & (df['timestamp'] <= end_datetime)]
    else:
        filtered_df = df

    services = st.sidebar.multiselect(
        "Services",
        options=df['service'].unique(),
        default=df['service'].unique()
    )

    payment_methods = st.sidebar.multiselect(
        "Payment Methods",
        options=df['payment_method'].unique(),
        default=df['payment_method'].unique()
    )

    statuses = st.sidebar.multiselect(
        "Transaction Status",
        options=df['status'].unique(),
        default=df['status'].unique()
    )

    # Apply filters
    filtered_df = filtered_df[
        (filtered_df['service'].isin(services)) &
        (filtered_df['payment_method'].isin(payment_methods)) &
        (filtered_df['status'].isin(statuses))
    ]
except Exception as e:
    st.error(f"Error applying filters: {str(e)}")
    filtered_df = df

# Dashboard UI
st.title("📊 Transaction Analytics Dashboard")
st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
        color: #2a9fd6;
    }
    .metric-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin-bottom: 20px;
    }
    </style>
    <p class="big-font">Executive dashboard for monitoring transaction performance</p>
    """, unsafe_allow_html=True)

# KPI Metrics with improved styling
st.subheader("Key Performance Indicators")
cols = st.columns(4)

with cols[0]:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    total_transactions = filtered_df.shape[0]
    st.metric("Total Transactions", f"{total_transactions:,}")
    st.markdown('</div>', unsafe_allow_html=True)

with cols[1]:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    total_amount = filtered_df['amount'].sum()
    st.metric("Total Amount", f"${total_amount:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with cols[2]:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    success_rate = (filtered_df[filtered_df['status'] == 'Success'].shape[0] / filtered_df.shape[0]) * 100
    st.metric("Success Rate", f"{success_rate:.1f}%")
    st.markdown('</div>', unsafe_allow_html=True)

with cols[3]:
    st.markdown('<div class="metric-box">', unsafe_allow_html=True)
    unique_users = filtered_df['user_id'].nunique()
    st.metric("Unique Users", f"{unique_users:,}")
    st.markdown('</div>', unsafe_allow_html=True)

# Visualization functions with error handling
def create_daily_transactions_chart(data):
    try:
        daily_trans = data.groupby('date')['transaction_id'].count().reset_index()
        fig = px.line(
            daily_trans,
            x='date',
            y='transaction_id',
            title='Daily Transaction Volume',
            labels={'date': 'Date', 'transaction_id': 'Transactions'}
        )
        fig.update_layout(height=400)
        return fig
    except Exception as e:
        st.error(f"Error creating daily transactions chart: {str(e)}")
        return None

# Add more visualization functions similarly...

# Layout for charts
try:
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_daily_transactions_chart(filtered_df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        try:
            fig = px.box(
                filtered_df,
                y='amount',
                title='Transaction Amount Distribution'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating box plot: {str(e)}")

    # Add more visualizations in similar try-except blocks...
    
except Exception as e:
    st.error(f"Error rendering charts: {str(e)}")

# Raw data and download
if st.checkbox("Show raw data"):
    st.dataframe(filtered_df)

try:
    st.download_button(
        label="Download Data",
        data=filtered_df.to_csv(index=False),
        file_name="transaction_data.csv",
        mime="text/csv"
    )
except Exception as e:
    st.error(f"Error creating download button: {str(e)}")
