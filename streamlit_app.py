# -*- coding: utf-8 -*-
"""Enhanced Transaction Analytics Dashboard"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Advanced Transaction Analytics",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=600)
def load_data():
    url = "https://raw.githubusercontent.com/Matshisela/imali_simulation_24/refs/heads/main/data/transaction_data_1_year.csv"
    df = pd.read_csv(url, parse_dates=['timestamp'])
    
    # Create business-relevant features
    df['date'] = df['timestamp'].dt.date
    df['day_of_week'] = df['timestamp'].dt.day_name()
    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month_name()
    df['week_number'] = df['timestamp'].dt.isocalendar().week
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday'])
    df['value_segment'] = pd.cut(df['amount'],
                               bins=[0, 10, 20, 50, 100, np.inf],
                               labels=['Micro (<$10)', 'Small ($10-20)', 
                                      'Medium ($20-50)', 'Large ($50-100)', 
                                      'Premium ($100+)'])
    return df

df = load_data()

# =============================================
# SIDEBAR FILTERS
# =============================================
st.sidebar.header("Business Filters")

# Date range filter
date_range = st.sidebar.date_input(
    "Date Range",
    value=[df['date'].min(), df['date'].max()],
    min_value=df['date'].min(),
    max_value=df['date'].max()
)

# Dynamic filters
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
    "Status",
    options=df['status'].unique(),
    default=['Success']  # Default to only successful transactions
)

value_segments = st.sidebar.multiselect(
    "Value Segments",
    options=df['value_segment'].unique(),
    default=df['value_segment'].unique()
)

devices = st.sidebar.multiselect(
    "Device Types",
    options=df['device'].unique(),
    default=df['device'].unique()
)

# Apply filters
if len(date_range) == 2:
    mask = (
        (df['date'] >= date_range[0]) & 
        (df['date'] <= date_range[1]) &
        (df['service'].isin(services)) &
        (df['payment_method'].isin(payment_methods)) &
        (df['status'].isin(statuses)) &
        (df['value_segment'].isin(value_segments)) &
        (df['device'].isin(devices))
    )
    filtered_df = df[mask]
else:
    filtered_df = df.copy()

# =============================================
# BUSINESS METRICS DASHBOARD
# =============================================
st.title("💰 Transaction Performance Dashboard")
st.markdown("""
<style>
.metric-card {
    padding: 15px;
    border-radius: 10px;
    background-color: #f8f9fa;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}
.metric-title {
    font-size: 14px;
    color: #6c757d;
    margin-bottom: 5px;
}
.metric-value {
    font-size: 24px;
    font-weight: bold;
    color: #212529;
}
.metric-change {
    font-size: 12px;
}
.positive {
    color: #28a745;
}
.negative {
    color: #dc3545;
}
</style>
""", unsafe_allow_html=True)

# Calculate comparison metrics (previous period)
def get_comparison_metrics(df, filtered_df):
    if len(date_range) == 2:
        # Calculate previous period dates
        delta_days = (date_range[1] - date_range[0]).days
        prev_start = date_range[0] - pd.Timedelta(days=delta_days+1)
        prev_end = date_range[0] - pd.Timedelta(days=1)
        
        # Filter previous period data with same other filters
        prev_mask = (
            (df['date'] >= prev_start) & 
            (df['date'] <= prev_end) &
            (df['service'].isin(services)) &
            (df['payment_method'].isin(payment_methods)) &
            (df['status'].isin(statuses)) &
            (df['value_segment'].isin(value_segments)) &
            (df['device'].isin(devices))
        )
        prev_df = df[prev_mask]
        
        # Calculate metrics
        current_volume = filtered_df.shape[0]
        prev_volume = prev_df.shape[0]
        volume_change = ((current_volume - prev_volume) / prev_volume * 100) if prev_volume > 0 else 0
        
        current_amount = filtered_df['amount'].sum()
        prev_amount = prev_df['amount'].sum()
        amount_change = ((current_amount - prev_amount) / prev_amount * 100) if prev_amount > 0 else 0
        
        current_users = filtered_df['user_id'].nunique()
        prev_users = prev_df['user_id'].nunique()
        users_change = ((current_users - prev_users) / prev_users * 100) if prev_users > 0 else 0
        
        return {
            'volume_change': volume_change,
            'amount_change': amount_change,
            'users_change': users_change,
            'prev_volume': prev_volume,
            'prev_amount': prev_amount,
            'prev_users': prev_users
        }
    return None

comparison = get_comparison_metrics(df, filtered_df)

# Row 1: Key Business Metrics
st.subheader("Core Business Performance")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Total Transactions</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{filtered_df.shape[0]:,}</div>', unsafe_allow_html=True)
    if comparison:
        change_class = "positive" if comparison['volume_change'] >= 0 else "negative"
        st.markdown(f'<div class="metric-change {change_class}">{"↑" if comparison["volume_change"] >=0 else "↓"} {abs(comparison["volume_change"]):.1f}% vs previous period</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-change">Prev: {comparison["prev_volume"]:,}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Total Value</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">${filtered_df["amount"].sum():,.2f}</div>', unsafe_allow_html=True)
    if comparison:
        change_class = "positive" if comparison['amount_change'] >= 0 else "negative"
        st.markdown(f'<div class="metric-change {change_class}">{"↑" if comparison["amount_change"] >=0 else "↓"} {abs(comparison["amount_change"]):.1f}% vs previous period</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-change">Prev: ${comparison["prev_amount"]:,.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Unique Customers</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{filtered_df["user_id"].nunique():,}</div>', unsafe_allow_html=True)
    if comparison:
        change_class = "positive" if comparison['users_change'] >= 0 else "negative"
        st.markdown(f'<div class="metric-change {change_class}">{"↑" if comparison["users_change"] >=0 else "↓"} {abs(comparison["users_change"]):.1f}% vs previous period</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-change">Prev: {comparison["prev_users"]:,}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Avg. Transaction Value</div>', unsafe_allow_html=True)
    avg_value = filtered_df['amount'].mean()
    st.markdown(f'<div class="metric-value">${avg_value:,.2f}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col5:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Success Rate</div>', unsafe_allow_html=True)
    success_rate = (filtered_df['status'] == 'Success').mean() * 100
    st.markdown(f'<div class="metric-value">{success_rate:.1f}%</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Row 2: Revenue Analysis
st.subheader("Revenue Analysis")
col1, col2 = st.columns(2)

with col1:
    # Revenue by service
    revenue_by_service = filtered_df.groupby('service')['amount'].sum().sort_values(ascending=False)
    fig = px.bar(
        revenue_by_service,
        title="Revenue by Service",
        labels={'value': 'Revenue ($)', 'index': 'Service'},
        color=revenue_by_service.index,
        text=[f"${x:,.0f}" for x in revenue_by_service.values]
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Revenue trend
    daily_revenue = filtered_df.groupby('date')['amount'].sum().reset_index()
    fig = px.line(
        daily_revenue,
        x='date',
        y='amount',
        title="Daily Revenue Trend",
        labels={'amount': 'Revenue ($)', 'date': 'Date'}
    )
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig, use_container_width=True)

# =============================================
# CUSTOMER & VALUE ANALYSIS SECTION - REORGANIZED
# =============================================

st.subheader("Customer Value Distribution")
# Full width for the customer value histogram
fig = px.histogram(
    filtered_df.groupby('user_id')['amount'].sum(),
    nbins=20,
    title="Customer Lifetime Value Distribution",
    labels={'value': 'Total Spend per Customer ($)', 'count': 'Number of Customers'},
    color_discrete_sequence=['#1f77b4']
)
fig.update_layout(
    height=400,
    xaxis_title="Total Customer Spend",
    yaxis_title="Number of Customers",
    bargap=0.1
)
st.plotly_chart(fig, use_container_width=True)

# Value Segment Analysis in separate row
st.subheader("Transaction Value Segmentation")
col1, col2 = st.columns(2)

with col1:
    # Transactions by value segment (pie chart)
    segment_counts = filtered_df['value_segment'].value_counts()
    fig = px.pie(
        segment_counts,
        names=segment_counts.index,
        values=segment_counts.values,
        title="Transaction Volume by Value Segment",
        hole=0.3,
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate="<b>%{label}</b><br>%{value:,} transactions (%{percent})"
    )
    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='hide',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Average transaction by segment (horizontal bar chart)
    avg_by_segment = filtered_df.groupby('value_segment')['amount'].mean().sort_values()
    fig = px.bar(
        avg_by_segment,
        orientation='h',
        title="Average Transaction Value by Segment",
        labels={'value': 'Average Amount ($)', 'index': 'Value Segment'},
        color=avg_by_segment.index,
        color_discrete_sequence=px.colors.sequential.Blues_r,
        text=[f"${x:,.2f}" for x in avg_by_segment.values]
    )
    fig.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="Average Transaction Amount ($)",
        yaxis_title="Value Segment",
        hovermode="y unified"
    )
    fig.update_traces(
        textposition='outside',
        textfont_size=12
    )
    st.plotly_chart(fig, use_container_width=True)


# Row 4: Payment and Operational Metrics
st.subheader("Payment & Operational Performance")
col1, col2 = st.columns(2)

with col1:
    # Payment method performance
    payment_stats = filtered_df.groupby('payment_method').agg(
        total_amount=('amount', 'sum'),
        success_rate=('status', lambda x: (x == 'Success').mean()),
        avg_amount=('amount', 'mean')
    ).sort_values('total_amount', ascending=False)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=payment_stats.index,
            y=payment_stats['total_amount'],
            name="Total Revenue",
            marker_color='#1f77b4'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=payment_stats.index,
            y=payment_stats['success_rate'],
            name="Success Rate",
            mode='lines+markers',
            line=dict(color='#ff7f0e', width=2)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title="Payment Method Performance",
        yaxis_title="Revenue ($)",
        yaxis2=dict(tickformat=".0%", range=[0, 1]),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Device and time analysis
    device_hour = filtered_df.pivot_table(
        index='hour',
        columns='device',
        values='transaction_id',
        aggfunc='count',
        fill_value=0
    )
    
    fig = px.line(
        device_hour,
        title="Hourly Activity by Device",
        labels={'value': 'Transactions', 'hour': 'Hour of Day'}
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# Row 5: Customer Retention Metrics
st.subheader("Customer Behavior Insights")

# Calculate repeat customers
user_activity = filtered_df.groupby('user_id').agg(
    transaction_count=('transaction_id', 'count'),
    first_date=('date', 'min'),
    last_date=('date', 'max'),
    total_value=('amount', 'sum')
).reset_index()

repeat_customers = user_activity[user_activity['transaction_count'] > 1]

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Repeat Customers</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-value">{len(repeat_customers):,}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-change">{len(repeat_customers)/len(user_activity):.1%} of total</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Avg. Frequency</div>', unsafe_allow_html=True)
    avg_freq = user_activity['transaction_count'].mean()
    st.markdown(f'<div class="metric-value">{avg_freq:.1f} tx/user</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown('<div class="metric-title">Top 10% Customer Share</div>', unsafe_allow_html=True)
    top_10_value = user_activity.nlargest(int(len(user_activity)*0.1), 'total_value')['total_value'].sum()
    total_value = user_activity['total_value'].sum()
    share = top_10_value / total_value if total_value > 0 else 0
    st.markdown(f'<div class="metric-value">{share:.1%}</div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-change">of total revenue</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Raw data and download
if st.checkbox("Show filtered data"):
    st.dataframe(filtered_df.sort_values('timestamp', ascending=False))

st.download_button(
    label="Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name="filtered_transactions.csv",
    mime="text/csv"
)
