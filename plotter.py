import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st

def plot_fill_distribution(fill_data, shovels, desired_mean=100, desired_std=5):
    """Plot actual vs desired truck fill distribution."""
    if not fill_data.size:
        st.write("No data available for selected shovel(s).")
        return None

    actual_mean, actual_std = np.mean(fill_data), np.std(fill_data)
    x_range = np.linspace(65, 125, 200)
    actual_y = norm.pdf(x_range, actual_mean, actual_std)
    desired_y = norm.pdf(x_range, desired_mean, desired_std)

    fig = go.Figure([
        go.Scatter(x=x_range, y=actual_y, mode='lines', name='Actual', line=dict(color='red')),
        go.Scatter(x=x_range, y=desired_y, mode='lines', name='Desired', line=dict(color='#00B7F1')),
        go.Scatter(x=[actual_mean, actual_mean], y=[0, max(actual_y)], mode='lines', name='Actual Mean', line=dict(color='red', dash='dash')),
        go.Scatter(x=[desired_mean, desired_mean], y=[0, max(desired_y)], mode='lines', name='Desired Mean', line=dict(color='#00B7F1', dash='dash'))
    ])

    fig.add_annotation(
        text=f"<b>Actual Mean:</b> {actual_mean:.2f}%<br><b>Actual Std:</b> {actual_std:.2f}%<br>"
             f"<b>Desired Mean:</b> {desired_mean}%<br><b>Desired Std:</b> {desired_std}%",
        xref='paper', yref='paper', x=1, y=1, showarrow=False, align='left',
        bordercolor='black', borderwidth=1, bgcolor='white'
    )

    fig.update_layout(
        title=f'Truck Fill Distribution for {shovels}',
        xaxis_title='Truck Fill %', yaxis_title='Probability Density',
        height=500, width=900, xaxis=dict(range=[65, 125]),
        legend=dict(x=1.05, y=0.5, bgcolor='rgba(255,255,255,0.5)')
    )
    return fig

def plot_timeseries(data):
    """Plot average truck fill by hour and shift."""
    hourly_shift_avg = data.groupby(['Adjusted Hour', 'Shift'])['Truck Fill (%)'].mean().reset_index()
    fig = go.Figure([
        go.Scatter(
            x=hourly_shift_avg[hourly_shift_avg['Shift'] == 'Day']['Adjusted Hour'],
            y=hourly_shift_avg[hourly_shift_avg['Shift'] == 'Day']['Truck Fill (%)'],
            mode='lines', name='Day Shift', line=dict(color='red')
        ),
        go.Scatter(
            x=hourly_shift_avg[hourly_shift_avg['Shift'] == 'Night']['Adjusted Hour'],
            y=hourly_shift_avg[hourly_shift_avg['Shift'] == 'Night']['Truck Fill (%)'],
            mode='lines', name='Night Shift', line=dict(color='blue')
        )
    ])
    fig.update_layout(
        title='Average Truck Fill by Hour and Shift (7 AM to 7 AM)',
        xaxis_title='Hour (7 AM to 7 AM)',
        yaxis_title='Average Truck Fill (%)',
        xaxis=dict(dtick=1, tickvals=list(range(24)), ticktext=[f"{(h+7)%24}:00" for h in range(24)])
    )
    return fig

def plot_destination_pie(data, shovels):
    """Plot pie chart for material destination distribution."""
    df = data[data['Shovel'].isin(shovels)].dropna()
    df['Destination'] = df['Assigned Dump'].apply(
        lambda x: 'Crusher' if x.upper().startswith('CRUSHER') else
                  'STK' if x.startswith('STK') else
                  'DLP' if x.startswith('DLP') else 'Other'
    )
    counts = df['Destination'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.3, textinfo='percent+label')])
    return fig

def plot_material_grade_pie(data, shovels):
    """Plot pie chart for material grade distribution."""
    df = data[data['Shovel'].isin(shovels)].dropna()
    counts = df['Material'].value_counts()
    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.3, textinfo='percent+label')])
    return fig

def plot_monthly_trends(data, shovels):
    """Plot monthly truck fill rate trends by season."""
    df = data[(data['Shovel'].isin(shovels)) & (data['Tonnage'] != 0) & (data['Truck Factor'] != 0)]
    df['Truck Fill Rate (%)'] = (df['Tonnage'] / df['Truck Factor']) * 100
    monthly = df.groupby(['Season', 'Month'])['Truck Fill Rate (%)'].mean().reset_index()

    colors = {'Winter': '#1f77b4', 'Spring': '#ff7f0e', 'Summer': '#2ca02c', 'Fall': '#d62728'}
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    fig = go.Figure()
    for season, color in colors.items():
        season_data = monthly[monthly['Season'] == season]
        fig.add_trace(go.Bar(
            x=season_data['Month'].map(month_names), y=season_data['Truck Fill Rate (%)'],
            name=f'{season} Average', marker_color=color
        ))
        fig.add_trace(go.Scatter(
            x=season_data['Month'].map(month_names), y=[season_data['Truck Fill Rate (%)'].mean()] * len(season_data),
            mode='lines+markers', name=f'{season} Avg Line', line=dict(color=color, dash='dash'),
            marker=dict(color=color, size=8)
        ))

    fig.update_layout(
        title=f'Monthly Truck Fill Rate Trends for {shovels}',
        xaxis_title='Month', yaxis_title='Average Truck Fill Rate (%)',
        yaxis=dict(range=[80, 105]), xaxis=dict(type='category', categoryorder='array',
                                               categoryarray=list(month_names.values()))
    )
    return fig
