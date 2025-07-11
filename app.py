import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

# Define CSV file names for data loading
CSV_FILES = [
    'Load DetailJanuary2023', 'Load DetailFebruary2023', 'Load DetailMarch2023',
    'Load DetailApril2023', 'Load DetailMay2023', 'Load DetailJUNE2023',
    'Load DetailJuly2023', 'Load DetailAugust2023.1-15', 'Load DetailAugust2023.16-31',
    'Load DetailSeptember2023', 'Load DetailNovember1-15.2023', 'Load DetailNovember16-30.2023',
    'Load DetailDecember1-15.2023', 'Load DetailDecember16-31.2023'
]

def load_data():
    """Load and filter CSV data, ensuring required columns exist."""
    data = []
    for file in CSV_FILES:
        try:
            df = pd.read_csv(f'./{file}.csv')
            if all(col in df for col in ['Tonnage', 'Truck Factor', 'Shovel']):
                df = df[df['Truck Factor'] > 0]
                data.append(df)
        except FileNotFoundError:
            st.warning(f"CSV file not found: {file}")
    return data

def get_shovel_fill_data(data, shovels):
    """Calculate truck fill percentages for selected shovels."""
    fill_data = []
    for df in data:
        df = df[df['Shovel'].isin(shovels)]
        fill_percent = (df['Tonnage'] / df['Truck Factor']) * 100
        fill_data.extend(fill_percent.dropna())
    return fill_data

def plot_fill_distribution(fill_data, shovels, desired_mean=100, desired_std=5):
    """Plot actual vs desired truck fill distribution."""
    if not fill_data:
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

def process_data(data):
    """Process concatenated data, adding time-based and fill rate columns."""
    df = pd.concat(data, ignore_index=True)
    df['Time Full'] = pd.to_datetime(df['Time Full'], errors='coerce')
    df['Hour'] = df['Time Full'].dt.hour
    df['Shift'] = df['Hour'].apply(lambda x: 'Day' if 7 <= x < 19 else 'Night')
    df['Truck Fill (%)'] = (df['Tonnage'] / df['Truck Factor']) * 100
    df['Month'] = df['Time Full'].dt.month
    df['Season'] = df['Month'].map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
                                    6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
                                    11: 'Fall', 12: 'Winter'})
    df['Year'] = df['Time Full'].dt.year
    df['Adjusted Hour'] = (df['Hour'] + 17) % 24
    return df.dropna(subset=['Time Full', 'Material', 'Truck Fill (%)'])

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

def calculate_material_increase(current_mean, current_std, desired_mean, desired_std):
    """Calculate potential material increase based on fill rate distributions."""
    z_current = (desired_mean - current_mean) / current_std
    if desired_mean < current_mean:
        z_current = -z_current
    return (norm.cdf(z_current) - norm.cdf(0)) * 100

def load_truck_fill_data(data, shovels, desired_mean, desired_std):
    """Generate table of current vs desired truck fill rates and material moved."""
    df = data[data['Shovel'].isin(shovels)].copy()
    df['Month-Year'] = df['Time Full'].dt.strftime('%B %Y')
    results = []

    for month_year in df['Month-Year'].unique():
        month_data = df[df['Month-Year'] == month_year]
        if len(month_data) <= 1:
            continue
        current_fill = (month_data['Tonnage'] / month_data['Truck Factor']) * 100
        current_material = month_data['Tonnage'].sum()
        increase = calculate_material_increase(current_fill.mean(), current_fill.std(), desired_mean, desired_std)
        desired_material = current_material * (1 + increase / 100)

        results.append({
            'Month': month_data['Month'].iloc[0],
            'Year': month_data['Year'].iloc[0],
            'Current Truck Fill Rate': current_fill.mean(),
            'Desired Truck Fill Rate': desired_mean,
            'Current Material (tonnes)': int(current_material),
            'Desired Material (tonnes)': int(desired_material),
            'Improvement (tonnes)': int(desired_material - current_material)
        })

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df
    total_row = {
        'Month': 'Total', 'Year': '',
        'Current Truck Fill Rate': result_df['Current Truck Fill Rate'].mean(),
        'Desired Truck Fill Rate': desired_mean,
        'Current Material (tonnes)': result_df['Current Material (tonnes)'].sum(),
        'Desired Material (tonnes)': result_df['Desired Material (tonnes)'].sum(),
        'Improvement (tonnes)': result_df['Improvement (tonnes)'].sum()
    }
    return pd.concat([result_df, pd.DataFrame([total_row])], ignore_index=True)

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

def main():
    """Main Streamlit app for truck fill analysis."""
    st.title("ShovelMetrics™ PLM Efficiency Analysis")
    data = load_data()

    # Sidebar filters
    materials = set(value for df in data for value in df['Material'].unique() if 'Material' in df.columns)
    selected_materials = st.sidebar.multiselect("Material Type", sorted(materials), default=list(materials))
    data = [df[df['Material'].isin(selected_materials)] for df in data if 'Material' in df.columns]

    shovels = set(value for df in data for value in df['Shovel'].unique() if 'Shovel' in df.columns)
    selected_shovels = st.sidebar.multiselect("Shovel", ['All'] + sorted(shovels),
                                             default=st.session_state.get('selected_shovels', []))
    st.session_state.selected_shovels = selected_shovels

    if 'All' in selected_shovels or not selected_shovels:
        selected_shovels = list(shovels)
    if not selected_shovels:
        st.warning("No data for selected material/shovel combination.")
        return

    desired_mean = st.sidebar.slider("Desired Mean (%)", 98, 110, 100)
    desired_std = st.sidebar.slider("Desired Std Dev (%)", 1, 10, 5)

    # Distribution plot
    fill_data = get_shovel_fill_data(data, selected_shovels)
    fig = plot_fill_distribution(fill_data, selected_shovels, desired_mean, desired_std)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Process data and create time series plot
    processed_data = process_data(data)
    processed_data = processed_data[processed_data['Shovel'].isin(selected_shovels)]
    st.plotly_chart(plot_timeseries(processed_data), use_container_width=True)

    # Peak period analysis
    shift_hourly = processed_data.groupby(['Shift', 'Hour'])['Truck Fill (%)'].mean().reset_index()
    peak_day = shift_hourly[shift_hourly['Shift'] == 'Day'].nlargest(1, 'Truck Fill (%)')
    peak_night = shift_hourly[shift_hourly['Shift'] == 'Night'].nlargest(1, 'Truck Fill (%)')

    if not peak_day.empty and not peak_night.empty:
        def format_hour(h): return f"{h%12 if h != 12 else 12} {'AM' if h < 12 else 'PM'}"
        st.write(
            f"Peak night shift fill: {peak_night['Truck Fill (%)'].values[0]:.2f}% between "
            f"{format_hour(peak_night['Hour'].values[0])} and {format_hour(peak_night['Hour'].values[0] + 1)}.<br>"
            f"Peak day shift fill: {peak_day['Truck Fill (%)'].values[0]:.2f}% between "
            f"{format_hour(peak_day['Hour'].values[0])} and {format_hour(peak_day['Hour'].values[0] + 1)}.",
            unsafe_allow_html=True
        )

    # Destination and material grade distributions
    st.markdown(f"### Material Destination for {selected_shovels}")
    st.plotly_chart(plot_destination_pie(processed_data, selected_shovels))
    st.markdown("### Material Grade Distribution")
    st.plotly_chart(plot_material_grade_pie(processed_data, selected_shovels))

    # Monthly trends
    st.plotly_chart(plot_monthly_trends(processed_data, selected_shovels), use_container_width=True)

    # Truck fill rate table
    results_df = load_truck_fill_data(processed_data, selected_shovels, desired_mean, desired_std)
    if not results_df.empty:
        st.markdown(f"### Truck Fill Rates for {', '.join(selected_shovels)}")
        st.markdown("<style>th {background-color: #00B7F1; color: white;}</style>", unsafe_allow_html=True)
        st.table(results_df.sort_values(['Year', 'Month']))

if __name__ == '__main__':
    main()
