import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

def load_data():
    """
    Load and validate CSV files containing truck fill data.
    
    Returns:
        list: List of pandas DataFrames with valid data.
    """
    path_to_csvs = './'
    months = [
        'Load DetailApril2023',
        'Load DetailAugust2023.1-15',
        'Load DetailSeptember2023',
        'Load DetailAugust2023.16-31',
        'Load DetailDecember1-15.2023',
        'Load DetailDecember16-31.2023',
        'Load DetailFebruary2023',
        'Load DetailJanuary2023',
        'Load DetailJuly2023',
        'Load DetailJUNE2023',
        'Load DetailMarch2023',
        'Load DetailMay2023',
        'Load DetailNovember1-15.2023',
        'Load DetailNovember16-30.2023'
    ]
    
    data = []
    required_columns = ['Tonnage', 'Truck Factor', 'Shovel']
    
    for month in months:
        file_path = f'{path_to_csvs}{month}.csv'
        try:
            month_data = pd.read_csv(file_path)
            if all(col in month_data.columns for col in required_columns):
                month_data = month_data[(month_data['Truck Factor'] > 0) & (month_data['Tonnage'] > 0)]
                if not month_data.empty:
                    data.append(month_data)
                else:
                    st.warning(f"No valid rows in {month}.csv after filtering.")
            else:
                st.warning(f"Missing required columns in {month}.csv. Skipping...")
        except FileNotFoundError:
            st.warning(f"File not found: {file_path}. Skipping...")
        except pd.errors.EmptyDataError:
            st.warning(f"Empty file: {file_path}. Skipping...")
    
    if not data:
        st.error("No valid CSV files found or no data with required columns.")
    return data

def load_shovel_fill_data(data, shovels):
    """
    Load truck fill percentage data for selected shovels.
    
    Args:
        data (list): List of pandas DataFrames.
        shovels (list): List of shovel names to filter.
    
    Returns:
        list: List of truck fill percentages.
    """
    shovel_fill_data = []
    
    for df in data:
        df = df[df['Shovel'].isin(shovels)]
        valid_rows = df[(df['Tonnage'] > 0) & (df['Truck Factor'] > 0)]
        if not valid_rows.empty:
            truck_fill_percentage = (valid_rows['Tonnage'] / valid_rows['Truck Factor']) * 100
            shovel_fill_data.extend(truck_fill_percentage.dropna())
        else:
            st.warning(f"No valid data for shovels {shovels} in this dataset.")
    
    return shovel_fill_data

def plot_distribution(shovel_fill_data, shovel, desired_mean=100, desired_std=5):
    """
    Plot actual vs. desired truck fill distribution.
    
    Args:
        shovel_fill_data (list): List of truck fill percentages.
        shovel (list or str): Selected shovel(s) for title.
        desired_mean (float): Desired mean truck fill percentage.
        desired_std (float): Desired standard deviation.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    if not shovel_fill_data:
        st.warning("No data available for the selected shovel(s).")
        return None
    
    # Filter outliers (keep data within reasonable range)
    shovel_fill_data = [x for x in shovel_fill_data if 0 < x < 200]
    if not shovel_fill_data:
        st.warning("No valid data after filtering outliers.")
        return None
    
    actual_mean = np.mean(shovel_fill_data)
    actual_std = np.std(shovel_fill_data)
    
    # Dynamic x-axis range
    x_min = max(0, min(shovel_fill_data) - 10)
    x_max = max(shovel_fill_data) + 10
    x_range = np.linspace(x_min, x_max, 200)
    
    actual_distribution_y = norm.pdf(x_range, actual_mean, actual_std)
    desired_distribution_y = norm.pdf(x_range, desired_mean, desired_std)
    
    mean_std_text = f"""
    <b>Actual Mean:</b> {actual_mean:.2f}%<br>
    <b>Actual Std Dev:</b> {actual_std:.2f}%<br>
    <b>Desired Mean:</b> {desired_mean}%<br>
    <b>Desired Std Dev:</b> {desired_std}%
    """
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=actual_distribution_y,
        mode='lines',
        name='Actual Distribution',
        line=dict(color='#D81B60')  # Colorblind-friendly
    ))
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=desired_distribution_y,
        mode='lines',
        name='Desired Distribution',
        line=dict(color='#1E88E5')
    ))
    
    fig.add_trace(go.Scatter(
        x=[actual_mean, actual_mean],
        y=[0, max(actual_distribution_y)],
        mode='lines',
        name='Actual Mean',
        line=dict(color='#D81B60', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=[desired_mean, desired_mean],
        y=[0, max(desired_distribution_y)],
        mode='lines',
        name='Desired Mean',
        line=dict(color='#1E88E5', dash='dash')
    ))
    
    fig.add_annotation(
        text=mean_std_text,
        align='left',
        showarrow=False,
        xref='paper',
        yref='paper',
        x=1,
        y=1,
        bordercolor='black',
        borderwidth=1,
        bgcolor='white',
        xanchor='left',
        yanchor='top'
    )
    
    shovel_title = ", ".join(shovel) if isinstance(shovel, list) else shovel
    fig.update_layout(
        title=f'Actual vs Desired Truck Fill Distribution for {shovel_title}',
        xaxis_title='Truck Fill %',
        yaxis_title='Probability Density',
        legend_title='Legend',
        height=500,
        width=900,
        legend=dict(
            x=1.05,
            y=0.5,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)'
        ),
        xaxis=dict(range=[x_min, x_max])
    )
    
    return fig

def process_loaded_data(data):
    """
    Process and transform loaded data for analysis.
    
    Args:
        data (list): List of pandas DataFrames.
    
    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    processed_dfs = []
    
    for df in data:
        df = df.copy()
        df['Time Full'] = pd.to_datetime(df['Time Full'], errors='coerce')
        invalid_dates = df['Time Full'].isna().sum()
        if invalid_dates > 0:
            st.warning(f"{invalid_dates} rows have invalid 'Time Full' values and will be excluded.")
        df = df.dropna(subset=['Time Full'])
        df['Hour'] = df['Time Full'].dt.hour
        df['Shift'] = df['Hour'].apply(lambda x: 'Day' if 7 <= x < 19 else 'Night')
        df['Truck fill (%)'] = (df['Tonnage'] / df['Truck Factor']) * 100
        df['Month'] = df['Time Full'].dt.month
        df['Season'] = df['Month'].apply(month_to_season)
        df['Year'] = df['Time Full'].dt.year
        df['Adjusted Hour'] = (df['Hour'] + 17) % 24
        df = df.dropna(subset=['Material', 'Truck fill (%)'])
        df = df.dropna(axis=1, how='all')
        processed_dfs.append(df)
    
    if not processed_dfs:
        st.error("No valid data after processing.")
        return pd.DataFrame()
    
    return pd.concat(processed_dfs, ignore_index=True)

def month_to_season(month):
    """
    Map month to season.
    
    Args:
        month (int): Month number (1-12).
    
    Returns:
        str: Season name.
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def create_time_series_plot(data):
    """
    Create a time series plot of average truck fill by hour and shift.
    
    Args:
        data (pd.DataFrame): Input DataFrame with processed data.
    
    Returns:
        go.Figure: Plotly figure object.
    """
    if data.empty:
        st.warning("No data available for time series plot.")
        return None
    
    average_fill_by_hour_shift = data.groupby(['Adjusted Hour', 'Shift'])['Truck fill (%)'].mean().reset_index()
    
    trace_day = go.Scatter(
        x=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Day']['Adjusted Hour'],
        y=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Day']['Truck fill (%)'],
        mode='lines',
        name='Day Shift',
        line=dict(color='#D81B60')
    )
    
    trace_night = go.Scatter(
        x=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Night']['Adjusted Hour'],
        y=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Night']['Truck fill (%)'],
        mode='lines',
        name='Night Shift',
        line=dict(color='#1E88E5')
    )
    
    layout = go.Layout(
        title='Average Truck Fill by Hour and Shift (7 AM to 7 AM)',
        xaxis=dict(title='Hour (7 AM to 7 AM)', dtick=1, tickvals=list(range(24)), ticktext=[f"{(h+7)%24}:00" for h in range(24)]),
        yaxis=dict(title='Average Truck Fill (%)')
    )
    
    fig = go.Figure(data=[trace_day, trace_night], layout=layout)
    return fig

def calculate_material_increase(current_mean, current_std, desired_mean, desired_std):
    """
    Calculate potential material increase based on truck fill distribution.
    
    Args:
        current_mean (float): Current mean truck fill percentage.
        current_std (float): Current standard deviation.
        desired_mean (float): Desired mean truck fill percentage.
        desired_std (float): Desired standard deviation.
    
    Returns:
        float: Potential increase percentage.
    """
    z_current = (desired_mean - current_mean) / current_std
    z_desired = 0
    
    if desired_mean < current_mean:
        z_current = -z_current
        z_desired = 0
    
    potential_increase = norm.cdf(z_current) - norm.cdf(z_desired)
    return potential_increase * 100

def load_truck_fill_data(data, shovels, selected_mean, selected_std):
    """
    Load and aggregate truck fill data by month-year.
    
    Args:
        data (pd.DataFrame): Processed DataFrame.
        shovels (list): List of shovel names.
        selected_mean (float): Desired mean truck fill percentage.
        selected_std (float): Desired standard deviation.
    
    Returns:
        pd.DataFrame: Aggregated results.
    """
    data = data[data['Shovel'].isin(shovels)]
    total_improvement = 0
    month_aggregated_data = {}
    
    data['Month'] = data['Time Full'].dt.strftime('%B')
    data['Month-Year'] = data['Month'].astype(str) + ' ' + data['Year'].astype(str)
    data = data.sort_values(by='Year')
    data = data.dropna(subset=['Time Full'])
    
    for month in data['Month-Year'].unique():
        month_data = data[data['Month-Year'] == month]
        if len(month_data) > 1:
            current_truck_fill = (month_data['Tonnage'] / month_data['Truck Factor']) * 100
            current_material_moved = month_data['Tonnage'].sum()
            desired_material_moved = current_material_moved * (1 + calculate_material_increase(
                current_truck_fill.mean(), current_truck_fill.std(), selected_mean, selected_std) / 100)
            improvement = desired_material_moved - current_material_moved
            total_improvement += improvement
            
            month_year_key = (month_data['Month'].iloc[0], month_data['Year'].iloc[0])
            
            if month_year_key in month_aggregated_data:
                month_aggregated_data[month_year_key]['Current Material (tonnes)'] += current_material_moved
                month_aggregated_data[month_year_key]['Desired Material (tonnes)'] += desired_material_moved
                month_aggregated_data[month_year_key]['Improvement (tonnes)'] += improvement
            else:
                month_aggregated_data[month_year_key] = {
                    'Month': month_data['Month'].iloc[0],
                    'Year': month_data['Year'].iloc[0],
                    'Current Truck Fill Rate': current_truck_fill.mean(),
                    'Desired Truck Fill Rate': selected_mean,
                    'Current Material (tonnes)': current_material_moved,
                    'Desired Material (tonnes)': desired_material_moved,
                    'Improvement (tonnes)': improvement
                }
    
    all_months_data = list(month_aggregated_data.values())
    total_row = {
        'Month': 'Total',
        'Year': '',
        'Current Truck Fill Rate': np.mean([data['Current Truck Fill Rate'] for data in all_months_data]),
        'Desired Truck Fill Rate': selected_mean,
        'Current Material (tonnes)': sum(data['Current Material (tonnes)'] for data in all_months_data),
        'Desired Material (tonnes)': sum(data['Desired Material (tonnes)'] for data in all_months_data),
        'Improvement (tonnes)': total_improvement
    }
    
    result_df = pd.DataFrame(all_months_data + [total_row])
    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    result_df.dropna(inplace=True)
    result_df[['Current Material (tonnes)', 'Desired Material (tonnes)', 'Improvement (tonnes)']] = \
        result_df[['Current Material (tonnes)', 'Desired Material (tonnes)', 'Improvement (tonnes)']].round().astype(int)
    
    return result_df

def generate_markdown_explanation(actual_mean, actual_std, desired_mean, desired_std, shovel):
    """
    Generate markdown explanation for truck fill analysis.
    
    Args:
        actual_mean (float): Actual mean truck fill percentage.
        actual_std (float): Actual standard deviation.
        desired_mean (float): Desired mean truck fill percentage.
        desired_std (float): Desired standard deviation.
        shovel (list or str): Selected shovel(s).
    
    Returns:
        str: Markdown text.
    """
    shovel_text = ", ".join(shovel) if isinstance(shovel, list) else shovel
    explanation = f"""
    The purpose of this analysis is to evaluate the potential improvements in operational efficiency with the implementation of ShovelMetrics™ Payload Monitoring (SM-PLM). By analyzing the truck fill distribution data, we aim to identify areas where optimizations can be made to enhance productivity and reduce operational risks. To illustrate potential improvements with SM-PLM for shovel '{shovel_text}', the below distributions are shown with a target fill of {desired_mean}% and a standard deviation of {desired_std}% to emulate the distribution with SM-PLM.
    """
    return explanation

def main():
    """
    Main Streamlit application for truck fill analysis.
    """
    st.title("Potential Improvements to Operational Efficiency with ShovelMetrics™ PLM")
    
    data = load_data()
    if not data:
        return
    
    session_state = st.session_state
    if 'selected_shovels' not in session_state:
        session_state.selected_shovels = []
    
    all_materials = list(set(value for df in data for value in df.get('Material', pd.Series()).unique()))
    selected_materials = st.sidebar.multiselect("Select Material Type", all_materials, default=all_materials)
    
    if not selected_materials:
        selected_materials = all_materials
    
    data = [df[df['Material'].isin(selected_materials)] for df in data if 'Material' in df.columns]
    
    all_shovels = list(set(value for df in data for value in df.get('Shovel', pd.Series()).unique()))
    selected_shovels = st.sidebar.multiselect(
        "Select Shovel", all_shovels + ['All'], default=session_state.selected_shovels
    )
    
    if 'All' in selected_shovels or not selected_shovels:
        selected_shovels = all_shovels
    session_state.selected_shovels = selected_shovels
    
    if not selected_shovels:
        st.warning("No shovels selected or available for the selected material.")
        return
    
    selected_mean = st.sidebar.slider("Select Mean (%)", 98, 110, 100, step=1)
    selected_std = st.sidebar.slider("Select Standard Deviation (%)", 1, 10, 5, step=1)
    
    shovel_fill_data = load_shovel_fill_data(data, selected_shovels)
    fig = plot_distribution(shovel_fill_data, selected_shovels, selected_mean, selected_std)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    actual_mean = np.mean(shovel_fill_data) if shovel_fill_data else 0
    actual_std = np.std(shovel_fill_data) if shovel_fill_data else 0
    explanation_text = generate_markdown_explanation(actual_mean, actual_std, selected_mean, selected_std, selected_shovels)
    st.markdown(explanation_text)
    
    data = process_loaded_data(data)
    data = data[data['Shovel'].isin(selected_shovels)]
    fig = create_time_series_plot(data)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    day_mean = data[data['Shift'] == 'Day']['Truck fill (%)'].mean()
    night_mean = data[data['Shift'] == 'Night']['Truck fill (%)'].mean()
    
    shift_hourly_average = data.groupby(['Shift', 'Hour'])['Truck fill (%)'].mean().reset_index()
    peak_day = shift_hourly_average[(shift_hourly_average['Shift'] == 'Day')].nlargest(1, 'Truck fill (%)')
    peak_night = shift_hourly_average[(shift_hourly_average['Shift'] == 'Night')].nlargest(1, 'Truck fill (%)')
    
    def format_hour(hour):
        return f"{hour%12 if hour != 12 else 12} {'AM' if hour < 12 or hour == 24 else 'PM'}"
    
    if not peak_day.empty and not peak_night.empty:
        peak_day_hour = peak_day['Hour'].values[0]
        peak_night_hour = peak_night['Hour'].values[0]
        peak_day_fill = peak_day['Truck fill (%)'].values[0]
        peak_night_fill = peak_night['Truck fill (%)'].values[0]
        peak_message = (
            f"The peak average truck fill percentage during night shifts occurs between {format_hour(peak_night_hour)} and {format_hour((peak_night_hour + 1) % 24)} with {peak_night_fill:.2f}%, "
            f"while during day shifts, it occurs between {format_hour(peak_day_hour)} and {format_hour((peak_day_hour + 1) % 24)} with {peak_day_fill:.2f}%."
        )
        st.write(peak_message)
    
    selected_title = f"Material Destination Distribution for {', '.join(selected_shovels)}"
    st.markdown(f'<h1 style="font-size: 24px;">{selected_title}</h1>', unsafe_allow_html=True)
    st.markdown("The chart below illustrates the proportion of trucks sent to one of four locations: Crusher, Stockpiles, DLP, and Other. Kindly note that we are only analyzing truck fill for trucks that are sent to the crusher.")
    
    shovel_data = data[data['Shovel'].isin(selected_shovels)]
    data_cleaned = shovel_data.dropna()
    
    def categorize_destination(destination):
        if pd.isna(destination):
            return 'Other'
        if destination.upper().startswith('CRUSHER'):
            return 'Crusher'
        elif destination.startswith('STK'):
            return 'STK'
        elif destination.startswith('DLP'):
            return 'DLP'
        else:
            return 'Other'
    
    data_cleaned['Destination Category'] = data_cleaned['Assigned Dump'].apply(categorize_destination)
    destination_counts = data_cleaned['Destination Category'].value_counts()
    
    st.markdown(f'<h2 style="font-size: 20px;">Material Destination Distribution</h2>', unsafe_allow_html=True)
    fig_destination = go.Figure(data=[go.Pie(labels=destination_counts.index, values=destination_counts.values, hole=0.3)])
    fig_destination.update_traces(textinfo='percent+label', textposition='inside')
    st.plotly_chart(fig_destination)
    
    material_data = data_cleaned['Material'].value_counts()
    fig_material_grade = go.Figure(data=[go.Pie(labels=material_data.index, values=material_data.values, hole=0.3)])
    fig_material_grade.update_traces(textinfo='percent+label', textposition='inside')
    st.markdown(f'<h2 style="font-size: 20px;">Material Grade Distribution</h2>', unsafe_allow_html=True)
    st.plotly_chart(fig_material_grade)
    
    shovel_data = data[(data['Shovel'].isin(selected_shovels)) & (data['Tonnage'] != 0) & (data['Truck Factor'] != 0)].dropna(subset=['Truck Factor', 'Tonnage'])
    shovel_data['Truck Fill Rate (%)'] = (shovel_data['Tonnage'] / shovel_data['Truck Factor']) * 100
    
    monthly_performance = shovel_data.groupby(['Season', 'Month'])['Truck Fill Rate (%)'].mean().reset_index()
    
    colors = {'Winter': '#1f77b4', 'Spring': '#ff7f0e', 'Summer': '#2ca02c', 'Fall': '#d62728'}
    monthly_performance['Color'] = monthly_performance['Season'].map(colors)
    
    month_names = {12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov'}
    
    fig_monthly = go.Figure()
    for season, color in colors.items():
        season_data = monthly_performance[monthly_performance['Season'] == season]
        fig_monthly.add_trace(go.Bar(
            x=season_data['Month'].map(month_names),
            y=season_data['Truck Fill Rate (%)'],
            name=f'{season} Average',
            marker_color=color
        ))
        fig_monthly.add_trace(go.Scatter(
            x=season_data['Month'].map(month_names),
            y=[season_data['Truck Fill Rate (%)'].mean()] * len(season_data),
            mode='lines+markers',
            name=f'{season} Average',
            line=dict(color=color, dash='dash', width=2),
            marker=dict(color=color, size=8)
        ))
    
    fig_monthly.update_layout(
        xaxis_title='Month',
        yaxis_title='Average Truck Fill Rate (%)',
        template='plotly_white',
        yaxis=dict(range=[80, 105]),
        title=dict(text=f'Monthly Truck Fill Rate Trends for {", ".join(selected_shovels)}', font=dict(size=18, color='black', family="Arial")),
        xaxis=dict(type='category', categoryorder='array', categoryarray=list(month_names.values()))
    )
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    results_df = load_truck_fill_data(data, selected_shovels, selected_mean, selected_std)
    
    import calendar
    results_df['Month'] = pd.Categorical(
        results_df['Month'], categories=list(calendar.month_name)[1:] + ['Total'], ordered=True
    )
    results_df = results_df.sort_values(by=['Year', 'Month'])
    results_df = results_df.reset_index(drop=True)
    
    shovel_text = ", ".join(selected_shovels) if selected_shovels else "All Shovels"
    st.markdown(f"<h4><b>Current and Desired Truck Fill Rates for {shovel_text}</b></h4>", unsafe_allow_html=True)
    
    header_html = """
    <style>
    th {
        background-color: #00B7F1;
    }
    th div {
        color: white;
    }
    </style>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    if not results_df.empty:
        st.table(results_df)
    else:
        st.warning("No data available for the results table.")

if __name__ == '__main__':
    main()
