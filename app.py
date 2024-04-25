import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

def load_data():
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

    for month in months:
        file_path = f'{path_to_csvs}{month}.csv'
        try:
            month_data = pd.read_csv(file_path)
            if 'Tonnage' in month_data.columns and 'Truck Factor' in month_data.columns and 'Shovel' in month_data.columns:
                month_data = month_data[month_data['Truck Factor'] > 0]
                data.append(month_data)
          
        except FileNotFoundError:
            print(f"CSV file not found for {month}. Skipping...")

    return data

def load_shovel_fill_data(data, shovel):
    shovel_fill_data = []

    for df in data:
        df = df[df['Shovel'].isin(shovel)]
        truck_fill_percentage = (df['Tonnage'] / df['Truck Factor']) * 100
        shovel_fill_data.extend(truck_fill_percentage.dropna())

    return shovel_fill_data

def plot_distribution(shovel_fill_data, shovel, desired_mean=100, desired_std=5):
    if not shovel_fill_data:
        st.write("No data available for the selected shovel(s).")
        return

    actual_mean = np.mean(shovel_fill_data)
    actual_std = np.std(shovel_fill_data)

    x_min = 65
    x_max = 125
    x_range = np.linspace(x_min, x_max, 200)

    actual_distribution_y = norm.pdf(x_range, actual_mean, actual_std)
    desired_distribution_y = norm.pdf(x_range, desired_mean, desired_std)

    mean_std_text = (f"<b>Actual Mean:</b> {actual_mean:.2f}%<br>"
                     f"<b>Actual Std Dev:</b> {actual_std:.2f}%<br>"
                     f"<b>Desired Mean:</b> {desired_mean}%<br>"
                     f"<b>Desired Std Dev:</b> {desired_std}%")

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_range, y=actual_distribution_y, mode='lines', name='Actual Distribution', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_range, y=desired_distribution_y, mode='lines', name='Desired Distribution', line=dict(color='#00B7F1')))

    fig.add_trace(go.Scatter(x=[actual_mean, actual_mean], y=[0, max(actual_distribution_y)], mode='lines', name='Actual Mean', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=[desired_mean, desired_mean], y=[0, max(desired_distribution_y)], mode='lines', name='Desired Mean', line=dict(color='#00B7F1', dash='dash')))

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

    fig.update_layout(
        title=f'Actual vs Desired Truck Fill Distribution for {shovel}',
        xaxis_title='Truck Fill %',
        yaxis_title='Probability Density',
        legend_title='Legend',
        height=500,  # Adjusted height to fit the screen
        width=900,   # Adjusted width to fit the screen
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
    all_data = pd.concat([df for df in data])
    all_data['Time Full'] = pd.to_datetime(all_data['Time Full'], errors="coerce")
    all_data['Hour'] = all_data['Time Full'].dt.hour
    all_data['Shift'] = all_data['Hour'].apply(lambda x: 'Day' if 7 <= x < 19 else 'Night')
    all_data['Truck fill (%)'] = (all_data['Tonnage'] / all_data['Truck Factor']) * 100
    all_data['Month'] = all_data['Time Full'].dt.month
    all_data['Season'] = all_data['Month'].apply(month_to_season)
    all_data['Year'] = all_data['Time Full'].dt.year
    all_data['Adjusted Hour'] = (all_data['Hour'] + 17) % 24  # Adjust the hour for plotting
    all_data = all_data.dropna(subset=['Time Full', 'Material', 'Truck fill (%)'])

    # Drop columns with all NaN values
    all_data = all_data.dropna(axis=1, how='all')

    return all_data



def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

def create_timeseries_plot(data):
    average_fill_by_hour_shift = data.groupby(['Adjusted Hour', 'Shift'])['Truck fill (%)'].mean().reset_index()

    trace_day = go.Scatter(
        x=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Day']['Adjusted Hour'],
        y=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Day']['Truck fill (%)'],
        mode='lines',
        name='Day Shift',
        line=dict(color='red')
    )

    trace_night = go.Scatter(
        x=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Night']['Adjusted Hour'],
        y=average_fill_by_hour_shift[average_fill_by_hour_shift['Shift'] == 'Night']['Truck fill (%)'],
        mode='lines',
        name='Night Shift',
        line=dict(color='blue')
    )

    layout = go.Layout(
        title='Average Truck Fill by Hour and Shift (7 AM to 7 AM)',
        xaxis=dict(title='Hour (7 AM to 7 AM)', dtick=1, tickvals=list(range(24)), ticktext=[f"{(h+7)%24}:00" for h in range(24)]),
        yaxis=dict(title='Average Truck Fill (%)')
    )

    fig = go.Figure(data=[trace_day, trace_night], layout=layout)

    return fig



def adjust_hours_for_plot(df):
    # Convert 'Time Full' to datetime, if not already
    df['Time Full'] = pd.to_datetime(df['Time Full'], errors="coerce")
    
    # Shift 'Time Full' by -7 hours so the day starts at 7 AM
    df['Adjusted Time Full'] = df['Time Full'] - pd.Timedelta(hours=7)

    # Extract the adjusted hour and day
    df['Adjusted Hour'] = df['Adjusted Time Full'].dt.hour
    df['Adjusted Day'] = df['Adjusted Time Full'].dt.date

    # Now sort by the adjusted time for proper plotting and analysis
    df.sort_values(by=['Adjusted Time Full'], inplace=True)
    
    return df

def calculate_material_increase(current_mean, current_std, desired_mean, desired_std):
    # Calculate the z-score for the current and desired means
    z_current = (desired_mean - current_mean) / current_std
    z_desired = 0
    
    # Adjust z-score if desired_mean is lower than current_mean
    if desired_mean < current_mean:
        z_current = -z_current
        z_desired = 0

    # Calculate potential increase based on cumulative distribution function
    potential_increase = norm.cdf(z_current) - norm.cdf(z_desired)

    # Return potential increase as a percentage
    return potential_increase * 100


def load_truck_fill_data(data, shovels, selected_mean, selected_std):
    # Filter data for selected shovels
    data = data[data['Shovel'].isin(shovels)]
    total_improvement = 0

    month_aggregated_data = {}

    # Group data by month-year
    data['Month'] = data['Time Full'].dt.strftime('%B')
    data['Month-Year'] = data['Month'].astype(str) + ' ' + data['Year'].astype(str)
    data = data.sort_values(by='Year')

    # Drop rows with missing values in 'Time Full' column
    data = data.dropna(subset=['Time Full'])

    # Iterate over unique month-year combinations
    for month in data['Month-Year'].unique():
        month_data = data[data['Month-Year'] == month]
        
        # Calculate current truck fill rate and material moved
        current_truck_fill = (month_data['Tonnage'] / month_data['Truck Factor']) * 100
        current_material_moved = month_data['Tonnage'].sum()
        
        # Calculate desired material moved using calculate_material_increase() function
        desired_material_moved = current_material_moved * (1 + calculate_material_increase(current_truck_fill.mean(), current_truck_fill.std(), selected_mean, selected_std) / 100)
        
        # Calculate improvement as the difference between desired and current material moved
        improvement = desired_material_moved - current_material_moved
        total_improvement += improvement

        # Aggregate data by month-year
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

    # Create DataFrame from aggregated data
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

    # Handle missing or infinite values before rounding and integer conversion
    result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    result_df.dropna(inplace=True)

    # Round values to the nearest tonne and convert to integers
    result_df[['Current Material (tonnes)', 'Desired Material (tonnes)', 'Improvement (tonnes)']] = \
        result_df[['Current Material (tonnes)', 'Desired Material (tonnes)', 'Improvement (tonnes)']].round().astype(int)

    return result_df




def generate_markdown_explanation(actual_mean, actual_std, desired_mean, desired_std, shovel):
    explanation = f"""
    The purpose of this analysis is to evaluate the potential improvements in operational efficiency with the implementation of ShovelMetrics™ Payload Monitoring (SM-PLM). By analyzing the truck fill distribution data, we aim to identify areas where optimizations can be made to enhance productivity and reduce operational risks. To illustrate potential improvements with SM-PLM for shovel '{shovel}', the below distributions are shown with a target fill of {desired_mean}% and a standard deviation of {desired_std}% to emulate the distribution with SM-PLM.
    """
    return explanation

def main():
    st.title("Potential Improvements to Operational Efficiency with ShovelMetrics™ PLM")
    st.markdown("Prepared for: Aktogay Mine")
    st.markdown("Date: 2024-04-25")
    intro_placeholder = st.empty()

    data = load_data()

    # Add a global filter for material type
    all_materials = list(set([value for df in data for value in df['Material'].unique() if 'Material' in df.columns]))
    selected_materials = st.sidebar.multiselect("Select Material Type", all_materials)

    # If no material is selected, use all materials
    if not selected_materials:
        selected_materials = all_materials

    # Filter data based on selected material type
    data = [df[df['Material'].isin(selected_materials)] for df in data if 'Material' in df.columns]

    all_shovels = list(set([value for df in data for value in df['Shovel'].unique() if 'Shovel' in df.columns]))
    all_shovels.append('All')

    selected_shovels = st.sidebar.multiselect("Select Shovel", all_shovels, default=['All'])

    if 'All' in selected_shovels or len(selected_shovels) == 0:
        selected_shovels = [shovel for shovel in all_shovels if shovel != 'All']

    if len(selected_shovels) == 0:
        selected_shovels = ['All']

    if 'All' in selected_shovels:
        selected_shovels = all_shovels
        selected_shovels.remove('All')

    selected_mean = st.sidebar.slider("Select Mean (%)", 98, 110, 100, step=1)
    selected_std = st.sidebar.slider("Select Standard Deviation (%)", 1, 10, 5, step=1)

    explanation_placeholder = st.empty()

    shovel_fill_data = load_shovel_fill_data(data, selected_shovels)
    fig = plot_distribution(shovel_fill_data, selected_shovels, selected_mean, selected_std)
    st.plotly_chart(fig, use_container_width=True)

    actual_mean = np.mean(shovel_fill_data) if shovel_fill_data else 0
    actual_std = np.std(shovel_fill_data) if shovel_fill_data else 0
    explanation_text = generate_markdown_explanation(actual_mean, actual_std, selected_mean, selected_std, selected_shovels)
    explanation_placeholder.markdown(explanation_text)

    data = process_loaded_data(data)
    data = data[data['Shovel'].isin(selected_shovels)]
    fig = create_timeseries_plot(data)
    st.plotly_chart(fig, use_container_width=True)

    # The rest of your code remains unchanged...

    day_mean = data[data['Shift'] == 'Day']['Truck fill (%)'].mean()
    night_mean = data[data['Shift'] == 'Night']['Truck fill (%)'].mean()

    # Calculate the peak periods for average truck fill during day and night shifts
    peak_day_start = None
    peak_day_end = None
    peak_night_start = None
    peak_night_end = None

    # Group data by shift and hour, then calculate the average truck fill
    shift_hourly_average = data.groupby(['Shift', 'Hour'])['Truck fill (%)'].mean().reset_index()

    # Find the peak periods for day and night shifts
    day_shift_data = shift_hourly_average[shift_hourly_average['Shift'] == 'Day']
    night_shift_data = shift_hourly_average[shift_hourly_average['Shift'] == 'Night']

    try:
        peak_day = day_shift_data.loc[day_shift_data['Truck fill (%)'].idxmax()]
        peak_day_start = peak_day['Hour']
        peak_day_end = (peak_day_start + 6) % 24  # Assuming the peak period is 6 hours long
    except ValueError:
        pass

    try:
        peak_night = night_shift_data.loc[night_shift_data['Truck fill (%)'].idxmax()]
        peak_night_start = peak_night['Hour']
        peak_night_end = (peak_night_start + 6) % 24  # Assuming the peak period is 6 hours long
    except ValueError:
        pass

    # Output the peak periods
    if peak_day_start is not None and peak_day_end is not None and peak_night_start is not None and peak_night_end is not None:
        peak_message = ("The peak average truck fill percentage during day shifts occurs between {} {} and {} {} with {:.2f}%, "
                        "while during night shifts, it occurs between {} {} and {} {} with {:.2f}%.")

        peak_day_start_period = "AM" if peak_day_start < 12 else "PM"
        peak_day_end_period = "AM" if peak_day_end < 12 else "PM"
        peak_night_start_period = "AM" if peak_night_start < 12 else "PM"
        peak_night_end_period = "AM" if peak_night_end < 12 else "PM"

        st.write(peak_message.format(
            peak_day_start % 12, peak_day_start_period,
            peak_day_end % 12, peak_day_end_period,
            peak_day['Truck fill (%)'],
            peak_night_start % 12, peak_night_start_period,
            peak_night_end % 12, peak_night_end_period,
            peak_night['Truck fill (%)']
        ))

    selected_title = f"Material Destination Distribution for {selected_shovels}"
    st.markdown(f'<h1 style="font-size: 24px;">{selected_title}</h1>', unsafe_allow_html=True)
    st.markdown("The chart below illustrates the proportion of trucks sent to one of four locations: Crusher, Stockpiles, DLP, and Other. Kindly note that We are only analyzing truck fill for trucks that are sent to the crusher.")
    shovel_data = data[data['Shovel'].isin(selected_shovels)]
    data_cleaned = shovel_data.dropna()

    def categorize_destination(destination):
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

    other_destinations = data_cleaned[data_cleaned['Destination Category'] == 'Other']['Assigned Dump'].unique()


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

    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    fig_monthly = go.Figure()
    for season, color in colors.items():
        season_data = monthly_performance[monthly_performance['Season'] == season]
        fig_monthly.add_trace(go.Bar(x=season_data['Month'].map(month_names), y=season_data['Truck Fill Rate (%)'],
                                    name=f'{season} Average', marker_color=color))
        fig_monthly.add_trace(go.Scatter(x=season_data['Month'].map(month_names), y=[season_data['Truck Fill Rate (%)'].mean()] * len(season_data),
                                        mode='lines+markers', name=f'{season} Average', line=dict(color=color, dash='dash', width=2),
                                        marker=dict(color=color, size=8)))

    fig_monthly.update_layout(xaxis_title='Month', yaxis_title='Average Truck Fill Rate (%)',
                            template='plotly_white', yaxis=dict(range=[80, 105]),
                            title=dict(text=f'Monthly Truck Fill Rate Trends for {selected_shovels}', font=dict(size=18, color='black', family="Arial")))
    st.plotly_chart(fig_monthly, use_container_width=True)

       # ---------------------------- Tabular View ---------------------------------------------------
    # Load your truck fill rate data


    # Sort the DataFrame by year and month
    results_df = load_truck_fill_data(data, selected_shovels, selected_mean, selected_std)

    results_df['Month'] = pd.Categorical(results_df['Month'], categories=[
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December', 'Total'
    ], ordered=True)
    results_df = results_df.sort_values(by=['Year', 'Month'])
    results_df = results_df.reset_index(drop=True)

    shovel_text = ", ".join(selected_shovels) if selected_shovels else "All Shovels"
    st.markdown(f"<h4><b>Current and Desired Truck Fill Rates for {shovel_text}</b></h4>", unsafe_allow_html=True)

    # Add CSS styling to the header of the table to change the background color
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
    st.table(results_df)

if __name__ == '__main__':
    main()
