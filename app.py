import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st
import calendar

def load_data(shovel):
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

    shovel_fill_data = []

    for month in months:
        file_path = f'{path_to_csvs}Cleaned_{month}.csv'
        try:
            month_data = pd.read_csv(file_path)
            
            if 'Tonnage' in month_data.columns and 'Truck Factor' in month_data.columns and 'Shovel' in month_data.columns:
                month_data = month_data[month_data['Truck Factor'] > 0]
                if shovel == 'All':
                    filtered_data = month_data
                else:
                    filtered_data = month_data[month_data['Shovel'] == shovel]
                truck_fill_percentage = (filtered_data['Tonnage'] / filtered_data['Truck Factor']) * 100
                shovel_fill_data.extend(truck_fill_percentage.dropna())
          
        except FileNotFoundError:
            print(f"CSV file not found for {month}. Skipping...")

    return shovel_fill_data

def plot_distribution(shovel_fill_data, shovel, desired_mean=100, desired_std=5):
    if not shovel_fill_data:
        st.write("No data available for the selected shovel.")
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

    st.plotly_chart(fig)

def generate_markdown_explanation(actual_mean, actual_std, desired_mean, desired_std):
    explanation = f"""
    To illustrate potential improvements with the implemented approach, the above distributions represent the actual and desired truck fill rates for the selected shovel.
    
    Actual Mean: {actual_mean:.2f}%
    Actual Std Dev: {actual_std:.2f}%
    Desired Mean: {desired_mean}%
    Desired Std Dev: {desired_std}%
    
    By optimizing the truck fill rates to achieve the desired mean and standard deviation, we can enhance operational efficiency and ensure a more consistent material movement process.
    """
    return explanation

def main():
    st.title("Potential Improvements to Operational Efficiency with ShovelMetrics™ Payload Monitoring")
    st.markdown("Prepared for: Aktogay Mine")
    st.markdown("Date: 2024-04-17")
    st.markdown("## Introduction")
    st.markdown("The purpose of this analysis is to evaluate the potential improvements in operational efficiency with the implementation of ShovelMetrics™ Payload Monitoring (SM-PLM). By analyzing the truck fill distribution data, we aim to identify areas where optimizations can be made to enhance productivity and reduce operational risks.")
    st.markdown("To illustrate potential improvements with SM-PLM, the above distributions are shown in Figures 4 - 6 along with a target fill of 100% and a standard deviation of 5% to emulate the distribution with SM-PLM. The number of trucks is kept constant, and both curves share the same area and cumulative distribution.")
    
    # Add the rest of the code here...



    # Get all available shovels dynamically
    all_shovels = set()
    months = ['Load DetailApril2023',
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
              'Load DetailNovember16-30.2023']  

    for month in months:
        file_path = f'Cleaned_{month}.csv'
        month_data = pd.read_csv(file_path)
        if 'Shovel' in month_data.columns:
            all_shovels.update(set(month_data['Shovel']))

    all_shovels = sorted(list(all_shovels))

    # Dropdown for selecting shovel
    selected_shovel = st.sidebar.selectbox("Select Shovel", all_shovels)

    # Dropdowns for mean and standard deviation
    selected_mean = st.sidebar.slider("Select Mean (%)", 98, 110, 100, step=1)
    selected_std = st.sidebar.slider("Select Standard Deviation (%)", 1, 10, 5, step=1)

    # Plot distribution for selected shovel with selected mean and standard deviation
    shovel_fill_data = load_data(selected_shovel)
    plot_distribution(shovel_fill_data, selected_shovel, selected_mean, selected_std)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.express as px

# Define file paths for your CSV files
file_paths = [
    'Load DetailApril2023.csv',
    'Load DetailAugust2023.1-15.csv',
    'Load DetailSeptember2023.csv',
    'Load DetailAugust2023.16-31.csv',
    'Load DetailDecember1-15.2023.csv',
    'Load DetailDecember16-31.2023.csv',
    'Load DetailFebruary2023.csv',
    'Load DetailJanuary2023.csv',
    'Load DetailJuly2023.csv',
    'Load DetailJUNE2023.csv',
    'Load DetailMarch2023.csv',
    'Load DetailMay2023.csv',
    'Load DetailNovember1-15.2023.csv',
    'Load DetailNovember16-30.2023.csv'
]

def load_data(file_paths):
    all_data = pd.concat([pd.read_csv(file_path) for file_path in file_paths])
    all_data['Time Full'] = pd.to_datetime(all_data['Time Full'])
    all_data['Hour'] = all_data['Time Full'].dt.hour
    
    # Adjust the shift definition to 7 AM - 7 PM for Day, and 7 PM - 7 AM for Night
    all_data['Shift'] = all_data['Hour'].apply(lambda x: 'Day' if 7 <= x < 19 else 'Night')
    
    # Calculate 'Truck fill (%)' as the ratio of 'Tonnage' to 'Truck Factor', multiplied by 100 to get a percentage
    all_data['Truck fill (%)'] = (all_data['Tonnage'] / all_data['Truck Factor']) * 100
    
    # Filter data to cover only the period from 7 AM to 7 AM
    all_data = all_data[(all_data['Hour'] >= 7) | (all_data['Hour'] < 7)]
    
    return all_data


def create_plot(data):
    average_fill_by_hour_shift = data.groupby(['Hour', 'Shift'])['Truck fill (%)'].mean().reset_index()
    
    # Plot with adjusted colors
    fig = px.line(average_fill_by_hour_shift, x='Hour', y='Truck fill (%)', color='Shift',
                  labels={'Truck fill (%)': 'Average Truck Fill (%)'}, title='Average Truck Fill by Hour and Shift',
                  color_discrete_map={'Day': 'red', 'Night': 'blue'})
    
    # Add another x-axis for Night shift (7 PM - 7 AM)
    fig.update_layout(xaxis2=dict(dtick=1, title='Hour (Night Shift)', overlaying='x', side='top'))
    
    fig.update_xaxes(dtick=1)  # Adjusting tick frequency for better readability
    return fig

# Loading data and creating the plot
data = load_data(file_paths)
fig = create_plot(data)

# Displaying the plot in your Streamlit app
st.plotly_chart(fig)

# Additional analysis option
st.markdown(
"""
During day shifts, the average truck fill percentage is 94.89%, indicating efficient operations during daytime hours.Night shifts show slightly lower average truck fill percentages at 94.76%, but still demonstrate good operational efficiency.
Peak hours, from 10 AM to 4 PM, exhibit the highest average truck fill percentage at 94.92%, suggesting optimized operations during this period. Similarly, peak night hours, from 10 PM to 4 AM, maintain a high average truck fill percentage of 94.75%, indicating efficient operations during the night shift.

These insights collectively suggest that both day and night shifts are well-managed, with peak hours showing particularly high efficiency.
"""
)


import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# Load data
file_paths = [
    'Load DetailApril2023.csv',
    'Load DetailAugust2023.1-15.csv',
    'Load DetailSeptember2023.csv',
    'Load DetailAugust2023.16-31.csv',
    'Load DetailDecember1-15.2023.csv',
    'Load DetailDecember16-31.2023.csv',
    'Load DetailFebruary2023.csv',
    'Load DetailJanuary2023.csv',
    'Load DetailJuly2023.csv',
    'Load DetailJUNE2023.csv',
    'Load DetailMarch2023.csv',
    'Load DetailMay2023.csv',
    'Load DetailNovember1-15.2023.csv',
    'Load DetailNovember16-30.2023.csv'
]

@st.cache(allow_output_mutation=True)
def load_data(file_paths):
    data = pd.concat([pd.read_csv(file) for file in file_paths])
    return data

def main():
    st.subheader('Material Destination Analysis')

    # Load data for all shovels
    data = load_data(file_paths)

    # Create a unique widget ID for the multiselect widget
    widget_id = hash('select_shovels')

    # Create a multi-select dropdown for shovel selection in the sidebar
    with st.sidebar:
        st.write("Material Analysis")
        selected_shovels = st.multiselect("Select Shovels", ['All'] + data['Shovel'].unique().tolist(), default=['All'], key=widget_id)

    # Filter data for the selected shovels
    if 'All' in selected_shovels:
        shovel_data = data
    else:
        shovel_data = data[data['Shovel'].isin(selected_shovels)]

    # Drop rows with any NaN values in the dataframe to avoid errors in processing
    data_cleaned = shovel_data.dropna()

    # Filter and summarize tonnage for each destination category
    crusher_data = data_cleaned[data_cleaned['Assigned Dump'].str.contains('CRUSHER')]
    dlp_data = data_cleaned[data_cleaned['Assigned Dump'].str.contains('DLP')]
    stockpile_data = data_cleaned[~data_cleaned['Assigned Dump'].str.contains('CRUSHER|DLP', regex=True)]
    hg_data = data_cleaned[data_cleaned['Material'] == 'HG']
    lg_data = data_cleaned[data_cleaned['Material'].str.contains('LG')]

    # Calculate total tonnage for each category
    hg_total = hg_data['Tonnage'].sum()
    lg_total = lg_data['Tonnage'].sum()
    crusher_total = crusher_data['Tonnage'].sum()
    dlp_total = dlp_data['Tonnage'].sum()
    stockpile_total = stockpile_data['Tonnage'].sum()

    # Create a dictionary for destination tonnage
    destination_tonnage = {
        'High Grade (HG)': hg_total,
        'Low Grade (LG)': lg_total,
        'Crusher': crusher_total,
        'Acid Leach Pad (DLP)': dlp_total,
        'Stockpiles': stockpile_total
    }

    # Create pie chart
    fig_pie = go.Figure(data=[go.Pie(labels=list(destination_tonnage.keys()), values=list(destination_tonnage.values()), hole=0.3)])
    fig_pie.update_traces(textinfo='percent+label', textposition='inside')

    st.plotly_chart(fig_pie)

if __name__ == "__main__":
    main()

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

file_paths = [
    'Load DetailApril2023.csv',
    'Load DetailAugust2023.1-15.csv',
    'Load DetailSeptember2023.csv',
    'Load DetailAugust2023.16-31.csv',
    'Load DetailDecember1-15.2023.csv',
    'Load DetailDecember16-31.2023.csv',
    'Load DetailFebruary2023.csv',
    'Load DetailJanuary2023.csv',
    'Load DetailJuly2023.csv',
    'Load DetailJUNE2023.csv',
    'Load DetailMarch2023.csv',
    'Load DetailMay2023.csv',
    'Load DetailNovember1-15.2023.csv',
    'Load DetailNovember16-30.2023.csv'
]

def month_to_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

@st.cache(allow_output_mutation=True)
def load_data(file_paths):
    data = pd.concat([pd.read_csv(file) for file in file_paths])
    data['Time Full'] = pd.to_datetime(data['Time Full'])
    data['Hour'] = data['Time Full'].dt.hour
    data['Month'] = data['Time Full'].dt.month
    data['Season'] = data['Month'].apply(month_to_season)
    return data

data = load_data(file_paths)

# Create a unique widget ID for the selectbox widget
widget_id = hash('select_shovel')

# Create a dropdown for shovel selection in the sidebar
with st.sidebar:
    st.write("Monthly Truck Rates")
    selected_shovel = st.selectbox("Select Shovel", ['All'] + list(data['Shovel'].unique()), key=widget_id)

# Filter data for the selected shovel
if selected_shovel == 'All':
    shovel_data = data[(data['Truck Factor'] != 0) & (data['Tonnage'] != 0)].dropna(subset=['Tonnage', 'Truck Factor'])
else:
    shovel_data = data[(data['Shovel'] == selected_shovel) & (data['Tonnage'] != 0) & (data['Truck Factor'] != 0)].dropna(subset=['Truck Factor', 'Tonnage'])
shovel_data['Truck Fill Rate (%)'] = (shovel_data['Tonnage'] / shovel_data['Truck Factor']) * 100

# Monthly Performance
monthly_performance = shovel_data.groupby(['Season', 'Month'])['Truck Fill Rate (%)'].mean().reset_index()

# Assign colors to seasons
colors = {'Winter': '#1f77b4', 'Spring': '#ff7f0e', 'Summer': '#2ca02c', 'Fall': '#d62728'}

# Create a list of colors for each month based on its season
monthly_performance['Color'] = monthly_performance['Season'].map(colors)

# Define month names
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

# Monthly Trend Visualization
fig_monthly = go.Figure()
for season, color in colors.items():
    season_data = monthly_performance[monthly_performance['Season'] == season]
    fig_monthly.add_trace(go.Bar(x=season_data['Month'].map(month_names), y=season_data['Truck Fill Rate (%)'],
                                 name=f'{season} Average', marker_color=color))
    # Adding average line for each season with markers
    fig_monthly.add_trace(go.Scatter(x=season_data['Month'].map(month_names), y=[season_data['Truck Fill Rate (%)'].mean()] * len(season_data),
                                      mode='lines+markers', name=f'{season} Average', line=dict(color=color, dash='dash', width=2),
                                      marker=dict(color=color, size=8)))

fig_monthly.update_layout(xaxis_title='Month', yaxis_title='Average Truck Fill Rate (%)',
                          template='plotly_white', yaxis=dict(range=[80, 100]),
                          title=dict(text=f'Monthly Truck Fill Rate Trends for {selected_shovel}', font=dict(size=18, color='black', family="Arial")))
st.plotly_chart(fig_monthly)


