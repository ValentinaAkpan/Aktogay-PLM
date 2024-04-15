import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Initialize an empty list to hold dataframes
dataframes = []

# Loop through the file paths, read each file into a dataframe, and append to the list
for file_path in file_paths:
    df = pd.read_csv(file_path)
    # Example preprocessing: convert 'Time Full' to datetime and calculate the performance metric
    df['Time Full'] = pd.to_datetime(df['Time Full'])
    df['Performance Metric'] = df['Tonnage'] / df['Truck Factor']
    dataframes.append(df)

# Concatenate all the dataframes into one
all_data = pd.concat(dataframes, ignore_index=True)

# Function to map the hour of the shift based on 'Time Full'
def map_to_shift_hour(row):
    hour = row['Time Full'].hour
    if 7 <= hour < 19:
        return hour - 7  # Shift starts at 7AM
    else:
        return (hour + 17) % 24  # Shift starts at 7PM

# Apply the function to the 'Time Full' column
all_data['Shift Hour'] = all_data.apply(map_to_shift_hour, axis=1)

# Function to calculate hourly performance
def calculate_hourly_performance(df):
    # Group by Shift Hour and calculate mean performance metric
    hourly_performance = df.groupby(['Shift Hour'])['Performance Metric'].mean()
    return hourly_performance

# Calculate hourly performance for both shifts
day_shift_data = all_data[(all_data['Time Full'].dt.hour >= 7) & (all_data['Time Full'].dt.hour < 19)]
night_shift_data = all_data[(all_data['Time Full'].dt.hour < 7) | (all_data['Time Full'].dt.hour >= 19)]

day_shift_performance = calculate_hourly_performance(day_shift_data)
night_shift_performance = calculate_hourly_performance(night_shift_data)

# Create a single plot for both shifts
fig = go.Figure()

# Add traces for both shifts
fig.add_trace(
    go.Scatter(x=day_shift_performance.index, y=day_shift_performance,
               name='Day Shift (7 AM to 7 PM)', mode='lines+markers', line=dict(color='red'))
)

fig.add_trace(
    go.Scatter(x=night_shift_performance.index, y=night_shift_performance,
               name='Night Shift (7 PM to 7 AM)', mode='lines+markers', line=dict(color='blue'))
)

# Update x-axis to start at 07:00 and end at 06:59
fig.update_layout(
    xaxis=dict(
        title='Hour of the Shift',
        tickvals=list(range(0, 24)),
        ticktext=[f'{hour:02d}:00' for hour in range(7, 24)] + [f'{hour:02d}:00' for hour in range(0, 7)]
    ),
    yaxis=dict(title='Truck Factor/Average Tonnage'),
    title='Hourly Performance: Truck Factor/Average Tonnage by Shift'
)

# Show figure
st.plotly_chart(fig)



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

    x_min = min(shovel_fill_data) * 0.9
    x_max = max(shovel_fill_data) * 1.1
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
        height=600,
        width=1200,
        legend=dict(
            x=1.05,
            y=0.5,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.5)'
        ),
    )

    st.plotly_chart(fig)

def main():
    st.title("Truck Fill Distribution Analysis")

    # Dynamic shovel extraction and loading data logic remains
    # Dynamic shovel extraction and loading data logic remains unchanged.
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

    # Dynamically get the list of shovels from the dataset
    shovels = set()
    for month in months:
        try:
            month_data = pd.read_csv(f'{path_to_csvs}Cleaned_{month}.csv')
            shovels.update(month_data['Shovel'].dropna().unique())
        except FileNotFoundError:
            st.error(f"File not found: Cleaned_{month}.csv")

    if not shovels:
        st.error("No shovel data available. Please check the dataset.")
        return

    selected_shovel = st.selectbox("Select a Shovel", sorted(list(shovels)))
    shovel_fill_data = load_data(selected_shovel)

    if shovel_fill_data:
        plot_distribution(shovel_fill_data, selected_shovel)
    else:
        st.write("No fill data available for the selected shovel. Please select a different shovel.")

if __name__ == "__main__":
    main()




def main():
    st.title("Truck Fill Distribution Analysis")

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
    selected_shovel = st.selectbox("Select Shovel", all_shovels)

    # Plot distribution for selected shovel
    shovel_fill_data = load_data(selected_shovel)
    plot_distribution(shovel_fill_data, selected_shovel)


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

# Read and concatenate CSV files
data = pd.concat([pd.read_csv(f) for f in file_paths])# Convert 'Time Full' to datetime and extract necessary time components
data['Time Full'] = pd.to_datetime(data['Time Full'])
data['Year'] = data['Time Full'].dt.year
data['Month'] = data['Time Full'].dt.month.apply(lambda x: calendar.month_abbr[x])
data['Hour'] = data['Time Full'].dt.hour

# Exclude rows with zero or NaN in 'Truck Factor' or 'Tonnage'
data = data[(data['Truck Factor'] != 0) & (data['Tonnage'] != 0)].dropna(subset=['Truck Factor', 'Tonnage'])

# Calculate 'Truck Fill Rate (%)' with two decimal places
data['Truck Fill Rate (%)'] = ((data['Tonnage'] / data['Truck Factor']) * 100).round(2)

# Prepare grouped data
hourly_performance = data.groupby(['Year', 'Month', 'Hour'])['Truck Fill Rate (%)'].mean().reset_index()
hourly_performance['AM/PM'] = hourly_performance['Hour'].apply(lambda x: 'AM' if x < 12 else 'PM')

# Define material categories
material_categories = ['ALP', 'LG', 'HG', 'Crusher']


import streamlit as st
import calendar
import pandas as pd
import plotly.graph_objects as go


st.title('Truck Fill Rate Analysis')

# Define all months and add "All" option
month_options = ['All'] + list(calendar.month_abbr[1:])
# Use st.multiselect to let the user pick months, including an "All" option
selected_months = st.multiselect('Select months:', options=month_options, default='All')

# Check if "All" is selected and adjust the selection accordingly
if 'All' in selected_months:
    selected_months = calendar.month_abbr[1:]  # Select all months if "All" is chosen

# Filter data for selected months
filtered_data = hourly_performance[hourly_performance['Month'].isin(selected_months)]

# Calculate total truck fill rate for the selected months
total_monthly_fill_rate = filtered_data['Truck Fill Rate (%)'].mean()

# Formatting fill rate for display with two decimal places for filtered data
filtered_data['Truck Fill Rate (%)'] = filtered_data['Truck Fill Rate (%)'].apply(lambda x: f'{x:.2f}%')

# Add a total row to the filtered data
total_row = pd.DataFrame({
    'Year': ['Total'],
    'Month': [''],
    'Hour': [''],
    'AM/PM': [''],
    'Truck Fill Rate (%)': [f'{total_monthly_fill_rate:.2f}%']  # Formatted total fill rate
})

filtered_data_with_total = pd.concat([filtered_data, total_row], ignore_index=True)

fig = go.Figure(data=[go.Table(
    header=dict(values=['Year', 'Month', 'Hour', 'AM/PM', 'Truck Fill Rate (%)'],
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[filtered_data_with_total.Year, filtered_data_with_total.Month, filtered_data_with_total.Hour,
                       filtered_data_with_total['AM/PM'], filtered_data_with_total['Truck Fill Rate (%)']],
               fill_color='lavender',
               align='left'))
])

if not filtered_data.empty:
    fig.update_layout(title='Hourly Performance: Truck Fill Rate by Month - {}'.format(', '.join(selected_months) if 'All' not in selected_months else 'All'),
                      height=min(800, (len(filtered_data_with_total) + 1) * 35))  # Adjust table height dynamically
else:
    fig.update_layout(title='No data selected', height=800)

st.plotly_chart(fig)

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

    # Create a multi-select dropdown for shovel selection
    all_shovels = ['All'] + data['Shovel'].unique().tolist()
    selected_shovels = st.multiselect("Select Shovels", all_shovels, default=['All'])

    # Filter data for the selected shovels
    if 'All' in selected_shovels:
        shovel_data = data
    else:
        shovel_data = data[data['Shovel'].isin(selected_shovels)]

    # Drop rows with any NaN values in the dataframe to avoid errors in processing
    data_cleaned = shovel_data.dropna()

    # Filter and summarize tonnage for each destination category
    crusher_tonnage = data_cleaned[data_cleaned['Assigned Dump'].str.contains('CRUSHER')]['Tonnage'].sum()
    dlp_tonnage = data_cleaned[data_cleaned['Assigned Dump'].str.contains('DLP')]['Tonnage'].sum()

    # For stockpiles, assuming everything not going to the crusher or DLP
    stockpile_tonnage = data_cleaned[~data_cleaned['Assigned Dump'].str.contains('CRUSHER|DLP', regex=True)]['Tonnage'].sum()

    # High Grade (HG) and Low Grade (LG) materials summary
    hg_tonnage = data_cleaned[data_cleaned['Material'] == 'HG']['Tonnage'].sum()
    lg_tonnage = data_cleaned[data_cleaned['Material'].str.contains('LG')]['Tonnage'].sum()

    # Creating a dictionary for destination tonnage
    destination_tonnage = {
        'High Grade (HG)': hg_tonnage,
        'Low Grade (LG)': lg_tonnage,
        'Crusher': crusher_tonnage,
        'Acid Leach Pad (DLP)': dlp_tonnage,
        'Stockpiles': stockpile_tonnage
    }

    # Create bar chart
    destination_fig = go.Figure(data=[go.Bar(
        x=list(destination_tonnage.keys()),
        y=list(destination_tonnage.values()),
        text=[f"{v:,.0f}" for v in destination_tonnage.values()],  # Format numbers with commas
        textposition='auto',
        marker_color=['#0693e3', '#8ed1fc', '#d62728', '#FFA07A', '#20B2AA'],  # Colors for each destination category
        name='Destination Tonnage'  # Legend name
    )])

    destination_fig.update_layout(
                           xaxis_title='Material Destination / Category',
                           yaxis_title='Tonnage',
                           legend=dict(title='Legend', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

    st.plotly_chart(destination_fig)

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

# Create a dropdown for shovel selection
selected_shovel = st.selectbox("Select Shovel", data['Shovel'].unique())

# Filter data for the selected shovel
shovel_data = data[(data['Shovel'] == selected_shovel) & (data['Truck Factor'] != 0) & (data['Tonnage'] != 0)].dropna(subset=['Truck Factor', 'Tonnage'])
shovel_data['Truck Fill Rate (%)'] = (shovel_data['Tonnage'] / shovel_data['Truck Factor']) * 100

# Seasonal Performance
seasonal_performance = shovel_data.groupby('Season')['Truck Fill Rate (%)'].mean().reset_index()

# Seasonal Trend Visualization
st.title(f'Seasonal Truck Fill Rate Trends for {selected_shovel}')
fig_seasonal = go.Figure(data=[go.Bar(x=seasonal_performance['Season'], y=seasonal_performance['Truck Fill Rate (%)'],
                                      text=seasonal_performance['Truck Fill Rate (%)'].apply(lambda x: f"{x:.2f}%"),
                                      textposition='auto',
                                      marker_color='#00B7F1')])
fig_seasonal.update_layout(xaxis_title='Season', yaxis_title='Average Truck Fill Rate (%)', template='plotly_white')
st.plotly_chart(fig_seasonal)


