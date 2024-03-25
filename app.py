import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st
import calendar


def load_data(shovel):
    path_to_csvs = './'
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

    shovel_fill_data = []

    for month in months:
        file_path = f'{path_to_csvs}Cleaned_{month}.csv'
        try:
            month_data = pd.read_csv(file_path)
            
            if 'Tonnage' in month_data.columns and 'Truck Factor' in month_data.columns and 'Shovel' in month_data.columns:
                # Ensure Truck Factor is not zero to avoid division by zero errors
                month_data = month_data[month_data['Truck Factor'] > 0]
                filtered_data = month_data[month_data['Shovel'] == shovel]
                # Calculate Truck Fill as Tonnage divided by Truck Factor, multiplied by 100 for percentage
                truck_fill_percentage = (filtered_data['Tonnage'] / filtered_data['Truck Factor']) * 100
                shovel_fill_data.extend(truck_fill_percentage.dropna())
          
        except FileNotFoundError:
            print(f"CSV file not found for {month}. Skipping...")

    return shovel_fill_data


def plot_distribution(shovel_fill_data, shovel, desired_mean=100, desired_std=5):
    actual_mean = np.mean(shovel_fill_data)
    actual_std = np.std(shovel_fill_data)

    # Dynamically determine the x-axis range with padding
    x_min = min(min(shovel_fill_data), desired_mean - 3 * desired_std)
    x_max = max(max(shovel_fill_data), desired_mean + 3 * desired_std)
    x_range_padding = (x_max - x_min) * 0.05  # 5% padding on each side
    x_range = np.linspace(x_min - x_range_padding, x_max + x_range_padding, 200)

    actual_distribution_y = norm.pdf(x_range, actual_mean, actual_std)
    desired_distribution_y = norm.pdf(x_range, desired_mean, desired_std)

    # Determine the maximum y-value for the y-axis range dynamically
    y_max = max(max(actual_distribution_y), max(desired_distribution_y)) * 1.1

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_range, y=actual_distribution_y, mode='lines', name=f'Actual Distribution for {shovel}', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_range, y=desired_distribution_y, mode='lines', name='Desired Distribution', line=dict(color='#00B7F1')))

    fig.add_trace(go.Scatter(x=[actual_mean, actual_mean], y=[0, max(actual_distribution_y)], mode='lines', name='Actual Mean', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=[desired_mean, desired_mean], y=[0, max(desired_distribution_y)], mode='lines', name='Desired Mean', line=dict(color='#00B7F1', dash='dash')))

    mean_std_text = (f"<b>Actual Mean:</b> {actual_mean:.2f}%<br>"
                     f"<b>Actual Std Dev:</b> {actual_std:.2f}%<br>"
                     f"<b>Desired Mean:</b> {desired_mean}%<br>"
                     f"<b>Desired Std Dev:</b> {desired_std}%")
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
        xaxis=dict(range=[x_min - x_range_padding, x_max + x_range_padding], dtick=10),
        yaxis=dict(range=[0, y_max], dtick=0.01),
        legend_title='Legend',
        margin=dict(r=250, t=100),  # Adjusted right margin for annotations
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

# Assuming 'hourly_performance' and 'data' DataFrames, along with 'material_categories', are defined above this code.

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


def main():
    st.subheader('Material Analysis')



    # Categorizing materials: HG, LG, and material directed to the Crusher
    hg_tonnage = data[data['Material'] == 'HG']['Tonnage'].sum()
    lg_tonnage = data[data['Material'].str.contains('LG')]['Tonnage'].sum()
    crusher_tonnage = data[data['Assigned Dump'] == 'CRUSHER']['Tonnage'].sum()

    # Creating a dictionary for material counts (tonnage in this context)
    material_tonnage = {
        'High Grade (HG)': hg_tonnage,
        'Low Grade (LG)': lg_tonnage,
        'Crusher': crusher_tonnage
    }

    # Create bar chart
    material_fig = go.Figure(data=[go.Bar(
        x=list(material_tonnage.keys()),
        y=list(material_tonnage.values()),
        text=[f"{v:,.0f}" for v in material_tonnage.values()],  # Format numbers with commas
        textposition='auto',
        marker_color=['#0693e3', '#8ed1fc', '#d62728'],  # Colors for each category
        name='Material Tonnage'  # Legend name
    )])

    material_fig.update_layout(
                           xaxis_title='Material Category',
                           yaxis_title='Tonnage',
                           legend=dict(title='Material Legend', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

    st.plotly_chart(material_fig)

if __name__ == "__main__":
    main()


import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title('Hourly Performance: Truck Fill Rate by Shift')

# Step 1: Read the CSV files and concatenate them into a single DataFrame
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

# Read each CSV file and concatenate them
@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.concat([pd.read_csv(file) for file in file_paths])
    return data

data = load_data()

# Convert 'Time Full' to datetime to extract the hour if needed
data['Time Full'] = pd.to_datetime(data['Time Full'])
data['Hour'] = data['Time Full'].dt.hour

# Drop rows where 'Truck Factor' or 'Tonnage' is zero or NaN
data = data[(data['Truck Factor'] != 0) & (data['Tonnage'] != 0)].dropna(subset=['Truck Factor', 'Tonnage'])

# Step 2: Calculate the Corrected Truck Fill Rate (%)
data['Truck Fill Rate (%)'] = (data['Tonnage'] / data['Truck Factor']) * 100

# Create a column for shifts
data['Shift'] = 'Day Shift'  # Initialize as Day Shift
data.loc[(data['Hour'] >= 19) | (data['Hour'] < 7), 'Shift'] = 'Night Shift'

# Step 3: Group by Hour and Shift and calculate mean corrected truck fill rate
hourly_performance = data.groupby(['Hour', 'Shift'])['Truck Fill Rate (%)'].mean().reset_index()

# Step 4: Visualize with Plotly
fig = go.Figure()

# Add the lines for Truck Fill Rate (%) for each shift
for shift in data['Shift'].unique():
    shift_data = hourly_performance[hourly_performance['Shift'] == shift]
    fig.add_trace(go.Scatter(x=shift_data['Hour'], 
                             y=shift_data['Truck Fill Rate (%)'], 
                             mode='lines+markers',
                             name=shift,
                             marker=dict(color='Green' if shift == 'Day Shift' else 'Blue')))

# Add layout details
fig.update_layout(title='Hourly Performance: Truck Fill Rate by Shift',
                  xaxis_title='Hour of the Day',
                  yaxis_title='Truck Fill Rate (%)',  # Display as percentage
                  yaxis_tickformat='.0f',  # Specify y-axis tick format as percentage without decimal places
                  legend_title='Shift',
                  template='plotly_white',
                  xaxis=dict(
                      tickvals=list(range(0, 24)),
                      ticktext=[f"{i} AM" if i < 12 else f"{i-12} PM" if i > 12 else "12 PM" for i in range(24)]
                  ))

# Show the figure
st.plotly_chart(fig)

# Find the best and worst performing hours for each shift
best_hours = hourly_performance.groupby('Shift').apply(lambda x: x.nlargest(3, 'Truck Fill Rate (%)')).reset_index(drop=True)
worst_hours = hourly_performance.groupby('Shift').apply(lambda x: x.nsmallest(3, 'Truck Fill Rate (%)')).reset_index(drop=True)

# Generate analysis report
analysis_report = """
 Analysis Report:
----------------------

Key Findings:
1. Best Performing Hours:
{best_hours}

2. Worst Performing Hours:
{worst_hours}
"""

# Format best hours information
best_hours_info = ""
for shift, hours in best_hours.groupby('Shift'):
    best_hours_info += f"- {shift}:\n"
    for i, hour_data in hours.iterrows():
        best_hours_info += f"  - Hour {hour_data['Hour']}: {hour_data['Truck Fill Rate (%)']:.2f}%\n"
    best_hours_info += "\n"

# Format worst hours information
worst_hours_info = ""
for shift, hours in worst_hours.groupby('Shift'):
    worst_hours_info += f"- {shift}:\n"
    for i, hour_data in hours.iterrows():
        worst_hours_info += f"  - Hour {hour_data['Hour']}: {hour_data['Truck Fill Rate (%)']:.2f}%\n"
    worst_hours_info += "\n"

# Display the analysis report
st.markdown(analysis_report.format(best_hours=best_hours_info, worst_hours=worst_hours_info))

