import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
import streamlit as st

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

def generate_markdown_explanation(actual_mean, actual_std, desired_mean, desired_std, shovel):
    explanation = f"""
    The purpose of this analysis is to evaluate the potential improvements in operational efficiency with the implementation of ShovelMetrics™ Payload Monitoring (SM-PLM). By analyzing the truck fill distribution data, we aim to identify areas where optimizations can be made to enhance productivity and reduce operational risks. To illustrate potential improvements with SM-PLM for shovel '{shovel}', the below distributions are shown with a target fill of {desired_mean}% and a standard deviation of {desired_std}% to emulate the distribution with SM-PLM.
    """
    return explanation

def main():
    st.title("Potential Improvements to Operational Efficiency with ShovelMetrics™ PLM")
    st.markdown("Prepared for: Aktogay Mine")
    st.markdown("Date: 2024-04-17")
    intro_placeholder = st.empty()  # Placeholder for the introductory text
    
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

    # Placeholder for the explanation text
    explanation_placeholder = st.empty()

    # Plot distribution for selected shovel with selected mean and standard deviation
    shovel_fill_data = load_data(selected_shovel)
    plot_distribution(shovel_fill_data, selected_shovel, selected_mean, selected_std)

    # Generate explanation text dynamically based on selected parameters
    actual_mean = np.mean(shovel_fill_data) if shovel_fill_data else 0
    actual_std = np.std(shovel_fill_data) if shovel_fill_data else 0
    explanation_text = generate_markdown_explanation(actual_mean, actual_std, selected_mean, selected_std, selected_shovel)
    explanation_placeholder.markdown(explanation_text)

 
if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
    all_data['Shift'] = all_data['Hour'].apply(lambda x: 'Day' if 7 <= x < 19 else 'Night')
    all_data['Truck fill (%)'] = (all_data['Tonnage'] / all_data['Truck Factor']) * 100
    return all_data

def adjust_hours_for_plot(df):
    # This shifts hours so that 7 AM starts at 0 (for easier plotting and logical grouping)
    df['Adjusted Hour'] = df['Hour'].apply(lambda x: (x + 17) % 24)
    df.sort_values(by=['Adjusted Hour'], inplace=True)
    return df

def create_plot(data):
    data = adjust_hours_for_plot(data)
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

data = load_data(file_paths)
fig = create_plot(data)

st.plotly_chart(fig)

# Analyzing the data
day_mean = data[data['Shift'] == 'Day']['Truck fill (%)'].mean()
night_mean = data[data['Shift'] == 'Night']['Truck fill (%)'].mean()
peak_day = data[(data['Shift'] == 'Day') & (data['Hour'].between(10, 16))]['Truck fill (%)'].mean()
peak_night = data[(data['Shift'] == 'Night') & ((data['Hour'] >= 22) | (data['Hour'] <= 4))]['Truck fill (%)'].mean()

st.write("During day shifts, the average truck fill percentage is {:.2f}%, indicating efficient operations during daytime hours. Night shifts show slightly lower average truck fill percentages at {:.2f}%,.".format(day_mean, night_mean))

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
    st.subheader('Material Analysis')

    # Load data for all shovels
    data = load_data(file_paths)

    # Create a unique widget ID for the multiselect widget
    widget_id = hash('select_shovels')

    # Create a multi-select dropdown for shovel selection in the sidebar
    with st.sidebar:
        st.write("Material Analysis")
        selected_shovels = st.multiselect("Select Shovels", ['All'] + data['Shovel'].unique().tolist(), default=['All'], key=widget_id)

    # Update title based on selected shovels
    if selected_shovels:
        selected_title = "Material Destination Distribution for " + ", ".join(selected_shovels)
    else:
        selected_title = "Material Destination Distribution for All Shovels"

    st.markdown(f'<h1 style="font-size: 24px;">{selected_title}</h1>', unsafe_allow_html=True)


    # Filter data for the selected shovels
    if 'All' in selected_shovels or not selected_shovels:
        shovel_data = data
    else:
        shovel_data = data[data['Shovel'].isin(selected_shovels)]

    # Drop rows with any NaN values in the dataframe to avoid errors in processing
    data_cleaned = shovel_data.dropna()

    # Filter and summarize tonnage for each destination category
    assigned_dump_data = data_cleaned['Assigned Dump'].value_counts()
    material_data = data_cleaned['Material'].value_counts()

    # Find highest dump and its count
    highest_dump = assigned_dump_data.idxmax()
    highest_dump_count = assigned_dump_data.max()
    # Find lowest dump and its count
    lowest_dump = assigned_dump_data.idxmin()
    lowest_dump_count = assigned_dump_data.min()

    # Find highest, second highest, and lowest grades
    material_grade_counts = material_data.sort_values(ascending=False)
    highest_grade = material_grade_counts.index[0]
    second_highest_grade = material_grade_counts.index[1]
    lowest_grade = material_grade_counts.index[-1]

    # Display report as a paragraph
    report_paragraph = f"The highest dump is {highest_dump}, occurring {highest_dump_count} times, and the lowest dump is {lowest_dump}, occurring {lowest_dump_count} times. The highest grade is {highest_grade}, followed by {second_highest_grade} as the second highest grade, and {lowest_grade} as the lowest grade."
    st.write(report_paragraph, style="font-size: 16px;")  # Adjusting font size

    # Create pie chart for Assigned Dump
    fig_assigned_dump = go.Figure(data=[go.Pie(labels=assigned_dump_data.index, values=assigned_dump_data.values, hole=0.3)])
    fig_assigned_dump.update_traces(textinfo='percent+label', textposition='inside')
    st.markdown(f'<h2 style="font-size: 20px;">Material Destination Distribution</h2>', unsafe_allow_html=True)
    st.plotly_chart(fig_assigned_dump)

    # Create pie chart for Material Grade
    fig_material_grade = go.Figure(data=[go.Pie(labels=material_data.index, values=material_data.values, hole=0.3)])
    fig_material_grade.update_traces(textinfo='percent+label', textposition='inside')
    st.markdown(f'<h2 style="font-size: 20px;">Material Grade Distribution</h2>', unsafe_allow_html=True)

    st.plotly_chart(fig_material_grade)

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

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm

def calculate_material_increase(current_mean, current_std, desired_mean=100, desired_std=5):
    # Compute z-scores for desired and current means
    z_current = (desired_mean - current_mean) / current_std
    z_desired = 0  # The desired mean is at the peak of the normal distribution (z-score = 0)
    
    # Compute the potential increase using CDF
    potential_increase = norm.cdf(z_current) - norm.cdf(z_desired)
    
    # Return the potential increase as a percentage
    return potential_increase * 100

# Load your truck fill rate data from the CSV files
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

def load_truck_fill_data():
    all_months_data = []
    total_improvement = 0
    
    for file_path in file_paths:
        month_data = pd.read_csv(file_path)
        if 'Tonnage' in month_data.columns and 'Truck Factor' in month_data.columns:
            # Add a "Month" column based on the timestamp
            month_data['Month'] = pd.to_datetime(month_data['Time Full']).dt.strftime('%B')  # Extract the full month name
            month_data['Year'] = pd.to_datetime(month_data['Time Full']).dt.year  # Extract the year
            
            current_truck_fill = (month_data['Tonnage'] / month_data['Truck Factor']) * 100  # Calculate truck fill rate for each row
            current_material_moved = month_data['Tonnage'].sum()
            desired_material_moved = current_material_moved * (1 + calculate_material_increase(current_truck_fill.mean(), 100, 100, 5) / 100)
            improvement = desired_material_moved - current_material_moved
            total_improvement += improvement
            
            all_months_data.append({
                'Month': month_data['Month'].iloc[0],  # Full month name
                'Year': month_data['Year'].iloc[0],
                'Current Truck Fill Rate': f"{current_truck_fill.mean():.2f}%",  # Format as percentage
                'Desired Truck Fill Rate': "100%",
                'Current Material': f"{current_material_moved:.2e}",  # Scientific notation
                'Desired Material': f"{desired_material_moved:.2e}",  # Scientific notation
                'Improvement': f"{improvement:.2e}"  # Scientific notation
            })
    
    # Add a total row
    total_row = {
        'Month': 'Total',
        'Year': '',
        'Current Truck Fill Rate': '', 
        'Desired Truck Fill Rate': '',
        'Current Material': f"{sum(float(month_data['Current Material']) for month_data in all_months_data):.2e}",
        'Desired Material': f"{sum(float(month_data['Desired Material']) for month_data in all_months_data):.2e}",
        'Improvement': f"{total_improvement:.2e}"  # Scientific notation
    }
    all_months_data.append(total_row)
    
    return all_months_data

# Load your truck fill rate data
all_months_data = load_truck_fill_data()

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(all_months_data)

# Sort the DataFrame by year and month
results_df['Month'] = pd.Categorical(results_df['Month'], categories=[
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December', 'Total'
], ordered=True)
results_df = results_df.sort_values(by=['Year', 'Month'])

# Display the results using Streamlit
st.markdown("</h3>Current and Desired Truck Fill Rates</h3>", unsafe_allow_html=True)

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

# Display the results using Streamlit
st.markdown("<h3><b>Current and Desired Truck Fill Rates</b></h3>", unsafe_allow_html=True)

st.table(results_df)


# Calculate total trucks
total_trucks = 0  # Initialize total number of trucks
for file_path in file_paths:
    month_data = pd.read_csv(file_path)
    if 'Truck' in month_data.columns:
        # Convert values in the "Truck" column to numeric, ignoring errors
        month_data['Truck'] = pd.to_numeric(month_data['Truck'], errors='coerce')
        # Sum up the numeric values in the "Truck" column
        total_trucks += month_data['Truck'].sum()

mean_fill = 100  # Desired mean fill rate is 100%

actual_material = 0  # Initialize actual material moved
desired_material = 0  # Initialize desired material that could be moved


actual_material = 0  # Initialize actual material moved
desired_material = 0  # Initialize desired material that could be moved

