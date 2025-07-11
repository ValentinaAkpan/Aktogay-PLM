import pandas as pd
import numpy as np
from scipy.stats import norm

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
