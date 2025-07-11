import pandas as pd
import streamlit as st

CSV_FILES = [
    'Cleaned_Load DetailJanuary2023',
    'Cleaned_Load DetailFebruary2023',
    'Cleaned_Load DetailMarch2023',
    'Cleaned_Load DetailApril2023',
    'Cleaned_Load DetailMay2023',
    'Cleaned_Load DetailJUNE2023',
    'Cleaned_Load DetailJuly2023',
    'Cleaned_Load DetailAugust2023.1-15',
    'Cleaned_Load DetailAugust2023.16-31',
    'Cleaned_Load DetailSeptember2023',
    'Load DetailOctober2023',  # Included since it exists
    'Cleaned_Load DetailNovember1-15.2023',
    'Cleaned_Load DetailNovember16-30.2023',
    'Cleaned_Load DetailDecember1-15.2023',
    'Cleaned_Load DetailDecember16-31.2023'
]

def load_data():
    """Load and filter CSV data, ensuring required columns exist."""
    data = []
    for file in CSV_FILES:
        try:
            df = pd.read_csv(f'./{file}.csv')  # Changed path to root directory
            if all(col in df for col in ['Tonnage', 'Truck Factor', 'Shovel']):
                df = df[df['Truck Factor'] > 0]
                data.append(df)
            else:
                st.warning(f"File {file}.csv missing required columns")
        except FileNotFoundError:
            st.warning(f"CSV file not found: {file}")
    return data
