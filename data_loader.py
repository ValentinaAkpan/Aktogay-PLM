
import pandas as pd
import streamlit as st

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
            df = pd.read_csv(f'./data/{file}.csv')  # Assumes CSVs are in data/ directory
            if all(col in df for col in ['Tonnage', 'Truck Factor', 'Shovel']):
                df = df[df['Truck Factor'] > 0]
                data.append(df)
        except FileNotFoundError:
            st.warning(f"CSV file not found: {file}")
    return data
