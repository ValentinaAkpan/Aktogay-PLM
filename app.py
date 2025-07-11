import streamlit as st
from data_loader import load_data
from data_processor import process_data, load_truck_fill_data
from plotter import (
    plot_fill_distribution,
    plot_timeseries,
    plot_destination_pie,
    plot_material_grade_pie,
    plot_monthly_trends,
)

st.set_page_config(page_title="ShovelMetrics™ PLM Analysis", layout="wide")

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
    processed_data = process_data(data)
    processed_data = processed_data[processed_data['Shovel'].isin(selected_shovels)]
    fill_data = processed_data[processed_data['Shovel'].isin(selected_shovels)]['Truck Fill (%)']
    fig = plot_fill_distribution(fill_data, selected_shovels, desired_mean, desired_std)
    if fig:
        st.plotly_chart(fig, use_container_width=True)

    # Time series plot
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
