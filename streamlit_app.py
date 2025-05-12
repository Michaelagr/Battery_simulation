import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import locale
#import test_daily_load


# Streamlit config
st.set_page_config(page_title="Battery storage simulation dashboard", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f9f9f9;}
    .block-container {padding-top: 2rem;}
    h1, h2, h3 {color: #1f77b4;}
    .metric {background-color: #ffffff; padding: 10px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    .plot-box {border: 1px solid #ddd; padding: 10px; border-radius: 10px; background-color: #fff; margin-bottom: 1rem;}
    .metric-title {font-size: 1.8rem; color: #444;font-weight: bold}
    .metric-value {font-size: 1.7rem; color: #111}
    </style>
""", unsafe_allow_html=True)

# Set German locale for number formatting
try:
    locale.setlocale(locale.LC_ALL, 'de_DE.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'deu_deu')
    except locale.Error:
        st.write(" ")

# ADDED: This is the key change to prevent plots from opening in new tabs
plotly_config = {
    'displayModeBar': False,
    'showTips': False,
    'displaylogo': False,
    'scrollZoom': True,
    'staticPlot': False
}
#from graphs import demand_charge



# .metric-value {font-size: 1.5rem; color: #111; font-weight: bold;}
st.title("🔋 ecoplanet Battery Storage Simulation Dashboard")

#--------------------- Helper Definitions -------------------------
# Initial battery configuration
battery_efficiency = 0.9
discharge_percentage = 0.001
demand_charge = 200

# -------------------- Helper functions --------------------------
### Peak Shaving Simulation
def peak_shaving(load_data, threshold):
    peak_threshold = max(load_data) * (threshold / 100)
    optimized_load = np.where(load_data > peak_threshold, peak_threshold, load_data)
    return optimized_load



### New battery simulation
def battery_simulation_v02(df, battery_capacity, power_rating, depth_of_discharge, threshold_pct, battery_efficiency=1):
    total_capacity = battery_capacity  # kWh
    reserve_energy = total_capacity * (1 - depth_of_discharge / 100)  # minimum SoC (e.g., 20%) in kWh
    soc = total_capacity  # start fully charged in kWh
    interval_hours = 0.25  # 15-minute intervals
    peak = df["load"].max()
    threshold_kw = peak * (threshold_pct / 100)

    optimized = []
    discharge = []
    soc_state = []

    for load in df["load"]:
        grid_load = load  # start with original load

        # --- DISCHARGING ---
        if load > threshold_kw and soc > reserve_energy:
            power_needed = load - threshold_kw
            max_discharge_power = (soc - reserve_energy) / interval_hours
            actual_discharge_power = min(power_rating, power_needed, max_discharge_power)

            energy_used = actual_discharge_power * interval_hours / battery_efficiency
            # soc = max(soc - energy_used, reserve_energy)
            soc = soc - energy_used

            grid_load = load - actual_discharge_power
            discharge.append(actual_discharge_power)

        # --- CHARGING (only when load is below threshold to avoid peak increase) ---
        elif load <= threshold_kw and soc < total_capacity:

            max_possible_charge = threshold_kw - load  # Determine max possible charge power without exceeding the threshold

            max_charge_power = (total_capacity - soc) / interval_hours
            actual_charge_power = min(power_rating, max_charge_power, max_possible_charge)

            energy_stored = actual_charge_power * interval_hours * battery_efficiency
            soc = min(soc + energy_stored, total_capacity)

            grid_load = load + actual_charge_power
            discharge.append(0)

        else:
            discharge.append(0)

        optimized.append(grid_load)
        soc_state.append(soc)

    df["grid_load"] = optimized
    df["battery_discharge"] = discharge
    df["battery_soc"] = soc_state
    return df

#--------------------------------------------- FILE PROCESSING ------------------------------------------------------

############################################# File Processing #############################################
######### Uploader #########
with st.sidebar:
    st.subheader("File upload")
    uploaded_file = st.file_uploader("📁 Upload your load profile data (XLSX). Please make sure the file has two columns 'timestamp' and 'load'.", type=["xlsx"])
    ## Handling
    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        # Try to auto-detect datetime columns
        col_map = {col.lower(): col for col in df.columns}
        date_col = next((col for key, col in col_map.items() if any(kw in key for kw in ["date", "day", "tag", "datum"])), None)
        time_col = next((col for key, col in col_map.items() if any(kw in key for kw in ["time", "hour", "timestamp", "zeit","uhrzeit", "timestamps"])), None)
        load_col = next((col for key, col in col_map.items() if any(kw in key for kw in ["kw", "load", "value", "value_kw", "power", "entnahme"])), None)

        if date_col and time_col:
            df["timestamp"] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), format='%d.%m.%Y %H:%M')
        elif date_col:
            try:
                df["timestamp"] = pd.to_datetime(df[date_col], dayfirst=True)
            except Exception as e1:
                try:
                    df["timestamp"] = pd.to_datetime(df[date_col], format="mixed", dayfirst=True)
                except Exception as e2:
                    try:
                        df["timestamp"] = pd.to_datetime(df[date_col], format="ISO8601", dayfirst=True)
                    except Exception as e3:
                        raise ValueError(f"Failed to parse datetime. Errors:\n1. {e1}\n2. {e2}\n3. {e3}")
        elif time_col:
            df["timestamp"] = pd.to_datetime(df[time_col], dayfirst = True)
        else:
            df["timestamp"] = pd.to_datetime(df.iloc[:, 0], dayfirst = True)  # fallback

        if load_col:
            df["load"] = df[load_col]

        # Create date columns
        df["datetime"] = df["timestamp"]
        df["Date"] = df["timestamp"].dt.date
    #    df["week"] = df["timestamp"].dt.strftime("%G-W%V-%u")
        df["Week"] = df["timestamp"].dt.isocalendar().week
        df["Month"] = df["timestamp"].dt.strftime("%Y-%m")
        df["Year"] = df["timestamp"].dt.strftime("%Y")
        df["Weekday"] = df["timestamp"].dt.day_name()
        df["time"] = df["timestamp"].dt.time

        df_filter = df.copy()
        df_battery = df.copy()
        df_peaks = df.copy()

        with st.expander ("### 📊 Raw data preview"):
            st.write(df.head())


################################################################################ LOAD PROFILE INFORMATION ######################################################

tab1, tabsimulation = st.tabs(["Load profile overview", "Simplified battery sizing wizzard"])

with ((tab1)):
    if not uploaded_file:
        with st.container(border = True):
            st.subheader("⚠️ Please upload your load profile in the left sidebar.")
            st.write("**Ensure that the file contains two columns with a timestamp and the load in kW.**")
            st.write("**In case you encounter an error, ensure that the naming of the columns is 'timestamp' and 'load'.**")

    if uploaded_file:

        st.header("Load Profile Summary")

        st.subheader("📈 Statistics")
        st.write(f"From **{df["timestamp"].min()}** to **{df["timestamp"].max()}**")
        total_entries = len(df)
        avg_load = df["load"].mean()
        max_load = df["load"].max()
        min_load = df["load"].min()
        total_energy_kwh = df["load"].sum() *4 / 1000  # assuming 15-min intervals
        peak_load = df["load"].max()

        col1, col2, col3, col4, col5, col6 = st.columns(6)


        #col1.markdown("<div class='metric'><div class='metric-title'> 📅 Total datapoints </div><div class='metric-value'>" + f"{total_entries:,.0f}" + "</div></div>",unsafe_allow_html=True)
        col1.markdown(
            "<div class='metric'><div class='metric-title'> 🔋 Total Energy </div><div class='metric-value'>" + f"{total_energy_kwh:,.0f} MWh" + "</div></div>",
            unsafe_allow_html=True)
        col2.markdown(
            "<div class='metric'><div class='metric-title'> ⚡ Average Load </div><div class='metric-value'>" + f"{avg_load:,.2f} kW" + "</div></div>",
            unsafe_allow_html=True)
        col3.markdown(
            "<div class='metric'><div class='metric-title'> 🔺 Peak Load </div><div class='metric-value'>" + f"{peak_load:,.2f} kW" + "</div></div>",
            unsafe_allow_html=True)
        col4.markdown(
            "<div class='metric'><div class='metric-title'> 🔻 Min Load </div><div class='metric-value'>" + f"{min_load:.2f} kW" + "</div></div>",
            unsafe_allow_html=True)
        col5.markdown(
            "<div class='metric'><div class='metric-title'> 📈 Load factor </div><div class='metric-value'>" + f"{(avg_load / max_load * 100):.2f}%" + "</div></div>",
            unsafe_allow_html=True)


        #st.write("### 📊 Load profile visualization")
        st.write("\n")

        tab_agg, tab_toppeaks, tab_loadduration  = st.tabs(["Aggregated view", "Peak analysis", "Load duration curve"])

        with tab_agg:
            time_aggregation = st.radio(
                "Select view:",
                ["Total", "Monthly average", "Weekly average", "Daily average", "Hourly average"],
                horizontal=True
            )

            # Prepare the data based on selected aggregation
            if time_aggregation == "Total":
                # Full timeseries data
                fig_overview = px.line(df, x="timestamp", y="load",
                                       title="Complete Load Profile",
                                       labels={"timestamp": "Time", "load": "Power (kW)"})
                fig_overview.update_layout(height=400, xaxis_title="Time")


            elif time_aggregation == "Monthly average":
                # Group by month and calculate average
                monthly_avg = df.groupby("Month")["load"].mean().reset_index()
                # Convert month string back to datetime for better plotting
                monthly_avg["month_dt"] = pd.to_datetime(monthly_avg["Month"] + "-01")
                monthly_avg = monthly_avg.sort_values("month_dt")

                fig_overview = px.bar(monthly_avg, x="Month", y="load",
                                      title="Monthly Average Load",
                                      labels={"Month": "Month", "load": "Average Power (kW)"})
                fig_overview.update_layout(height=400, xaxis_title="Month", xaxis_tickangle=-45, xaxis=dict(nticks=12))


            elif time_aggregation == "Weekly average":
                # Extract ISO year and week directly
                df_iso = df.copy()
                df_iso[["iso_year", "iso_week"]] = df_iso["timestamp"].dt.isocalendar()[["year", "week"]]

                # Group by ISO year and week, compute average
                weekly_avg = df_iso.groupby(["iso_year", "iso_week"])["load"].mean().reset_index()
                # Create a datetime for the Monday of each ISO week
                weekly_avg["week_dt"] = pd.to_datetime(weekly_avg["iso_year"].astype(str) + "-W" +
                                                       weekly_avg["iso_week"].astype(str) + "-1", format="%G-W%V-%u")

                # Sort chronologically
                weekly_avg = weekly_avg.sort_values("week_dt")
                fig_overview = px.bar(weekly_avg, x="week_dt", y="load",
                                      title="Weekly load (average)",
                                      labels={"week_dt": "Week", "load": "Average Power (kW)"})
                fig_overview.update_layout(height=400, xaxis_title="Week", xaxis_tickangle=-45, xaxis_tickformat="%b W%W", xaxis=dict(nticks=52))


            elif time_aggregation == "Daily average":  # Daily Average

                # Grouped by day
                days_avg = df.groupby("Weekday")["load"].mean().reset_index()

                # Create sorting by days
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                days_avg['weekday_cat'] = pd.Categorical(days_avg['Weekday'], categories=weekday_order, ordered=True)
                days_avg = days_avg.sort_values('weekday_cat')
                days_avg = days_avg.drop('weekday_cat', axis=1)

                fig_overview = px.bar(days_avg, x="Weekday", y="load",
                                  title="Average load by weekday",
                                  labels={"Weekday": "Date", "load": "Average Power (kW)"})
                fig_overview.update_layout(height=400, xaxis_title="Days", xaxis_tickangle=-45)

            else:
                hourly_avg = df.groupby( ["time"])["load"].mean().reset_index()
#                weekly_avg = df_iso.groupby(["iso_year", "iso_week"])["load"].mean().reset_index()
                fig_overview = px.bar(hourly_avg, x="time", y="load",
                                      title="Average load (hour based) ",
                                      labels={"time": "Time", "load": "Average Power (kW)"})
                fig_overview.update_layout(height=400, xaxis_title="Week", xaxis_tickangle=-45, xaxis_tickformat="%b W%W", xaxis=dict(nticks=52))


            st.plotly_chart(fig_overview, use_container_width=True, config=plotly_config)

        with tab_toppeaks:
            st.subheader("🚩  Highest peaks in load profile")

            col1, col2, col3 = st.columns([2, 3, 5], vertical_alignment="top")
            with col1:
                with st.container(border = True):
                    # Let user select number of entries shown
                    n_peaks = st.number_input("Number of peaks to show", min_value=1, value=30)

                    top_peaks = df_peaks.nlargest(n_peaks, "load").reset_index()[["timestamp", "Weekday", "load"]]
                    st.metric("Highest value", f"{top_peaks['load'].max():,.0f} kW")
                    st.metric("Lowest value", f"{top_peaks['load'].min():,.0f} kW")

            with col2:

                st.write(top_peaks)

            with col3:
                fig_top20 = px.bar(top_peaks.sort_values("load"), x="timestamp", y="load",
                                   title=f"📊 Top {n_peaks} peak loads",
                                   labels={"Load": "Power (kW)", "Timestamp": "Time"})
                fig_top20.update_layout(xaxis_tickformat="%b")
                fig_top20.update_layout(xaxis_title="Timestamp", xaxis_tickangle=-45, xaxis=dict(nticks=20))
                st.plotly_chart(fig_top20, use_container_width=True)

            st.subheader("🔍 Explore context of specific peak")

            col_context_left, col_context_right, col_context_buffer = st.columns([1, 3, 5])
            with col_context_left:
                st.subheader(f"\n")
                # Let user select a specific peak timestamp
                st. write("Select a timestamp to explore context")
                selected_timestamp = st.selectbox("Select a timestamp to explore context", top_peaks["timestamp"].astype(str))
                # Convert back to datetime if needed
                selected_timestamp = pd.to_datetime(selected_timestamp)
                # Sort full df by timestamp
                df_sorted = df_peaks.copy()
                df_sorted = df_sorted.sort_values("timestamp").reset_index(drop=True)

                # Find index of selected timestamp in full sorted DataFrame
                selected_index = df_sorted[df_sorted["timestamp"] == selected_timestamp].index

                if not selected_index.empty:
                    idx = selected_index[0]
                    # Get 10 rows before and after
                    context_df = df_sorted.iloc[max(0, idx - 10): idx + 11]


                    with col_context_right:
                        st.subheader(f"📊 10 entries before and after selected peak (on {selected_timestamp})")
                       # st.dataframe(context_df[["date","Weekday","load"]])
                    #st.dataframe(context_df)
                        def highlight_selected_row(row):
                            if row["timestamp"] == selected_timestamp:
                                return ["background-color: #fdd835"] * len(row)  # yellow highlight
                            else:
                                return [""] * len(row)


                        # Apply the style
                        styled_context_df = context_df[["timestamp","Weekday","load"]]
                        styled_context_df = styled_context_df.style.apply(highlight_selected_row, axis=1)

                        # Show it in Streamlit
                        st.dataframe(styled_context_df)
                else:
                    st.warning("Selected timestamp not found in full data.")

            with col_context_buffer:
                selected_day = selected_timestamp.date()
                df_day = df_peaks[df_peaks["timestamp"].dt.date == selected_day]
                fig_day = px.line(df_day, x="timestamp", y="load", title=f"🔋 Load curve for selected day ({selected_day})")
                fig_day.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
                st.plotly_chart(fig_day, use_container_width=True)

        with tab_loadduration:
            st.subheader("📉  Load duration curve")
            sorted_loads = df_peaks["load"].sort_values(ascending=False).reset_index(drop=True)
            fig_load_duration = px.line(sorted_loads, title="🔺 Load Duration Curve")
            fig_load_duration.update_layout(yaxis_title="Power (kW)", xaxis_title="Hours (sorted by load)")
            st.plotly_chart(fig_load_duration, use_container_width=True)
            st.write("\n \n")





        # Add additional information about the load profile
        with st.expander("Additional load profile statistics"):
            st.write(f"**Data Points:** {len(df)}")
            st.write(f"**Date Range:** {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
            st.write(f"**Median Load:** {df['load'].median():.2f} kW")
            st.write(f"**Standard Deviation:** {df['load'].std():.2f} kW")
            st.write(f"**Load Factor:** {(avg_load / max_load * 100):.2f}%")


#################################################################################################################################################################


with tabsimulation:
    if uploaded_file:
        # Create test for battery
        df_exp = df.copy()
        st.header("🔋 📉 Battery Simulation Dashboard")
        st.warning("**Disclaimer**: This dashboard is intended for **indicative purposes only**. "
                   "All calculations and results are based on **simplified assumptions** and **should not be interpreted as precise or reliable forecasts.** "
                   "The tool provides no guarantee of accuracy, performance, or financial outcomes. Users must validate all assumptions independently and conduct a **thorough technical and financial analysis** before making any decisions based on the results shown. "
                   "The creators of this dashboard accept no liability for any actions taken based on its outputs.")

        # ----------- Battery configurations -----------------------
        battery_configs = {
            "Small": {"capacity": 90, "power": 92},
            "Medium": {"capacity": 215, "power": 100},
            "Large": {"capacity": 500, "power": 250}
        }

        if 'roi_battery_capacity' not in st.session_state:
            st.session_state.roi_battery_capacity = 215

        if 'roi_power_rating' not in st.session_state:
            st.session_state.roi_power_rating = 100

        if 'battery_cost_per_kwh' not in st.session_state:
            st.session_state.battery_cost_per_kwh = 400

        roi_battery_capacity = 0
        roi_power_rating = 0
        battery_cost_per_kwh = 220

        # Functions to update battery values
        def set_small():
            st.session_state.roi_battery_capacity = 90
            st.session_state.roi_power_rating = 92
            st.session_state.battery_cost_per_kwh = 400

        def set_medium():
            st.session_state.roi_battery_capacity = 215
            st.session_state.roi_power_rating = 100
            st.session_state.battery_cost_per_kwh = 220

        def set_large():
            st.session_state.roi_battery_capacity = 500
            st.session_state.roi_power_rating = 250
            st.session_state.battery_cost_per_kwh = 300

        def set_xlarge():
            st.session_state.roi_battery_capacity = 1000
            st.session_state.roi_power_rating = 500
            st.session_state.battery_cost_per_kwh = 280


        def set_custom():
            # This just activates the custom input section
            st.session_state.show_custom = True

        with st.container():
            cols = st.columns([1, 4])

            with cols[0]:

                with st.container(border=True):
                    st.markdown("### 🔧 Battery selection")

                    # Create simple buttons stacked vertically
                    st.button("Small (90 kWh / 92 kW)", on_click=set_small)
                    st.button("Medium (215 kWh / 100 kW)", on_click=set_medium)
                    st.button("Large (500 kWh / 250 kW)", on_click=set_large)
                    st.button("Extra large (1MWh / 500 kW)", on_click=set_xlarge)

                    if 'show_custom' not in st.session_state:
                        st.session_state.show_custom = False
                    st.button("Custom Configuration", on_click=set_custom)

                    if st.session_state.show_custom:
                        st.session_state.roi_battery_capacity = st.number_input(
                            "Custom Battery Capacity (kWh)",
                            min_value=1,
                            value=st.session_state.roi_battery_capacity
                        )

                        st.session_state.roi_power_rating = st.number_input(
                            "Custom Power Rating (kW)",
                            min_value=1,
                            value=st.session_state.roi_power_rating
                        )

                        # Display the current values
                    st.write(f"Seleted capacity: **{st.session_state.roi_battery_capacity} kWh**")
                    st.write(f"Selected power rating: **{st.session_state.roi_power_rating} kW**")
                    st.write("---")
                    st.write("**Battery cost assumptions**")
                    st.session_state.battery_cost_per_kwh = st.number_input("Battery cost per kWh (€/kWh)", min_value=100, value=st.session_state.battery_cost_per_kwh)
                    system_cost_multiplier = st.number_input("System cost multiplier", min_value=1.0, value=1.2)

                ## --------------------------------- PEAK REDUCTION --------------------------------------------------------------------
                with st.container(border=True):
                    st.markdown("### 🔺 Peak shaving threshold ")
                    st.write(f"Maximum peak reduction with selected battery: **{st.session_state.roi_power_rating}kW**")

                    value_peak_reduction = st.number_input("Peakshaving load (kW) ", 0, int(peak_load), st.session_state.roi_power_rating)

                    calculated_peakshaving_threshold = (peak_load - value_peak_reduction) / peak_load *100
                    st.write(f"Peak load reduction: {100-calculated_peakshaving_threshold:.2f}%")

                ## -------------------------------------------  FINANCIAL ASSUMPTIONS ----------------------------------------------------------------------------


                with st.container(border=True):
                    st.markdown("### 💰 Financial assumptions")
                    st.write("Peak shaving-related savings")
                    demand_charge_exp = st.number_input("Demand Charge (€/kW/year) ", min_value=1, value=200, help="Assumed annual savings for peak reduction.")

                ###############################################################################################################################

                # -----------------------------        PEAKSHAVING METRICS CALCULATION   -----------------------------------------------------

                df_exp["Optimized Load"] = peak_shaving(df_exp["load"].values, calculated_peakshaving_threshold)
                df_exp = battery_simulation_v02(df_exp, st.session_state.roi_battery_capacity,  st.session_state.roi_power_rating, 90, calculated_peakshaving_threshold)

                peak_before = df_exp["load"].max()
                peak_target = df_exp["Optimized Load"].max()
                peak_after = df_exp["grid_load"].max()

                peak_reduction_target = peak_before - peak_target
                peak_reduction_actual = peak_before - peak_after

                annual_savings_target = peak_reduction_target * demand_charge_exp
                annual_savings_actual = peak_reduction_actual * demand_charge_exp


                # --------------------------------   OTHER METRICS CALCULATION------------------------------------------------

                battery_lifetime = 10

                # total battery cost: cost per kWh * capacity *  system cost multiplier
                total_battery_cost = st.session_state.roi_battery_capacity * st.session_state.battery_cost_per_kwh * system_cost_multiplier
                payback_period = total_battery_cost / annual_savings_actual if annual_savings_actual > 0 else float("inf")
                payback_period_target = total_battery_cost / annual_savings_actual if annual_savings_actual > 0 else float("inf")
                roi = (annual_savings_actual * battery_lifetime - total_battery_cost) / total_battery_cost * 100


            with cols[1]:
                if (st.session_state.roi_battery_capacity == 0 and st.session_state.roi_power_rating == 0):
                    with st.container(border=True):
                        st.subheader("Please select a battery size")
                else:
                    if (peak_reduction_actual) == 0 :
                        st.warning("⚠️ The selected battery capacity is too low for succesful peakshaving with your load profile. \n "
                                   "Try using a battery with a higher capacity or adjusting the peakshaving threshold to a lower value.⚠️")
                    elif (peak_reduction_actual) < 0.98*st.session_state.roi_power_rating:
                        st.warning(f"Amount shaved: {peak_reduction_actual:,.0f}kW"
                                    "⚠️ The selected battery specifications are not sufficient for your load profile. \n "
                                   "Try using a battery with a higher capacity or adjusting the peakshaving threshold to a lower value.⚠️")


                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(
                            "<div class='metric'><div class='metric-title'>🔺 Peak load</div><div class='metric-value'>" + f"{peak_before:,.1f} kW" + "</div></div>",
                            unsafe_allow_html=True)
                    with col2:
                        st.markdown(
                            "<div class='metric'><div class='metric-title'>🔻 Reduced peak</div><div class='metric-value'>" + f"{peak_after:,.1f} kW" + "</div></div>",
                            unsafe_allow_html=True)
                    with col3:
                        st.markdown(
                            "<div class='metric'><div class='metric-title'>📉 Peak reduction</div><div class='metric-value'>" +
                            f"{(peak_reduction_actual / peak_before * 100):.1f}%" + "</div></div>",
                            unsafe_allow_html=True)
                    with col4:
                        st.markdown(
                            "<div class='metric'><div class='metric-title'>💸 Savings (p.a.)</div><div class='metric-value'>" + f"€{demand_charge_exp * (peak_before - df_exp["grid_load"].max()):,.1f}" + "</div></div>",
                            unsafe_allow_html=True,
                            help="Savings are calculated from peak reduction only, assuming a reduction of " + f"{((df_exp["load"].max() - df_exp["grid_load"].max())):,.1f} kW" + " and a peak demand charge of " + f"€/kW{demand_charge_exp:,.1f}" + " per kW per year.")

                    st.write("---")


                    fig_batt_exp = px.line(df_exp.reset_index(), x="timestamp",
                                           y=["load", "grid_load", "battery_discharge", "battery_soc"],
                                           title="⚡ Load, Optimized Load, and Battery Discharge with selected Battery",
                                           labels={"value": "Power (kW)", "timestamp": "Time",
                                                   "battery_discharge": "Battery discharge",
                                                   "soc_state": "Battery state of charge", "variable": "Legend"},
                                           line_shape='spline',
                                           color_discrete_map={"load": "#eb1b17", "grid_load": "#11a64c",
                                                               "battery_discharge": "#030ca8", "battery_soc": "#e64e02"})
                    fig_batt_exp.add_hline(y=df_exp["grid_load"].max(),
                                           line_dash="dot", line_color="orange",
                                           annotation_text=f"Achieved peak load ({df_exp["grid_load"].max():.0f}kW)",
                                           annotation_position="bottom right",
                                           name="Achieved peak load",  # Add this
                                           showlegend=True  # And this
                                           )

                    fig_batt_exp.add_hline(y=calculated_peakshaving_threshold*0.01* peak_before,
                                           line_dash="dot",
                                           line_color="blue",
                                           annotation_text=f"Target peak load ({calculated_peakshaving_threshold*0.01* peak_before:.1f}kW)",
                                           annotation_position="bottom left",
                                           name="Target peak load",
                                           showlegend=True
                                           )

                    fig_batt_exp.add_hline(y=df_exp["load"].max(),
                                           line_dash="dot", line_color="red",
                                           annotation_text="Original overall peak",
                                           annotation_position="top right",
                                           name="Original overall peak",
                                           showlegend=True
                                           )

                    fig_batt_exp.update_layout(height=500, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_batt_exp, use_container_width=True)

                    with st.container(border=True):


                        col1, col2, col3 = st.columns(3)
                        with col1:
                            col1.subheader("Load data")
                            st.metric("🔺 Original peak", f"{df["load"].max():,.0f} kW")
                            st.metric("🔸 Residual (Optimized) peak", f"{df_exp["grid_load"].max():,.0f} kW",
                                      f"{df_exp["grid_load"].max() - df["load"].max():,.0f} kW", delta_color="inverse")
                        with col2:
                            col2.subheader("Financials")
                            st.metric("💶 Estimated battery investment cost", f"~€ {total_battery_cost:,.0f}")
                            st.metric(f"💰 Estimated ann. savings from peak shaving (demand charge: {demand_charge_exp}€/kW)",
                                      f"€ {annual_savings_actual:,.1f}")
                        with col3:
                            col3.subheader("ROI")
                            st.metric("✅ Payback period", f"{payback_period:.1f} years")
                            st.metric(f"📈 ROI over {battery_lifetime} years (battery lifetime)", f"{roi:.1f}%")



                    #####################################   LOAD DURATION CURVE   ################################################

                    sorted_loads = df_exp["load"].sort_values(ascending=False).reset_index(drop=True)

                    st.write("---")

                    fig_ldc = px.line(sorted_loads, title="🔺 Load Duration Curve")
                    fig_ldc.add_hline(y=(peak_before - value_peak_reduction), line_dash="dot", line_color="red",
                                   annotation_text="Battery Threshold", annotation_position="bottom right")
                    fig_ldc.update_layout(yaxis_title="Power (kW)", xaxis_title="Hours (sorted by load)")
                    st.plotly_chart(fig_ldc, use_container_width=True)







