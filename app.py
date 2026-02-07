import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy import stats
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.io as pio

# =====================================================
# Page config
# =====================================================
st.set_page_config(page_title="Wind EVA & Hindcasting", layout="wide")
st.title("🌬️ Wind Extreme Value Analysis & Wave Hindcasting")

# =====================================================
# Constants
# =====================================================
DIRECTION_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 1000]

# Beaufort Scale bins (m/s)
BEAUFORT_BINS = [0, 0.5, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7, 100]
BEAUFORT_LABELS = [
    "0: Calm", "1: Light air", "2: Light breeze", "3: Gentle breeze",
    "4: Moderate breeze", "5: Fresh breeze", "6: Strong breeze",
    "7: Near gale", "8: Gale", "9: Strong gale", "10: Storm",
    "11: Violent storm", "12: Hurricane"
]

# =====================================================
# Helper functions
# =====================================================
def compute_wind_speed(u, v):
    return np.sqrt(u**2 + v**2)

def wind_dir_from(u, v):
    return (270 - np.degrees(np.arctan2(v, u))) % 360

def degree_to_compass(deg):
    idx = int((deg + 22.5) // 45) % 8
    return DIRECTION_ORDER[idx]

def rmse(obs, sim):
    return np.sqrt(np.mean((obs - sim) ** 2))

def get_rl(u):
    if 0 <= u < 1: return 1.85
    elif u < 2: return 1.75
    elif u < 3: return 1.65
    elif u < 4: return 1.55
    elif u < 5: return 1.45
    elif u < 6: return 1.40
    elif u < 7: return 1.35
    elif u < 8: return 1.25
    elif u < 9: return 1.20
    elif u < 10: return 1.15
    elif u < 11: return 1.10
    elif u < 12: return 1.08
    elif u < 13: return 1.06
    elif u < 14: return 1.04
    elif u < 15: return 1.02
    elif u < 16: return 1.00
    elif u < 17: return 0.98
    elif u < 18: return 0.95
    else: return 0.90

def roundup_excel(x, ndigits=1):
    """Round up to specified decimal places like Excel ROUNDUP"""
    if np.isnan(x) or np.isinf(x):
        return 0.0
    factor = 10**ndigits
    return math.ceil(x * factor) / factor

def create_windrose(df, use_beaufort=False, bin_interval=1.0, max_range=None):
    """
    Create a windrose plot using Plotly
    
    Parameters:
    - df: DataFrame with 'speed' and 'direction' columns
    - use_beaufort: If True, use Beaufort scale bins; if False, use custom bins
    - bin_interval: Interval for custom bins (m/s)
    - max_range: Maximum radial axis range for consistency across plots (%)
    """
    
    # Prepare bins
    if use_beaufort:
        bins = BEAUFORT_BINS
        labels = BEAUFORT_LABELS
    else:
        max_speed = df['speed'].max()
        bins = list(np.arange(0, max_speed + bin_interval, bin_interval))
        labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f} m/s" for i in range(len(bins)-1)]
    
    # Bin the speeds
    df_rose = df.copy()
    df_rose['speed_bin'] = pd.cut(df_rose['speed'], bins=bins, labels=labels, include_lowest=True)
    
    # Count occurrences by direction and speed bin
    rose_data = df_rose.groupby(['direction', 'speed_bin']).size().unstack(fill_value=0)
    
    # Reindex to ensure all directions are present
    rose_data = rose_data.reindex(DIRECTION_ORDER, fill_value=0)
    
    # Calculate percentages
    total_count = len(df_rose)
    rose_pct = (rose_data / total_count * 100)
    
    # Create polar bar chart
    fig = go.Figure()
    
    # Direction angles for polar plot
    dir_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # Add traces for each speed bin - Rainbow colors (blue to red)
    n_bins = len(rose_pct.columns)
    colors = []
    for i in range(n_bins):
        # Create rainbow gradient: blue -> cyan -> green -> yellow -> orange -> red
        ratio = i / max(n_bins - 1, 1)
        if ratio < 0.2:  # Blue to Cyan
            r, g, b = 0, int(255 * ratio / 0.2), 255
        elif ratio < 0.4:  # Cyan to Green
            r, g, b = 0, 255, int(255 * (1 - (ratio - 0.2) / 0.2))
        elif ratio < 0.6:  # Green to Yellow
            r, g, b = int(255 * (ratio - 0.4) / 0.2), 255, 0
        elif ratio < 0.8:  # Yellow to Orange
            r, g, b = 255, int(255 * (1 - (ratio - 0.6) / 0.2)), 0
        else:  # Orange to Red
            r, g, b = 255, 0, 0
        colors.append(f'rgb({r},{g},{b})')
    
    for i, speed_bin in enumerate(rose_pct.columns):
        fig.add_trace(go.Barpolar(
            r=rose_pct[speed_bin].values,
            theta=DIRECTION_ORDER,
            name=str(speed_bin),
            marker_color=colors[i % len(colors)],
            hovertemplate='<b>%{theta}</b><br>' +
                         f'{speed_bin}<br>' +
                         '%{r:.2f}%<extra></extra>'
        ))
    
    # Determine max range for radial axis
    if max_range is None:
        max_range = rose_pct.sum(axis=1).max() * 1.1  # 10% padding
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                ticksuffix='%',
                angle=90,
                range=[0, max_range],  # Set consistent range
                tickfont=dict(color='black', size=11),
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                direction='clockwise',
                rotation=90,
                tickfont=dict(color='black', size=12)
            ),
            bgcolor='white'
        ),
        title=dict(
            text="Wind Rose",
            x=0.5,
            xanchor='center',
            font=dict(color='black', size=16)
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="Wind Speed", font=dict(color='black', size=12)),
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,  # Back to original position for consistent export
            font=dict(color='black', size=10),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        height=600,
        width=1000,  # Fixed width for consistent export
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black')
    )
    
    return fig, rose_pct

def create_waverose(df, bin_interval=0.5, fetch_dict=None, max_range=None):
    """
    Create a waverose plot using Plotly
    
    Parameters:
    - df: DataFrame with 'wave_height' and 'direction' columns
    - bin_interval: Interval for wave height bins (m)
    - fetch_dict: Dictionary of fetch values by direction (to exclude zero-fetch directions from data)
    - max_range: Maximum radial axis range for consistency across plots (%)
    """
    
    # Filter out directions with zero fetch if fetch_dict is provided
    df_filtered = df.copy()
    if fetch_dict is not None:
        valid_directions = [d for d in DIRECTION_ORDER if fetch_dict.get(d, 0) > 0]
        df_filtered = df_filtered[df_filtered['direction'].isin(valid_directions)]
    
    # Prepare bins
    if len(df_filtered) > 0:
        max_wave = df_filtered['wave_height'].max()
    else:
        max_wave = 1.0  # Default if no data
    
    bins = list(np.arange(0, max_wave + bin_interval, bin_interval))
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f} m" for i in range(len(bins)-1)]
    
    # Bin the wave heights
    if len(df_filtered) > 0:
        df_rose = df_filtered.copy()
        df_rose['wave_bin'] = pd.cut(df_rose['wave_height'], bins=bins, labels=labels, include_lowest=True)
        
        # Count occurrences by direction and wave bin
        rose_data = df_rose.groupby(['direction', 'wave_bin']).size().unstack(fill_value=0)
    else:
        # Create empty data structure
        rose_data = pd.DataFrame(0, index=[], columns=labels if labels else ['0.0-0.5 m'])
    
    # Reindex to ALL 8 directions (this ensures all directions show on plot)
    rose_data = rose_data.reindex(DIRECTION_ORDER, fill_value=0)
    
    # Calculate percentages
    total_count = len(df_filtered) if len(df_filtered) > 0 else 1
    rose_pct = (rose_data / total_count * 100)
    
    # Create polar bar chart
    fig = go.Figure()
    
    # Rainbow colors (blue to red) for wave heights
    n_bins = len(rose_pct.columns)
    colors = []
    for i in range(n_bins):
        # Create rainbow gradient: blue -> cyan -> green -> yellow -> orange -> red
        ratio = i / max(n_bins - 1, 1)
        if ratio < 0.2:  # Blue to Cyan
            r, g, b = 0, int(255 * ratio / 0.2), 255
        elif ratio < 0.4:  # Cyan to Green
            r, g, b = 0, 255, int(255 * (1 - (ratio - 0.2) / 0.2))
        elif ratio < 0.6:  # Green to Yellow
            r, g, b = int(255 * (ratio - 0.4) / 0.2), 255, 0
        elif ratio < 0.8:  # Yellow to Orange
            r, g, b = 255, int(255 * (1 - (ratio - 0.6) / 0.2)), 0
        else:  # Orange to Red
            r, g, b = 255, 0, 0
        colors.append(f'rgb({r},{g},{b})')
    
    for i, wave_bin in enumerate(rose_pct.columns):
        fig.add_trace(go.Barpolar(
            r=rose_pct[wave_bin].values,
            theta=DIRECTION_ORDER,  # All 8 directions always shown
            name=str(wave_bin),
            marker_color=colors[i % len(colors)],
            hovertemplate='<b>%{theta}</b><br>' +
                         f'{wave_bin}<br>' +
                         '%{r:.2f}%<extra></extra>'
        ))
    
    # Determine max range for radial axis
    if max_range is None:
        max_range = rose_pct.sum(axis=1).max() * 1.1  # 10% padding
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                ticksuffix='%',
                angle=90,
                range=[0, max_range],  # Set consistent range
                tickfont=dict(color='black', size=11),
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                direction='clockwise',
                rotation=90,
                tickfont=dict(color='black', size=12)
            ),
            bgcolor='white'
        ),
        title=dict(
            text="Wave Rose",
            x=0.5,
            xanchor='center',
            font=dict(color='black', size=16)
        ),
        showlegend=True,
        legend=dict(
            title=dict(text="Wave Height (Hs)", font=dict(color='black', size=12)),
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.02,  # Back to original position for consistent export
            font=dict(color='black', size=10),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        ),
        height=600,
        width=1000,  # Fixed width for consistent export
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black')
    )
    
    return fig, rose_pct

def create_monthly_windroses(df, use_beaufort=False, bin_interval=1.0, max_range=None):
    """
    Create wind rose plots for each month
    
    Returns:
    - Dictionary of {month_name: (fig, data)}
    """
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    monthly_roses = {}
    
    for month_num in range(1, 13):
        df_month = df[df['month'] == month_num]
        
        if len(df_month) == 0:
            continue
        
        # Create rose for this month with consistent max_range
        fig, data = create_windrose(df_month, use_beaufort, bin_interval, max_range)
        
        # Update title to include month
        fig.update_layout(
            title=dict(
                text=f"Wind Rose - {month_names[month_num - 1]}",
                x=0.5,
                xanchor='center'
            )
        )
        
        monthly_roses[month_names[month_num - 1]] = (fig, data)
    
    return monthly_roses

def create_monthly_waveroses(df, bin_interval=0.5, fetch_dict=None, max_range=None):
    """
    Create wave rose plots for each month
    
    Returns:
    - Dictionary of {month_name: (fig, data)}
    """
    month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    monthly_roses = {}
    
    for month_num in range(1, 13):
        df_month = df[df['month'] == month_num]
        
        if len(df_month) == 0:
            continue
        
        # Create rose for this month with consistent max_range
        fig, data = create_waverose(df_month, bin_interval, fetch_dict, max_range)
        
        # Update title to include month
        fig.update_layout(
            title=dict(
                text=f"Wave Rose - {month_names[month_num - 1]}",
                x=0.5,
                xanchor='center'
            )
        )
        
        monthly_roses[month_names[month_num - 1]] = (fig, data)
    
    return monthly_roses


# =====================================================
# Initialize session state
# =====================================================
for k in [
    "processed","df","df_hindcast","annual_max","annual_max_wave",
    "eva_table","eva_table_wave","best_fit_df","best_fit_wave_df",
    "windrose_fig", "windrose_data", "waverose_fig", "waverose_data",
    "monthly_windroses", "monthly_waveroses"
]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "processed" else False

# =====================================================
# Sidebar input
# =====================================================
st.sidebar.header("⏱ Time Settings")
start_date = st.sidebar.date_input("Start date", value=datetime(2005, 1, 1))
start_hour = st.sidebar.selectbox("Start hour", [f"{h:02d}:00" for h in range(24)])
interval_hours = st.sidebar.number_input("Time interval (hours)", 1, step=1)

# Rose plot settings
st.sidebar.header("🌹 Rose Plot Settings")
enable_rose = st.sidebar.checkbox("Generate Rose Plots", value=False)

if enable_rose:
    st.sidebar.subheader("Wind Rose Binning")
    wind_bin_type = st.sidebar.radio(
        "Wind speed binning method:",
        options=["User Defined", "Beaufort Scale"],
        index=0
    )
    
    if wind_bin_type == "User Defined":
        wind_bin_interval = st.sidebar.number_input(
            "Wind speed bin interval (m/s)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
    else:
        wind_bin_interval = None  # Will use Beaufort scale
    
    st.sidebar.subheader("Wave Rose Binning")
    wave_bin_interval = st.sidebar.number_input(
        "Wave height bin interval (m)",
        min_value=0.1,
        max_value=5.0,
        value=0.5,
        step=0.1
    )

# =====================================================
# Data input
# =====================================================
col1, col2 = st.columns(2)
with col1:
    raw_text = st.text_area("📋 Paste u10 v10", height=220)
with col2:
    fetch_text = st.text_area("🧭 Paste Fetch (km)", height=220)

run_btn = st.button("🚀 Process & Run Analysis")

# =====================================================
# Main processing
# =====================================================
if run_btn and raw_text.strip():
    try:
        # -------------------------
        # Parse fetch
        # -------------------------
        fetch_dict = {d: 0.0 for d in DIRECTION_ORDER}
        for line in fetch_text.strip().splitlines():
            p = line.split()
            if len(p) >= 2 and p[0].upper() in fetch_dict:
                fetch_dict[p[0].upper()] = float(p[1])

        fetch = np.array([fetch_dict[d] for d in DIRECTION_ORDER])
        fetch = np.where(fetch <= 200, fetch, 200)

        # -------------------------
        # Parse wind
        # -------------------------
        rows = []
        for line in raw_text.strip().splitlines():
            p = line.split()
            if len(p) >= 2:
                rows.append([float(p[0]), float(p[1])])

        df = pd.DataFrame(rows, columns=["u10", "v10"])

        # -------------------------
        # Time index
        # -------------------------
        start_dt = datetime.combine(start_date, datetime.strptime(start_hour, "%H:%M").time())
        df["datetime"] = [start_dt + timedelta(hours=i * interval_hours) for i in range(len(df))]
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        df["day"] = df["datetime"].dt.day
        df["time"] = df["datetime"].dt.strftime("%H:%M")

        # -------------------------
        # Wind
        # -------------------------
        df["speed"] = compute_wind_speed(df["u10"], df["v10"])
        df["dir_deg"] = wind_dir_from(df["u10"], df["v10"])
        df["direction"] = df["dir_deg"].apply(degree_to_compass)

        # -------------------------
        # Hindcasting (CORRECTED)
        # -------------------------
        feff = np.array([fetch[DIRECTION_ORDER.index(d)] for d in df["direction"]])

        u_stab = df["speed"].values * 1.1
        rl = np.array([get_rl(u) for u in u_stab])
        u_a = 0.71 * (u_stab * rl) ** 1.23
        df["u_a"] = u_a

        duration_h = np.ones(len(df))
        for i in range(1, len(df)):
            same_dir = df["direction"].iloc[i] == df["direction"].iloc[i - 1]
            # ✅ CORRECTED: Only check direction like Excel, not u_a
            duration_h[i] = duration_h[i - 1] + 1 if same_dir else 1
            
        duration_s = duration_h * 3600
        g = 9.81

        # Initialize arrays to store results for ALL timesteps
        hs = []
        tp = []
        ts = []
        status = []
        time_critical_list = []

        for i in range(len(df)):

            # Safety guard for zero/negative fetch or wind
            if feff[i] <= 0 or u_a[i] <= 0:
                hs.append(0.0)
                tp.append(0.0)
                ts.append(0.0)
                status.append("Fetch Limited")
                time_critical_list.append(0.0)
                continue

            fetcheff = float(feff[i] * 1000)  # km → m, ensure float
            u_a_val = float(u_a[i])  # ensure float
            duration_s_val = float(duration_s[i])  # ensure float

            # ✅ Calculate time_critical with overflow protection
            try:
                # Calculate components separately to avoid overflow
                term1 = 68.8 * u_a_val / g
                term2_base = (g * fetcheff) / (u_a_val ** 2)
                
                # Check for reasonable values before exponentiation
                if term2_base <= 0 or term1 <= 0:
                    time_critical = 0.0
                    cek_time_crit = 0.0
                elif term2_base > 1e10:  # Prevent overflow
                    time_critical = 1e10  # Cap at large value
                    cek_time_crit = 1e10
                else:
                    term2 = term2_base ** (2/3)
                    time_critical = term1 * term2
                    cek_time_crit = (g * time_critical) / u_a_val
                    
            except (OverflowError, ValueError):
                time_critical = 1e10
                cek_time_crit = 1e10

            if cek_time_crit > 71500:
                # Fully Developed Sea
                hs_val = 0.2433 * (u_a_val ** 2) / g
                tp_val = 8.132 * u_a_val / g  # ✅ Corrected to match Excel
                ts_val = 0.95 * tp_val
                stat = "Fully Developed Sea"

            else:
                if duration_s_val < time_critical:
                    # Duration Limited
                    try:
                        fmin = (
                            (u_a_val ** 2) / g
                            * ((g * duration_s_val) / (u_a_val * 68.8)) ** (3 / 2)
                        )
                    except (OverflowError, ValueError):
                        fmin = fetcheff  # fallback to fetch limited

                    hs_val = (
                        0.0016 * (u_a_val ** 2) / g
                        * ((g * fmin) / (u_a_val ** 2)) ** 0.5
                    )

                    tp_val = (
                        0.2857
                        * ((g * fmin) / u_a_val ** 2) ** (1 / 3)
                        * (u_a_val / g)
                    )

                    ts_val = 0.95 * tp_val
                    stat = "Duration Limited"

                else:
                    # Fetch Limited
                    hs_val = (
                        0.0016 * (u_a_val ** 2) / g
                        * ((g * fetcheff) / (u_a_val ** 2)) ** 0.5
                    )

                    tp_val = (
                        0.2857
                        * ((g * fetcheff) / u_a_val ** 2) ** (1 / 3)
                        * (u_a_val / g)
                    )

                    ts_val = 0.95 * tp_val
                    stat = "Fetch Limited"

            hs.append(hs_val)
            tp.append(tp_val)
            ts.append(ts_val)
            status.append(stat)
            time_critical_list.append(time_critical)

        # ✅ Assign all calculated values to dataframe
        df["wave_height"] = np.round(hs, 2)
        df["peak_wave_period"] = np.round(tp, 2)
        df["wave_period"] = np.round(ts, 2)
        df["wave_type"] = status
        df["u_stab"] = u_stab
        df["rl"] = rl
        df["duration_h"] = duration_h
        df["duration_s"] = duration_s
        df["time_critical"] = time_critical_list

        # -------------------------
        # Hindcast output
        # -------------------------
        df_hindcast = df[[
            "year","month","day","time","speed","dir_deg","direction",
            "wave_height","peak_wave_period","wave_period"
        ]].copy()

        df_hindcast.columns = [
            "Year","Month","Day","Time","Wind_Speed","Wind_Dir_Deg",
            "Wind_Dir","Hs","Tp","Ts"
        ]

        # =====================================================
        # Annual maxima
        # =====================================================
        annual_max = df.groupby(["year","direction"])["speed"].max().unstack()
        annual_max_wave = df.groupby(["year","direction"])["wave_height"].max().unstack()

        annual_max = annual_max.reindex(columns=DIRECTION_ORDER)
        annual_max_wave = annual_max_wave.reindex(columns=DIRECTION_ORDER)

        # =====================================================
        # EVA – Wind
        # =====================================================
        eva_table = pd.DataFrame(index=RETURN_PERIODS, columns=DIRECTION_ORDER)
        best_fit_rows = []

        for d in DIRECTION_ORDER:
            data_dir = annual_max[d].dropna().values
            if len(data_dir) < 5:
                continue

            # ✅ Sanitize data: convert to float64 and remove infinities
            data_dir = np.array(data_dir, dtype=np.float64)
            data_dir = data_dir[np.isfinite(data_dir)]
            
            if len(data_dir) < 5:
                continue
                
            data_dir = np.sort(data_dir)
            probs = (np.arange(1,len(data_dir)+1)-0.44)/(len(data_dir)+0.12)
            fits = {}

            # Try Normal distribution
            try:
                mu, sd = stats.norm.fit(data_dir)
                fits["Normal"] = (rmse(data_dir, stats.norm.ppf(probs, mu, sd)), stats.norm(mu, sd))
            except Exception:
                pass

            # Try Lognormal distribution
            try:
                s, loc, sc = stats.lognorm.fit(data_dir, floc=0)
                fits["Lognormal"] = (rmse(data_dir, stats.lognorm.ppf(probs, s, loc, sc)), stats.lognorm(s, loc, sc))
            except Exception:
                pass

            # Try Gumbel distribution with error handling
            try:
                loc, sc = stats.gumbel_r.fit(data_dir)
                fits["Gumbel"] = (rmse(data_dir, stats.gumbel_r.ppf(probs, loc, sc)), stats.gumbel_r(loc, sc))
            except (OverflowError, ValueError, RuntimeError):
                pass

            # Try Weibull distribution
            try:
                c, loc, sc = stats.weibull_min.fit(data_dir, floc=0)
                fits["Weibull"] = (rmse(data_dir, stats.weibull_min.ppf(probs, c, loc, sc)), stats.weibull_min(c, loc, sc))
            except Exception:
                pass

            # Try Log-Pearson III distribution
            try:
                logx = np.log10(data_dir[data_dir > 0])
                sk, loc, sc = stats.pearson3.fit(logx)
                fits["Log-Pearson III"] = (
                    rmse(data_dir, 10**stats.pearson3.ppf(probs, sk, loc, sc)),
                    (sk, loc, sc)
                )
            except Exception:
                pass

            # Only proceed if at least one distribution fit succeeded
            if not fits:
                continue

            best = min(fits, key=lambda k: fits[k][0])
            best_fit_rows.append([d, best, round(fits[best][0], 4)])

            for rp in RETURN_PERIODS:
                p = 1 - 1/rp
                try:
                    eva_table.loc[rp, d] = round(
                        10**stats.pearson3.ppf(p, *fits[best][1]) if best=="Log-Pearson III"
                        else fits[best][1].ppf(p), 2
                    )
                except Exception:
                    eva_table.loc[rp, d] = np.nan

        best_fit_df = pd.DataFrame(best_fit_rows, columns=["Direction","Best Fit","RMSE"]).set_index("Direction") if best_fit_rows else pd.DataFrame()

        # =====================================================
        # EVA – Wave
        # =====================================================
        eva_table_wave = pd.DataFrame(index=RETURN_PERIODS, columns=DIRECTION_ORDER)
        best_fit_wave_rows = []

        for d in DIRECTION_ORDER:

            # ⛔ Skip wave EVA if fetch = 0
            if fetch_dict.get(d, 0) <= 0:
                continue

            data_dir = annual_max_wave[d].dropna().values
            data_dir = data_dir[data_dir > 0]

            if len(data_dir) < 5:
                continue

            # ✅ Sanitize data: convert to float64 and remove infinities
            data_dir = np.array(data_dir, dtype=np.float64)
            data_dir = data_dir[np.isfinite(data_dir)]
            
            if len(data_dir) < 5:
                continue

            data_dir = np.sort(data_dir)
            probs = (np.arange(1,len(data_dir)+1)-0.44)/(len(data_dir)+0.12)
            fits = {}

            # Try Normal distribution
            try:
                mu, sd = stats.norm.fit(data_dir)
                fits["Normal"] = (rmse(data_dir, stats.norm.ppf(probs, mu, sd)), stats.norm(mu, sd))
            except Exception:
                pass

            # Try Lognormal distribution
            try:
                s, loc, sc = stats.lognorm.fit(data_dir, floc=0)
                fits["Lognormal"] = (rmse(data_dir, stats.lognorm.ppf(probs, s, loc, sc)), stats.lognorm(s, loc, sc))
            except Exception:
                pass

            # Try Gumbel distribution with error handling
            try:
                loc, sc = stats.gumbel_r.fit(data_dir)
                fits["Gumbel"] = (rmse(data_dir, stats.gumbel_r.ppf(probs, loc, sc)), stats.gumbel_r(loc, sc))
            except (OverflowError, ValueError, RuntimeError):
                pass

            # Try Weibull distribution
            try:
                c, loc, sc = stats.weibull_min.fit(data_dir, floc=0)
                fits["Weibull"] = (rmse(data_dir, stats.weibull_min.ppf(probs, c, loc, sc)), stats.weibull_min(c, loc, sc))
            except Exception:
                pass

            # Try Log-Pearson III distribution
            try:
                logx = np.log10(data_dir)
                sk, loc, sc = stats.pearson3.fit(logx)
                fits["Log-Pearson III"] = (
                    rmse(data_dir, 10**stats.pearson3.ppf(probs, sk, loc, sc)),
                    (sk, loc, sc)
                )
            except Exception:
                pass

            # Only proceed if at least one distribution fit succeeded
            if not fits:
                continue

            best = min(fits, key=lambda k: fits[k][0])
            best_fit_wave_rows.append([d, best, round(fits[best][0], 4)])

            for rp in RETURN_PERIODS:
                p = 1 - 1/rp
                try:
                    eva_table_wave.loc[rp, d] = round(
                        10**stats.pearson3.ppf(p, *fits[best][1]) if best=="Log-Pearson III"
                        else fits[best][1].ppf(p), 2
                    )
                except Exception:
                    eva_table_wave.loc[rp, d] = np.nan

        best_fit_wave_df = pd.DataFrame(
            best_fit_wave_rows, columns=["Direction","Best Fit","RMSE"]
        ).set_index("Direction") if best_fit_wave_rows else pd.DataFrame()

        # =====================================================
        # Generate Rose Plots (if enabled)
        # =====================================================
        windrose_fig = None
        windrose_data = None
        waverose_fig = None
        waverose_data = None
        monthly_windroses = None
        monthly_waveroses = None
        
        if enable_rose:
            # Wind Rose (full dataset) - no max_range constraint for full dataset
            use_beaufort = (wind_bin_type == "Beaufort Scale")
            interval = None if use_beaufort else wind_bin_interval
            windrose_fig, windrose_data = create_windrose(df, use_beaufort=use_beaufort, bin_interval=interval, max_range=None)
            
            # Wave Rose (full dataset) - no max_range constraint for full dataset
            waverose_fig, waverose_data = create_waverose(df, bin_interval=wave_bin_interval, fetch_dict=fetch_dict, max_range=None)
            
            # Monthly Wind Roses - use auto-scaling (max_range=None) for optimal display
            monthly_windroses = create_monthly_windroses(df, use_beaufort=use_beaufort, bin_interval=interval, max_range=None)
            
            # Monthly Wave Roses - use auto-scaling (max_range=None) for optimal display
            monthly_waveroses = create_monthly_waveroses(df, bin_interval=wave_bin_interval, fetch_dict=fetch_dict, max_range=None)

        # -------------------------
        # Store
        # -------------------------
        st.session_state.update(
            processed=True,
            df=df,
            df_hindcast=df_hindcast,
            annual_max=annual_max,
            annual_max_wave=annual_max_wave,
            eva_table=eva_table,
            eva_table_wave=eva_table_wave,
            best_fit_df=best_fit_df,
            best_fit_wave_df=best_fit_wave_df,
            windrose_fig=windrose_fig,
            windrose_data=windrose_data,
            waverose_fig=waverose_fig,
            waverose_data=waverose_data,
            monthly_windroses=monthly_windroses,
            monthly_waveroses=monthly_waveroses
        )

        st.success("✅ Processing complete!")

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.error(traceback.format_exc())

# =====================================================
# Display results if processed
# =====================================================
if st.session_state.processed:

    # Create tabs dynamically based on whether rose plots are enabled
    if enable_rose and st.session_state.windrose_fig is not None:
        tabs = st.tabs([
            "📊 Hindcast Results",
            "🌬️ Wind EVA",
            "🌊 Wave EVA",
            "🎯 Wind Rose",
            "🌊 Wave Rose"
        ])
        tab1, tab2, tab3, tab4, tab5 = tabs
    else:
        tabs = st.tabs([
            "📊 Hindcast Results",
            "🌬️ Wind EVA",
            "🌊 Wave EVA"
        ])
        tab1, tab2, tab3 = tabs

    with tab1:
        st.subheader("📋 Hindcast Data Preview")
        st.dataframe(st.session_state.df_hindcast.head(50))

        st.download_button(
            "⬇️ Download Hindcast Results (CSV)",
            st.session_state.df_hindcast.to_csv(index=False),
            file_name="hindcast_results.csv",
            mime="text/csv"
        )

    with tab2:
        st.subheader("📈 Annual Maximum Wind Speed by Direction")
        st.dataframe(st.session_state.annual_max)

        if st.session_state.best_fit_df is not None and not st.session_state.best_fit_df.empty:
            st.subheader("🏆 Best Distribution per Direction (Wind)")
            st.dataframe(st.session_state.best_fit_df)

            st.subheader("📘 Wind EVA Return Levels")
            st.dataframe(st.session_state.eva_table)
        else:
            st.warning("⚠️ Insufficient data for Wind EVA analysis")

    with tab3:
        st.subheader("📈 Annual Maximum Wave Height by Direction")
        st.dataframe(st.session_state.annual_max_wave)

        if st.session_state.best_fit_wave_df is not None and not st.session_state.best_fit_wave_df.empty:
            st.subheader("🏆 Best Distribution per Direction (Wave)")
            st.dataframe(st.session_state.best_fit_wave_df)

            st.subheader("📘 Wave EVA Return Levels")
            st.dataframe(st.session_state.eva_table_wave)
        else:
            st.warning("⚠️ Insufficient data for Wave EVA analysis")

    # Display rose plots if they were generated
    if enable_rose and st.session_state.windrose_fig is not None:
        with tab4:
            st.subheader("🎯 Wind Rose Plot - Full Dataset")
            st.plotly_chart(st.session_state.windrose_fig, use_container_width=True)
            
            st.subheader("📊 Wind Rose Data (% Frequency)")
            st.dataframe(st.session_state.windrose_data.round(2))
            
            st.download_button(
                "⬇️ Download Wind Rose Data (CSV)",
                st.session_state.windrose_data.to_csv(),
                file_name="windrose_data.csv",
                mime="text/csv"
            )
            
            # Monthly Wind Roses
            if st.session_state.monthly_windroses:
                st.markdown("---")
                st.subheader("📅 Monthly Wind Roses")
                
                for month_name, (fig, data) in st.session_state.monthly_windroses.items():
                    st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.subheader("🌊 Wave Rose Plot - Full Dataset")
            st.plotly_chart(st.session_state.waverose_fig, use_container_width=True)
            
            st.subheader("📊 Wave Rose Data (% Frequency)")
            st.dataframe(st.session_state.waverose_data.round(2))
            
            st.download_button(
                "⬇️ Download Wave Rose Data (CSV)",
                st.session_state.waverose_data.to_csv(),
                file_name="waverose_data.csv",
                mime="text/csv"
            )
            
            # Monthly Wave Roses
            if st.session_state.monthly_waveroses:
                st.markdown("---")
                st.subheader("📅 Monthly Wave Roses")
                
                for month_name, (fig, data) in st.session_state.monthly_waveroses.items():
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download all roses button
            st.markdown("---")
            if st.button("📦 Generate Download Package for All Rose Plots"):
                try:
                    import io
                    import zipfile
                    from datetime import datetime as dt
                    
                    # Try to use kaleido, fall back to HTML if not available
                    try:
                        import kaleido
                        use_kaleido = True
                    except ImportError:
                        use_kaleido = False
                        st.warning("⚠️ Kaleido not available. Exporting as HTML files instead of PNG.")
                    
                    with st.spinner("Generating files... This may take a moment."):
                        # Create a BytesIO buffer for the zip file
                        zip_buffer = io.BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            
                            if use_kaleido:
                                # Export as PNG using kaleido
                                # Full wind rose image
                                if st.session_state.windrose_fig is not None:
                                    img_bytes = pio.to_image(st.session_state.windrose_fig, format="png", width=1000, height=800)
                                    zip_file.writestr("wind_rose_full.png", img_bytes)
                                
                                # Full wave rose image
                                if st.session_state.waverose_fig is not None:
                                    img_bytes = pio.to_image(st.session_state.waverose_fig, format="png", width=1000, height=800)
                                    zip_file.writestr("wave_rose_full.png", img_bytes)
                                
                                # Monthly wind roses
                                if st.session_state.monthly_windroses:
                                    for month_name, (fig, data) in st.session_state.monthly_windroses.items():
                                        img_bytes = pio.to_image(fig, format="png", width=1000, height=800)
                                        zip_file.writestr(f"wind_rose_{month_name}.png", img_bytes)
                                
                                # Monthly wave roses
                                if st.session_state.monthly_waveroses:
                                    for month_name, (fig, data) in st.session_state.monthly_waveroses.items():
                                        img_bytes = pio.to_image(fig, format="png", width=1000, height=800)
                                        zip_file.writestr(f"wave_rose_{month_name}.png", img_bytes)
                            
                            else:
                                # Export as interactive HTML
                                # Full wind rose
                                if st.session_state.windrose_fig is not None:
                                    html_str = pio.to_html(st.session_state.windrose_fig, include_plotlyjs='cdn')
                                    zip_file.writestr("wind_rose_full.html", html_str)
                                
                                # Full wave rose
                                if st.session_state.waverose_fig is not None:
                                    html_str = pio.to_html(st.session_state.waverose_fig, include_plotlyjs='cdn')
                                    zip_file.writestr("wave_rose_full.html", html_str)
                                
                                # Monthly wind roses
                                if st.session_state.monthly_windroses:
                                    for month_name, (fig, data) in st.session_state.monthly_windroses.items():
                                        html_str = pio.to_html(fig, include_plotlyjs='cdn')
                                        zip_file.writestr(f"wind_rose_{month_name}.html", html_str)
                                
                                # Monthly wave roses
                                if st.session_state.monthly_waveroses:
                                    for month_name, (fig, data) in st.session_state.monthly_waveroses.items():
                                        html_str = pio.to_html(fig, include_plotlyjs='cdn')
                                        zip_file.writestr(f"wave_rose_{month_name}.html", html_str)
                        
                        # Get the zip file contents
                        zip_buffer.seek(0)
                        
                        file_ext = "png" if use_kaleido else "html"
                        st.download_button(
                            label=f"⬇️ Download All Rose Plots (ZIP - {file_ext.upper()})",
                            data=zip_buffer.getvalue(),
                            file_name=f"all_rose_plots_{dt.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                        
                        st.success(f"✅ ZIP package ready for download! ({file_ext.upper()} format)")
                
                except Exception as e:
                    st.error(f"Error generating download package: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
