import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

# =====================================================
# Page config
# =====================================================
st.set_page_config(page_title="Wind EVA (RMSE-based)", layout="wide")
st.title("🌬️ Wind Extreme Value Analysis (RMSE-based Best Fit)")

# =====================================================
# Constants
# =====================================================
DIRECTION_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 1000]

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

# =====================================================
# Initialize session state
# =====================================================
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'annual_max' not in st.session_state:
    st.session_state.annual_max = None
if 'eva_table' not in st.session_state:
    st.session_state.eva_table = None
if 'best_fit_df' not in st.session_state:
    st.session_state.best_fit_df = None

# =====================================================
# Sidebar input
# =====================================================
st.sidebar.header("⏱ Time Settings")
start_date = st.sidebar.date_input(
    "Start date",
    value=datetime(2005, 1, 1)
)
start_hour = st.sidebar.selectbox(
    "Start hour",
    [f"{h:02d}:00" for h in range(24)],
    index=0
)
interval_hours = st.sidebar.number_input(
    "Time interval (hours)",
    min_value=1,
    value=1,
    step=1
)

# =====================================================
# Data input
# =====================================================
st.subheader("📋 Paste u10 and v10 data")
st.caption("Two columns: u10 v10 (space or tab separated)")
raw_text = st.text_area("Paste data here", height=220)
run_btn = st.button("🚀 Process & Run EVA")

# =====================================================
# Main processing
# =====================================================
if run_btn and raw_text.strip():
    try:
        # -------------------------
        # Parse input
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
        start_dt = datetime.combine(
            start_date,
            datetime.strptime(start_hour, "%H:%M").time()
        )
        df["datetime"] = [
            start_dt + timedelta(hours=i * interval_hours)
            for i in range(len(df))
        ]
        df["year"] = df["datetime"].dt.year
        
        # -------------------------
        # Wind calculation
        # -------------------------
        df["speed"] = compute_wind_speed(df["u10"], df["v10"])
        df["dir_deg"] = wind_dir_from(df["u10"], df["v10"])
        df["direction"] = df["dir_deg"].apply(degree_to_compass)
        
        # =====================================================
        # Annual maximum table
        # =====================================================
        annual_max = (
            df.groupby(["year", "direction"], observed=False)["speed"]
            .max()
            .unstack()
            .reindex(columns=DIRECTION_ORDER)
        )
        
        # =====================================================
        # EVA
        # =====================================================
        eva_table = pd.DataFrame(index=RETURN_PERIODS, columns=DIRECTION_ORDER)
        best_fit_rows = []
        
        for d in DIRECTION_ORDER:
            data_dir = annual_max[d].dropna().values
            if len(data_dir) < 5:
                continue
            
            data_dir = np.sort(data_dir)
            probs = (np.arange(1, len(data_dir) + 1) - 0.44) / (len(data_dir) + 0.12)
            fits = {}
            
            mu, sd = stats.norm.fit(data_dir)
            fits["Normal"] = (rmse(data_dir, stats.norm.ppf(probs, mu, sd)), stats.norm(mu, sd))
            
            s, loc, sc = stats.lognorm.fit(data_dir, floc=0)
            fits["Lognormal"] = (rmse(data_dir, stats.lognorm.ppf(probs, s, loc, sc)), stats.lognorm(s, loc, sc))
            
            loc, sc = stats.gumbel_r.fit(data_dir)
            fits["Gumbel"] = (rmse(data_dir, stats.gumbel_r.ppf(probs, loc, sc)), stats.gumbel_r(loc, sc))
            
            c, loc, sc = stats.weibull_min.fit(data_dir, floc=0)
            fits["Weibull"] = (rmse(data_dir, stats.weibull_min.ppf(probs, c, loc, sc)), stats.weibull_min(c, loc, sc))
            
            logx = np.log10(data_dir)
            sk, loc, sc = stats.pearson3.fit(logx)
            sim = 10 ** stats.pearson3.ppf(probs, sk, loc, sc)
            fits["Log-Pearson III"] = (rmse(data_dir, sim), (sk, loc, sc))
            
            best_name = min(fits, key=lambda k: fits[k][0])
            best_rmse = round(fits[best_name][0], 4)
            best_fit_rows.append([d, best_name, best_rmse])
            
            for rp in RETURN_PERIODS:
                p = 1 - 1 / rp
                if best_name == "Log-Pearson III":
                    sk, loc, sc = fits[best_name][1]
                    val = 10 ** stats.pearson3.ppf(p, sk, loc, sc)
                else:
                    val = fits[best_name][1].ppf(p)
                eva_table.loc[rp, d] = round(val, 2)
        
        best_fit_df = (
            pd.DataFrame(best_fit_rows, columns=["Direction", "Best Fit", "RMSE"])
            .set_index("Direction")
            .reindex(DIRECTION_ORDER)
        )
        
        # Store results in session state
        st.session_state.df = df
        st.session_state.annual_max = annual_max
        st.session_state.eva_table = eva_table
        st.session_state.best_fit_df = best_fit_df
        st.session_state.processed = True
        
    except Exception as e:
        st.error(f"Error processing data: {e}")

# =====================================================
# Display results if processed
# =====================================================
if st.session_state.processed:
    st.subheader("📊 Processed Data Preview")
    st.dataframe(st.session_state.df.head())
    
    st.subheader("📈 Annual Maximum Wind Speed")
    st.dataframe(st.session_state.annual_max)
    
    st.subheader("🏆 Best Distribution per Direction (RMSE)")
    st.dataframe(st.session_state.best_fit_df)
    
    st.subheader("📘 EVA Return Level Table")
    st.dataframe(st.session_state.eva_table)
    
    # =====================================================
    # DOWNLOAD BUTTONS (CSV)
    # =====================================================
    st.subheader("⬇️ Download Results (CSV)")
    
    st.download_button(
        "Download processed data (CSV)",
        st.session_state.df.to_csv(index=False),
        file_name="processed_data.csv",
        mime="text/csv"
    )
    
    st.download_button(
        "Download annual maximum table (CSV)",
        st.session_state.annual_max.to_csv(),
        file_name="annual_max_table.csv",
        mime="text/csv"
    )
    
    st.download_button(
        "Download best distribution + RMSE (CSV)",
        st.session_state.best_fit_df.to_csv(),
        file_name="best_distribution_rmse.csv",
        mime="text/csv"
    )
    
    st.download_button(
        "Download EVA result table (CSV)",
        st.session_state.eva_table.to_csv(),
        file_name="eva_results.csv",
        mime="text/csv"
    )
