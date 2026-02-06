import streamlit as st
import numpy as np
import pandas as pd
import math
from scipy import stats
from datetime import datetime, timedelta

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

# =====================================================
# Initialize session state
# =====================================================
for k in [
    "processed","df","df_hindcast","annual_max","annual_max_wave",
    "eva_table","eva_table_wave","best_fit_df","best_fit_wave_df"
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
            best_fit_wave_df=best_fit_wave_df
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

    tab1, tab2, tab3 = st.tabs([
        "📊 Hindcast Results",
        "🌬️ Wind EVA",
        "🌊 Wave EVA"
    ])

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
