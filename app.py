import streamlit as st
import pandas as pd
import utils

st.set_page_config(page_title="WoodyTrades Pro — Multi-Asset", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

DARK_CSS = """
<style>
:root, .stApp, .main, .block-container {background-color: #0E1117 !important; color: #E6EDF3 !important;}
[data-testid="stSidebar"] {background-color: #0B0E14 !important;}
.woody-card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 14px 16px; box-shadow: 0 0 0 1px rgba(255,255,255,0.02) inset; }
[data-testid="stMetricValue"] { color: #E6EDF3 !important; }
[data-testid="stMetricDelta"] { font-weight: 700; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

from tabs.tab_overview import render_overview
from tabs.tab_detailed import render_detailed
from tabs.tab_trends import render_trends
from tabs.tab_scenarios import render_scenarios

st.sidebar.title("WoodyTrades Pro")
st.sidebar.caption("Live data • Always Dark Theme • No training required")

with st.sidebar.expander("Data Settings", True):
    interval = st.selectbox("Timeframe", list(utils.INTERVALS.keys()), index=2)
    period = st.selectbox("History Window", ["7d","14d","30d","60d","90d","180d","1y","2y"], index=3)
    risk = st.selectbox("Risk Profile", ["Low","Medium","High"], index=1)
    show_help = st.toggle("Show How-To overlays", value=False)

st.sidebar.divider()
st.sidebar.write("**TP/SL Multipliers** (by risk)")
st.sidebar.write(utils.risk_tooltip_md(), unsafe_allow_html=True)

tabs = {
    "Overview": lambda: render_overview(interval, period, risk, show_help),
    "Detailed": lambda: render_detailed(interval, period, risk, show_help),
    "Trends":   lambda: render_trends(interval, period, risk, show_help),
    "Scenarios":lambda: render_scenarios(interval, period, risk, show_help),
}
choice = st.sidebar.radio("Navigate", list(tabs.keys()))
tabs[choice]()

st.sidebar.divider()
if st.sidebar.button("Export All Symbols CSV"):
    dfs = []
    for asset, symbol in utils.ASSET_SYMBOLS.items():
        df = utils.fetch_data(symbol, interval=interval, period=period)
        if df is None or df.empty:
            continue
        df["Asset"] = asset
        df["Symbol"] = symbol
        dfs.append(df.reset_index())
    if dfs:
        big = pd.concat(dfs, ignore_index=True)
        csv = big.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button("Download CSV", csv, file_name="woodytrades_all.csv", mime="text/csv")
    else:
        st.sidebar.info("No data available to export.")
