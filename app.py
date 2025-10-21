import streamlit as st

st.set_page_config(
    page_title="WoodyTrades Pro — MultiAsset",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark mode CSS
DARK_CSS = """
<style>
:root { color-scheme: dark; }
[data-testid="stAppViewContainer"], .stApp { background: #0f1116 !important; }
.stMarkdown, .stText, .stSelectbox, .stDataFrame, .stCaption, .stMetric { color: #e6e6e6 !important; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }
div[data-testid="stMetricValue"] { font-weight: 700; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# Import tabs from folder
from tabs.tab_overview import render_overview
from tabs.tab_trends import render_trends
from tabs.tab_detailed import render_detailed
from tabs.tab_scenarios import render_scenarios
from tabs.tab_help import render_help

# Sidebar Navigation
st.sidebar.title("WoodyTrades Pro")
st.sidebar.caption("Dark theme • ML predictions • Live market data")

tabs = {
    "Overview": render_overview,
    "Trends": render_trends,
    "Detailed": render_detailed,
    "Scenarios": render_scenarios,
    "Help": render_help
}

choice = st.sidebar.radio("Navigate", list(tabs.keys()))
tabs[choice]()