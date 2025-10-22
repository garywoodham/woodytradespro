import streamlit as st

def render_help():
    st.title("ℹ️ Help & About")
    st.markdown("""
    **WoodyTradesPro**  
    AI-enhanced trading assistant that analyzes multiple assets, generates live buy/sell signals,
    and visualizes take-profit and stop-loss zones.

    **Features:**
    - 🧠 Machine Learning–based trend prediction  
    - 💬 Sentiment and volatility integration  
    - 🎯 Risk-adjusted trade targets  
    - 📊 Multi-asset dashboard  

    **Tips:**
    - Adjust the *risk level* for tighter or wider stop-losses.  
    - Switch between *intervals* for intraday or swing views.  
    - Check *accuracy metrics* to validate reliability.
    """)