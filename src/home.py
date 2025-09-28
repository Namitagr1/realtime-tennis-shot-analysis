import streamlit as st

st.set_page_config(page_title="🏆 Tennis Analytics Hub", layout="wide")

st.title("🎾 Tennis Analytics Hub")
st.markdown("""
Welcome to the **Real-Time Tennis Analysis Project**.  
Choose a module below to explore:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/Front Shot Analysis.py", label="📹 Front Shot Analysis", icon="🎥")
with col2:
    st.page_link("pages/Stats Dashboard.py", label="📊 Stats Dashboard", icon="📈")
with col3:
    st.page_link("pages/2_Full_Court_Analysis.py", label="📐 Full Court Analysis", icon="🖼️")

st.markdown("---")
st.write("Use the navigation bar at the top to switch between modules at any time.")
