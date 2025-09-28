import streamlit as st

def navbar():
    st.markdown(
        """
        <style>
        .navbar {
            display: flex;
            justify-content: center;
            gap: 2rem;
            background-color: #222;
            padding: 0.75rem;
            border-radius: 12px;
        }
        .navbar a {
            text-decoration: none;
            color: white;
            font-weight: 600;
            padding: 6px 12px;
        }
        .navbar a:hover {
            background-color: #444;
            border-radius: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="navbar">
            <a href="/Front%20Shot%20Analysis" target="_self">Front Shot</a>
            <a href="/Stats%20Dashboard" target="_self">Stats Dashboard</a>
            <a href="/2_Full_Court_Analysis" target="_self">Full Court</a>
        </div>
        """,
        unsafe_allow_html=True
    )
