import streamlit as st

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# SIDEBAR NAVIGATION
# =====================================================================
st.sidebar.title("Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["Home", "Dataset", "EDA", "Model", "Prediction", "About"],
    label_visibility="collapsed"
)

# =====================================================================
# PAGE ROUTING
# =====================================================================
if page == "Home":
    st.switch_page("app.py")
elif page == "Dataset":
    st.switch_page("pages/1_Dataset.py")
elif page == "EDA":
    st.switch_page("pages/2_EDA.py")
elif page == "Model":
    st.switch_page("pages/3_Model.py")
elif page == "Prediction":
    st.switch_page("pages/4_Prediksi.py")
elif page == "About":
    st.switch_page("pages/5_Tentang.py")
