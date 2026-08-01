import streamlit as st
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart ML Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State Variables ---
# This ensures data persists when navigating between pages
if "raw_data" not in st.session_state:
    st.session_state.raw_data = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "target_column" not in st.session_state:
    st.session_state.target_column = None
if "features" not in st.session_state:
    st.session_state.features = None
if "task_type" not in st.session_state:
    st.session_state.task_type = None
if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}
if "best_model" not in st.session_state:
    st.session_state.best_model = None

# --- Ensure Directories Exist ---
os.makedirs("models", exist_ok=True)
os.makedirs("datasets", exist_ok=True)

# --- Navigation Setup ---
# Using Streamlit 1.36+ native navigation
pages = {
    "Start Here": [
        st.Page("pages/Home.py", title="Home", icon="🏠", default=True),
        st.Page("pages/Upload_Data.py", title="Data Upload & Prep", icon="📂"),
    ],
    "Machine Learning": [
        st.Page("pages/Model_Training.py", title="Train Models", icon="⚙️"),
        st.Page("pages/Comparison.py", title="Compare Models", icon="📊"),
        st.Page("pages/Prediction.py", title="Predictions", icon="🔮"),
    ],
    "Insights": [
        st.Page("pages/Visualization.py", title="Visualizations", icon="📈"),
        st.Page("pages/About.py", title="About", icon="ℹ️"),
    ]
}

pg = st.navigation(pages)

# Run the navigation router
pg.run()