import streamlit as st

def show_home():
    # --- Hero Section ---
    st.write("")
    st.markdown("<h1 style='text-align: center; font-weight: 700;'>🧠 Smart ML Studio</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.15rem; color: #666; margin-bottom: 2rem;'>"
        "Your professional, no-code machine learning companion.<br>"
        "Upload data, build models, compare algorithms, and make predictions in minutes."
        "</p>", 
        unsafe_allow_html=True
    )
    
    st.divider()
    st.write("")

    # --- Features & Algorithms (Card Layout) ---
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        with st.container(border=True):
            st.subheader("✨ Core Features")
            st.markdown(
                """
                - **Automated Data Processing**: Handle missing values, encoding, and scaling seamlessly.
                - **Intelligent Task Detection**: Automatically detects Classification vs. Regression.
                - **Comprehensive Comparison**: Train multiple models and rank them instantly.
                - **Interactive Visualizations**: Powered by Plotly for deep data insights.
                - **Export Ready**: Download trained pipelines and prediction reports.
                """
            )
            
    with col2:
        with st.container(border=True):
            st.subheader("🤖 Supported Algorithms")
            st.markdown(
                """
                **Regression Models**
                - Linear Regression
                - Multiple Linear Regression
                - Polynomial Regression
                
                **Classification Models**
                - K-Nearest Neighbors (KNN)
                - Support Vector Machine (SVM)
                - Decision Tree
                - Random Forest
                """
            )

    st.write("")
    st.write("")

    # --- Workflow Diagram (Horizontal Stepper) ---
    st.subheader("🗺️ How It Works")
    st.write("")
    
    step1, step2, step3, step4, step5 = st.columns(5)
    
    with step1:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; margin: 0;'>📂</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 10px;'><b>1. Upload</b><br><span style='font-size: 0.85rem; color: #666;'>Provide CSV & explore</span></p>", unsafe_allow_html=True)

    with step2:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; margin: 0;'>⚙️</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 10px;'><b>2. Preprocess</b><br><span style='font-size: 0.85rem; color: #666;'>Clean & select features</span></p>", unsafe_allow_html=True)

    with step3:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; margin: 0;'>🧠</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 10px;'><b>3. Train</b><br><span style='font-size: 0.85rem; color: #666;'>Compare & tune models</span></p>", unsafe_allow_html=True)

    with step4:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; margin: 0;'>📊</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 10px;'><b>4. Visualize</b><br><span style='font-size: 0.85rem; color: #666;'>Analyze performance</span></p>", unsafe_allow_html=True)

    with step5:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; margin: 0;'>🔮</h2>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; margin-top: 10px;'><b>5. Predict</b><br><span style='font-size: 0.85rem; color: #666;'>Test & download</span></p>", unsafe_allow_html=True)

    st.write("")
    st.divider()
    
    # --- Call to Action ---
    st.write("")
    cta_col1, cta_col2, cta_col3 = st.columns([1, 2, 1])
    with cta_col2:
        st.markdown("<h3 style='text-align: center; margin-bottom: 1rem;'>🚀 Ready to build models?</h3>", unsafe_allow_html=True)
        if st.button("Go to Data Upload & Preparation", type="primary", use_container_width=True):
            st.switch_page("pages/Upload_Data.py")
    st.write("")

if __name__ == "__main__":
    show_home()