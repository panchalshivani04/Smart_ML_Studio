import streamlit as st

def show_about():
    # --- SaaS-Style Centered Hero Section ---
    st.write("")
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 2rem 0;">
            <h1 style="font-size: 3.2rem; font-weight: 800; color: #0f172a; margin-bottom: 0.25rem; line-height: 1.2;">
                🧠 Smart ML Studio
            </h1>
            <p style="font-size: 1.25rem; color: #64748b; font-weight: 400; margin: 0;">
                The complete no-code workspace for end-to-end machine learning workflows.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # --- Platform Overview (The "Mission" Card) ---
    with st.container(border=True):
        st.markdown("<h4 style='color: #0f172a; margin-bottom: 1rem;'>About the Platform</h4>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color: #475569; font-size: 1.05rem; line-height: 1.7;'>"
            "Smart ML Studio bridges the gap between raw datasets and actionable predictive models. "
            "Designed for rapid prototyping and deployment, this platform eliminates the need for boilerplate code, "
            "allowing data scientists and domain experts to focus entirely on feature engineering, model evaluation, and business logic."
            "</p>",
            unsafe_allow_html=True
        )

    st.write("")
    st.write("")

    # --- Capabilities (SaaS Feature Grid) ---
    st.markdown("<h3 style='color: #0f172a; margin-bottom: 1.5rem;'>Platform Capabilities</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        with st.container(border=True):
            st.markdown("#### ⚡ Intelligent Automation")
            st.markdown("<p style='color: #64748b; font-size: 0.95rem;'>Automatically detects classification vs. regression tasks and seamlessly handles missing values, scaling, and categorical encoding.</p>", unsafe_allow_html=True)
            
        with st.container(border=True):
            st.markdown("#### 📊 Visual Analytics")
            st.markdown("<p style='color: #64748b; font-size: 0.95rem;'>Deep data insights powered by Plotly. Explore correlation matrices, residual plots, ROC curves, and feature importance interactively.</p>", unsafe_allow_html=True)

    with col2:
        with st.container(border=True):
            st.markdown("#### 🏆 Automated Benchmarking")
            st.markdown("<p style='color: #64748b; font-size: 0.95rem;'>Run batch comparisons across multiple algorithms instantly. Generate leaderboards to identify the optimal model architecture.</p>", unsafe_allow_html=True)
            
        with st.container(border=True):
            st.markdown("#### 📦 Production Ready")
            st.markdown("<p style='color: #64748b; font-size: 0.95rem;'>Export your trained models as fully contained <code>.pkl</code> pipelines, ready for immediate deployment in live production environments.</p>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    # --- Supported Algorithms (Clean Split View) ---
    with st.container(border=True):
        st.markdown("<h4 style='color: #0f172a; margin-bottom: 1.5rem;'>Model Architecture Support</h4>", unsafe_allow_html=True)
        alg_col1, alg_col2 = st.columns(2)
        
        with alg_col1:
            st.markdown("**Continuous Targets (Regression)**")
            st.markdown(
                "<ul style='color: #475569; line-height: 1.8;'>"
                "<li>Linear Regression</li>"
                "<li>Multiple Linear Regression</li>"
                "<li>Polynomial Regression</li>"
                "</ul>", 
                unsafe_allow_html=True
            )
            
        with alg_col2:
            st.markdown("**Categorical Targets (Classification)**")
            st.markdown(
                "<ul style='color: #475569; line-height: 1.8;'>"
                "<li>K-Nearest Neighbors (KNN)</li>"
                "<li>Support Vector Machine (SVM)</li>"
                "<li>Decision Tree Classification</li>"
                "<li>Random Forest Ensemble</li>"
                "</ul>", 
                unsafe_allow_html=True
            )

    st.write("")
    st.write("")

    # --- Infrastructure / Tech Stack (Horizontal Flow) ---
    st.markdown("<h3 style='color: #0f172a; margin-bottom: 1.5rem;'>Core Infrastructure</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.write("")
        flow_col1, flow_col2, flow_col3 = st.columns(3)
        
        with flow_col1:
            st.markdown("<div style='text-align: center;'><h2 style='margin: 0;'>🎨</h2><h4 style='color: #0f172a; margin-top: 10px;'>Presentation</h4><p style='color: #64748b; font-size: 0.9rem;'>Streamlit UI<br>Responsive HTML/CSS</p></div>", unsafe_allow_html=True)
            
        with flow_col2:
            st.markdown("<div style='text-align: center;'><h2 style='margin: 0;'>⚙️</h2><h4 style='color: #0f172a; margin-top: 10px;'>Compute Engine</h4><p style='color: #64748b; font-size: 0.9rem;'>Scikit-Learn ML Core<br>Pandas & NumPy</p></div>", unsafe_allow_html=True)
            
        with flow_col3:
            st.markdown("<div style='text-align: center;'><h2 style='margin: 0;'>📈</h2><h4 style='color: #0f172a; margin-top: 10px;'>Analytics & Export</h4><p style='color: #64748b; font-size: 0.9rem;'>Plotly Interactivity<br>Joblib Serialization</p></div>", unsafe_allow_html=True)
        st.write("")

    st.write("")
    st.write("")

    # --- Future Enhancements (Portfolio Project List) ---
    st.markdown("<h3 style='color: #0f172a; margin-bottom: 1.5rem;'>🚀 Future Enhancements</h3>", unsafe_allow_html=True)
    with st.container(border=True):
        st.write("")
        enh_col1, enh_col2 = st.columns(2, gap="large")
        
        with enh_col1:
            st.markdown(
                "<ul style='color: #475569; line-height: 2.2; list-style-type: none; padding-left: 5px; margin: 0;'>"
                "<li>🔄 <b>Cross Validation</b></li>"
                "<li>🚀 <b>XGBoost & LightGBM Integration</b></li>"
                "<li>🧠 <b>Deep Learning Support</b> (PyTorch/TensorFlow)</li>"
                "<li>🔍 <b>Unsupervised Learning</b> (K-Means, PCA)</li>"
                "<li>📊 <b>SHAP Model Explainability</b></li>"
                "</ul>",
                unsafe_allow_html=True
            )
            
        with enh_col2:
            st.markdown(
                "<ul style='color: #475569; line-height: 2.2; list-style-type: none; padding-left: 5px; margin: 0;'>"
                "<li>✨ <b>AutoML Recommendations</b></li>"
                "<li>📈 <b>Time-Series Forecasting</b></li>"
                "<li>☁️ <b>Cloud Deployment</b></li>"
                "<li>🔐 <b>User Authentication</b></li>"
                "<li>📌 <b>Model Versioning</b></li>"
                "</ul>",
                unsafe_allow_html=True
            )
        st.write("")

    st.write("")
    st.divider()
    
    # --- Footer ---
    st.markdown(
        "<p style='text-align: center; color: #94a3b8; font-size: 0.85rem; padding-bottom: 2rem;'>"
        "© 2026 Smart ML Studio &nbsp;|&nbsp; Built with Python 3.12+ &nbsp;|&nbsp; Clean Architecture"
        "</p>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    show_about()