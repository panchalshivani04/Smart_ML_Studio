import streamlit as st

def show_home():
    st.title("🧠 Welcome to Smart ML Studio")
    st.markdown(
        """
        **Smart ML Studio** is your no-code machine learning companion. 
        Upload your data, build models, compare algorithms, and make predictions in minutes.
        """
    )
    
    st.divider()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✨ Features")
        st.markdown(
            """
            * **Automated Data Processing**: Handle missing values, encoding, and scaling seamlessly.
            * **Intelligent Task Detection**: Automatically detects Classification vs. Regression.
            * **Comprehensive Model Comparison**: Train multiple models and rank them instantly.
            * **Interactive Visualizations**: Powered by Plotly for deep data insights.
            * **Export Ready**: Download trained models and prediction reports.
            """
        )
        
    with col2:
        st.subheader("🤖 Supported Algorithms")
        st.markdown(
            """
            **Regression:**
            * Linear Regression
            * Multiple Linear Regression
            * Polynomial Regression
            
            **Classification:**
            * K-Nearest Neighbors (KNN)
            * Support Vector Machine (SVM)
            * Decision Tree
            * Random Forest
            """
        )

    st.divider()
    
    st.subheader("🗺️ Workflow Diagram")
    
    # Visual workflow using Streamlit info boxes
    st.info("1️⃣ **Upload Data**: Provide a CSV file and explore data health.")
    st.info("2️⃣ **Preprocess**: Select features, target, and clean your dataset.")
    st.info("3️⃣ **Train & Compare**: Choose algorithms and tune hyperparameters.")
    st.info("4️⃣ **Visualize**: Analyze model performance and feature importance.")
    st.info("5️⃣ **Predict & Download**: Test on new data and export results.")

    st.divider()
    
    st.markdown("### 🚀 Ready to get started?")
    if st.button("Go to Data Upload ➔", type="primary"):
        st.switch_page("pages/Upload_Data.py")

if __name__ == "__main__":
    show_home()