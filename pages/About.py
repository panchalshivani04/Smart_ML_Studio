import streamlit as st

def show_about():
    st.title("ℹ️ About Smart ML Studio")
    st.markdown("---")
    
    st.markdown(
        """
        **Smart ML Studio** is a robust, no-code Machine Learning platform designed to bridge the gap 
        between raw data and actionable predictive models. 
        """
    )
    
    with st.expander("🚀 Core Features", expanded=True):
        st.markdown(
            """
            * **Automated Task Detection**: Intelligently identifies if your data requires Classification or Regression.
            * **Robust Preprocessing**: Handles missing values and scales/encodes features seamlessly using Scikit-Learn pipelines.
            * **Model Showdown**: Train individual models with hyperparameter tuning, or run a batch comparison to generate a leaderboard.
            * **Interactive Visualizations**: Powered by Plotly to give you dynamic, deep insights into model behavior and feature importance.
            * **Deployment Ready**: Export your trained pipelines as `.pkl` files for instant use in production environments.
            """
        )
        
    with st.expander("🧠 Supported Algorithms"):
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
        
    with st.expander("🛠️ Tech Stack"):
        st.markdown(
            """
            * **Frontend**: [Streamlit](https://streamlit.io/)
            * **Data Manipulation**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
            * **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/)
            * **Visualizations**: [Plotly](https://plotly.com/python/)
            * **Serialization**: Joblib
            """
        )
        
    with st.expander("🔮 Future Improvements"):
        st.markdown(
            """
            * Add support for Deep Learning models (PyTorch/TensorFlow).
            * Implement unsupervised learning (Clustering & PCA).
            * Add Advanced hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
            * Time-series forecasting support.
            """
        )
        
    st.divider()
    st.caption("Developed using Python 3.12+ | Clean Architecture | PEP8 Compliant")

if __name__ == "__main__":
    show_about()