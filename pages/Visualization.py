import streamlit as st
import pandas as pd
from utils.visualization import (
    plot_correlation_heatmap, plot_target_distribution, plot_feature_distribution,
    plot_actual_vs_predicted, plot_residuals, plot_confusion_matrix, 
    plot_roc_curve_binary, plot_feature_importance
)

def show_visualization():
    st.write("")
    st.title("📈 Insights & Visualizations")
    st.markdown(
        "<p style='font-size: 1.1rem; color: #666;'>Explore interactive visual analytics for your dataset and evaluate model performance.</p>", 
        unsafe_allow_html=True
    )
    st.write("")
    
    if st.session_state.get('processed_data') is None:
        st.warning("Please upload and configure your dataset first.")
        return

    # Tabs for separation of concerns
    tab1, tab2 = st.tabs(["📊 Data Exploration (EDA)", "🔬 Model Evaluation"])
    
    # ==========================================
    # TAB 1: Data EDA
    # ==========================================
    with tab1:
        st.write("")
        df = st.session_state.processed_data
        target_col = st.session_state.target_column
        task_type = st.session_state.task_type
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            with st.container(border=True):
                st.subheader("🎯 Target Variable")
                st.write("")
                fig_target = plot_target_distribution(df, target_col, task_type)
                st.plotly_chart(fig_target, use_container_width=True)
            
        with col2:
            with st.container(border=True):
                st.subheader("🔗 Correlation Matrix")
                st.write("")
                fig_corr = plot_correlation_heatmap(df)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("No numeric columns available for correlation heatmap.")
                
        st.write("")
        with st.container(border=True):
            st.subheader("🔎 Feature Deep-Dive")
            st.markdown(
                "<p style='font-size: 0.95rem; color: #666; margin-bottom: 10px;'>Select an individual feature to analyze its distribution and relationship with the target.</p>", 
                unsafe_allow_html=True
            )
            selected_feature = st.selectbox("Select a feature to explore", [c for c in df.columns if c != target_col], label_visibility="collapsed")
            st.write("")
            fig_feat = plot_feature_distribution(df, selected_feature, target_col, task_type)
            st.plotly_chart(fig_feat, use_container_width=True)

    # ==========================================
    # TAB 2: Model Evaluation
    # ==========================================
    with tab2:
        st.write("")
        
        if "trained_models" not in st.session_state or not st.session_state.trained_models:
            st.info("No models trained yet. Go to 'Train Models' to populate this section.")
            return
            
        trained_models = st.session_state.trained_models
        model_names = list(trained_models.keys())
        
        # --- Control Panel for Model Selection ---
        with st.container(border=True):
            st.subheader("🎛️ Select Model to Visualize")
            selected_model = st.selectbox("Choose model", model_names, label_visibility="collapsed")
        
        st.write("")
        
        model_data = trained_models[selected_model]
        pipeline = model_data["pipeline"]
        X_test = model_data["X_test"]
        y_test = model_data["y_test"]
        y_pred = model_data["y_pred"]
        
        # --- Shared Feature Importance Plot ---
        with st.container(border=True):
            st.subheader("🌟 Feature Importance")
            st.write("")
            fig_imp = plot_feature_importance(pipeline, st.session_state.features)
            if fig_imp:
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info(f"Feature importance or coefficients could not be extracted for {selected_model}.")
        
        st.write("")

        # --- Task-Specific Plots ---
        eval_col1, eval_col2 = st.columns(2, gap="large")
        
        if task_type == "Regression":
            with eval_col1:
                with st.container(border=True):
                    st.subheader("📈 Actual vs Predicted")
                    st.write("")
                    fig_avp = plot_actual_vs_predicted(y_test, y_pred)
                    st.plotly_chart(fig_avp, use_container_width=True)
                
            with eval_col2:
                with st.container(border=True):
                    st.subheader("📉 Residual Plot")
                    st.write("")
                    fig_res = plot_residuals(y_test, y_pred)
                    st.plotly_chart(fig_res, use_container_width=True)
                
        elif task_type == "Classification":
            with eval_col1:
                with st.container(border=True):
                    st.subheader("🟦 Confusion Matrix")
                    st.write("")
                    if task_type == "Classification":
                        fig_cm = plot_confusion_matrix(y_test, y_pred)
                        st.plotly_chart(fig_cm, use_container_width=True)
                
            with eval_col2:
                with st.container(border=True):
                    st.subheader("📈 ROC Curve")
                    st.write("")
                    # ROC Curve is primarily for Binary Classification
                    if len(set(y_test)) == 2:
                        # Check if model supports predict_proba
                        if hasattr(pipeline.named_steps['model'], "predict_proba"):
                            # Get probability of the positive class
                            y_prob = pipeline.predict_proba(X_test)[:, 1]
                            
                            # We need to map string labels to binary for roc_curve if they are strings
                            if y_test.dtype == 'object' or y_test.dtype == 'bool':
                                y_test_bin = pd.factorize(y_test)[0]
                            else:
                                y_test_bin = y_test
                                
                            fig_roc = plot_roc_curve_binary(y_test_bin, y_prob)
                            st.plotly_chart(fig_roc, use_container_width=True)
                        else:
                            st.info(f"{selected_model} does not support probability predictions required for ROC curves.")
                    else:
                        st.info("ROC Curve is currently supported for Binary Classification tasks only.")

if __name__ == "__main__":
    show_visualization()