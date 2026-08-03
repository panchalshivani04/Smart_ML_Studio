import streamlit as st
import pandas as pd
import plotly.express as px

def show_comparison():
    st.write("")
    st.title("📊 Model Comparison Dashboard")
    st.markdown(
        "<p style='font-size: 1.1rem; color: #666;'>Analyze, compare, and identify the optimal algorithm for your dataset.</p>", 
        unsafe_allow_html=True
    )
    st.write("")
    
    # Check if models have been trained
    if "trained_models" not in st.session_state or not st.session_state.trained_models:
        st.warning("No models have been trained yet. Please go to the 'Train Models' page first.")
        return
    
    task_type = st.session_state.task_type
    trained_models = st.session_state.trained_models
    
    # --- Prepare Data for Comparison ---
    results = []
    for model_name, data in trained_models.items():
        metrics = data["metrics"].copy()
        metrics["Model"] = model_name
        results.append(metrics)
        
    df_results = pd.DataFrame(results)
    
    # Reorder columns to have 'Model' first
    cols = ['Model'] + [c for c in df_results.columns if c != 'Model']
    df_results = df_results[cols]

    # --- Determine Default Primary Metric ---
    if task_type == "Classification":
        available_metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
        primary_metric = "Accuracy"
    else:
        available_metrics = ["R² Score", "MAE", "MSE", "RMSE"]
        primary_metric = "R² Score"

    # --- Control Panel ---
    with st.container(border=True):
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.markdown("### 🎛️ Evaluation Metric")
            selected_metric = st.selectbox("Select metric to visualize", available_metrics, index=0, label_visibility="collapsed")
        with col2:
            st.markdown("### ℹ️ Insight")
            st.markdown(
                f"<p style='color: #666; margin-top: 5px;'>Currently sorting models based on <b>{selected_metric}</b> and comparing their relative <b>Training Time</b>.</p>", 
                unsafe_allow_html=True
            )

    # Determine if higher is better for the selected metric
    higher_is_better = selected_metric in ["Accuracy", "F1 Score", "Precision", "Recall", "R² Score"]
    
    # Sort the dataframe based on the selected metric
    df_results = df_results.sort_values(by=selected_metric, ascending=not higher_is_better)
    best_model_name = df_results.iloc[0]['Model']
    
    st.write("")
    
    # --- Visualization Layout ---
    chart_col1, chart_col2 = st.columns(2, gap="large")
    
    with chart_col1:
        with st.container(border=True):
            st.subheader(f"🏆 Performance ({selected_metric})")
            st.write("")
            
            # Plotly Bar Chart for Performance
            fig_perf = px.bar(
                df_results, 
                x="Model", 
                y=selected_metric, 
                color="Model",
                text_auto='.3f',
                title=f"Higher is better" if higher_is_better else f"Lower is better"
            )
            fig_perf.update_layout(showlegend=False, xaxis_title="", yaxis_title=selected_metric, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_perf, use_container_width=True)
        
    with chart_col2:
        with st.container(border=True):
            st.subheader("⏱️ Training Time")
            st.write("")
            
            # Plotly Bar Chart for Time
            fig_time = px.bar(
                df_results.sort_values(by="Training Time (s)", ascending=True), 
                x="Model", 
                y="Training Time (s)", 
                color="Model",
                text_auto='.3f',
                title="Time in Seconds (Lower is better)"
            )
            fig_time.update_layout(showlegend=False, xaxis_title="", yaxis_title="Seconds", margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_time, use_container_width=True)

    st.write("")
    
    # --- Detailed Comparison Table ---
    with st.container(border=True):
        st.subheader("📋 Detailed Leaderboard")
        st.markdown(f"**Overall Best Model (by {selected_metric}):** 🥇 `{best_model_name}`")
        st.write("")
        
        # Highlight max or min depending on the metric
        if higher_is_better:
            styled_df = df_results.style.highlight_max(subset=[selected_metric], color='rgba(76, 175, 80, 0.2)', axis=0)
        else:
            styled_df = df_results.style.highlight_min(subset=[selected_metric], color='rgba(76, 175, 80, 0.2)', axis=0)
            
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Add Download button for Evaluation Report
        csv_report = df_results.to_csv(index=False).encode('utf-8')
        
        dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
        with dl_col2:
            st.download_button(
                label="📥 Download Evaluation Report (.csv)",
                data=csv_report,
                file_name="model_evaluation_report.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    show_comparison()