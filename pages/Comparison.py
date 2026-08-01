import streamlit as st
import pandas as pd
import plotly.express as px

def show_comparison():
    st.title("📊 Model Comparison")
    
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
    
    st.markdown("Compare the performance and efficiency of your trained models. Use the charts below to find the best fit for your dataset.")
    st.divider()

    # --- Determine Default Primary Metric ---
    if task_type == "Classification":
        available_metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
        primary_metric = "Accuracy"
    else:
        available_metrics = ["R² Score", "MAE", "MSE", "RMSE"]
        primary_metric = "R² Score"

    # Let user select which metric to visualize
    selected_metric = st.selectbox("Select Metric to Visualize", available_metrics, index=0)
    
    # Determine if higher is better for the selected metric
    higher_is_better = selected_metric in ["Accuracy", "F1 Score", "Precision", "Recall", "R² Score"]
    
    # Sort the dataframe based on the selected metric
    df_results = df_results.sort_values(by=selected_metric, ascending=not higher_is_better)
    best_model_name = df_results.iloc[0]['Model']
    
    # --- Visualization Layout ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🏆 Performance: {selected_metric}")
        
        # Plotly Bar Chart for Performance
        fig_perf = px.bar(
            df_results, 
            x="Model", 
            y=selected_metric, 
            color="Model",
            text_auto='.3f',
            title=f"{selected_metric} Comparison (Higher is better)" if higher_is_better else f"{selected_metric} Comparison (Lower is better)"
        )
        fig_perf.update_layout(showlegend=False, xaxis_title="", yaxis_title=selected_metric)
        # Highlight the best model visually if needed, but color="Model" is already nice.
        st.plotly_chart(fig_perf, use_container_width=True)
        
    with col2:
        st.subheader("⏱️ Training Time")
        
        # Plotly Bar Chart for Time
        fig_time = px.bar(
            df_results.sort_values(by="Training Time (s)", ascending=True), 
            x="Model", 
            y="Training Time (s)", 
            color="Model",
            text_auto='.3f',
            title="Training Time Comparison (Lower is better)"
        )
        fig_time.update_layout(showlegend=False, xaxis_title="", yaxis_title="Time in Seconds")
        st.plotly_chart(fig_time, use_container_width=True)

    st.divider()
    
    # --- Detailed Comparison Table ---
    st.subheader("📋 Detailed Comparison Table")
    
    st.markdown(f"**Current Best Model (based on {selected_metric}):** `{best_model_name}`")
    
    # Highlight max or min depending on the metric
    if higher_is_better:
        st.dataframe(
            df_results.style.highlight_max(subset=[selected_metric], color='lightgreen', axis=0),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(
            df_results.style.highlight_min(subset=[selected_metric], color='lightgreen', axis=0),
            use_container_width=True,
            hide_index=True
        )

    csv_report = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Evaluation Report (.csv)",
        data=csv_report,
        file_name="model_evaluation_report.csv",
        mime="text/csv",
    )

if __name__ == "__main__":
    show_comparison()