import streamlit as st
import pandas as pd
import joblib
import io

def show_prediction():
    st.title("🔮 Make Predictions")
    
    # Check if models have been trained
    if "trained_models" not in st.session_state or not st.session_state.trained_models:
        st.warning("No models are available. Please go to the 'Train Models' page and train at least one model first.")
        return
        
    df = st.session_state.processed_data
    features = st.session_state.features
    task_type = st.session_state.task_type
    trained_models = st.session_state.trained_models
    
    st.markdown("Use a trained model to make predictions on new data. The input form below is dynamically generated based on your selected features.")
    
    col1, col2 = st.columns([1, 2])
    
    # --- Model Selection ---
    with col1:
        st.subheader("1. Select Model")
        model_names = list(trained_models.keys())
        
        # Default to the best model if it was found during Comparison/Training
        default_index = 0
        if st.session_state.get('best_model') in model_names:
            default_index = model_names.index(st.session_state.best_model)
            
        selected_model_name = st.selectbox("Choose a trained model", model_names, index=default_index)
        
        selected_pipeline = trained_models[selected_model_name]["pipeline"]
        
        # --- Download Model ---
        st.markdown("---")
        st.subheader("📥 Export Model")
        st.caption("Download the complete pipeline (preprocessing + model) for deployment.")
        
        # Serialize model in memory
        model_buffer = io.BytesIO()
        joblib.dump(selected_pipeline, model_buffer)
        model_buffer.seek(0)
        
        st.download_button(
            label=f"Download {selected_model_name} (.pkl)",
            data=model_buffer,
            file_name=f"{selected_model_name.replace(' ', '_').lower()}_pipeline.pkl",
            mime="application/octet-stream"
        )
        
    # --- Dynamic Input Form ---
    with col2:
        st.subheader("2. Enter Feature Values")
        
        with st.form("prediction_form"):
            user_inputs = {}
            
            # Create a 2-column layout for the form
            form_col1, form_col2 = st.columns(2)
            
            for idx, feature in enumerate(features):
                # Alternate between columns for better layout
                current_col = form_col1 if idx % 2 == 0 else form_col2
                
                # Check data type to render appropriate input widget
                col_type = df[feature].dtype
                
                with current_col:
                    if pd.api.types.is_numeric_dtype(col_type):
                        # For numeric, use number_input with the mean as default
                        mean_val = float(df[feature].mean())
                        user_inputs[feature] = st.number_input(
                            label=feature, 
                            value=mean_val,
                            format="%.4f"
                        )
                    else:
                        # For categorical, use selectbox with unique values
                        unique_values = df[feature].dropna().unique().tolist()
                        user_inputs[feature] = st.selectbox(
                            label=feature, 
                            options=unique_values
                        )
                        
            submit_button = st.form_submit_button(label="Generate Prediction 🚀", type="primary")

    st.divider()
    
    # --- Execute Prediction ---
    if submit_button:
        st.subheader("🎯 Prediction Result")
        
        # Convert dictionary to DataFrame (must match training feature shape)
        input_df = pd.DataFrame([user_inputs])
        
        try:
            with st.spinner("Processing..."):
                # Make prediction
                prediction = selected_pipeline.predict(input_df)[0]
                
                # Check for prediction probabilities (confidence)
                confidence = None
                if task_type == "Classification" and hasattr(selected_pipeline.named_steps['model'], "predict_proba"):
                    proba = selected_pipeline.predict_proba(input_df)[0]
                    confidence = max(proba) * 100
                    
            # Display results beautifully
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.success("Prediction Generated Successfully!")
                if task_type == "Regression":
                    st.metric(label="Predicted Value", value=f"{prediction:,.4f}")
                else:
                    st.metric(label="Predicted Class", value=str(prediction))
                    
            with res_col2:
                if confidence is not None:
                    st.info("Model Confidence")
                    st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
                elif task_type == "Classification":
                    st.info("Confidence score not supported by this algorithm (e.g., standard SVM without probability).")
                else:
                    st.info("Confidence score is not applicable for Regression tasks.")
                    
            # Display the DataFrame that was passed to the model
            with st.expander("View Input Data"):
                st.dataframe(input_df, hide_index=True)

            result_df = input_df.copy()
            result_df['Predicted_Target'] = prediction
            if confidence is not None:
                result_df['Confidence_Score'] = f"{confidence:.2f}%"
                
            csv_pred = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Prediction Result (.csv)",
                data=csv_pred,
                file_name="prediction_result.csv",
                mime="text/csv",
            )
                
        except Exception as e:
            st.error(f"An error occurred during prediction. Details: {e}")

if __name__ == "__main__":
    show_prediction()