import streamlit as st
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils.preprocessing import build_preprocessor
from utils.regression import get_regression_model
from utils.classification import get_classification_model
from utils.metrics import evaluate_regression, evaluate_classification

def show_model_training():
    st.title("⚙️ Model Training")
    
    # Ensure data is prepared
    if st.session_state.get('processed_data') is None:
        st.warning("Please upload and configure your dataset in the 'Data Upload & Prep' section first.")
        return

    # --- Load Configuration ---
    df = st.session_state.processed_data
    target_col = st.session_state.target_column
    features = st.session_state.features
    task_type = st.session_state.task_type
    
    X = df[features]
    y = df[target_col]
    
    # Identify numeric and categorical features for the preprocessor
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    st.markdown(f"**Task Detected:** `{task_type}` | **Target:** `{target_col}` | **Features:** `{len(features)}`")
    st.divider()
    
    # --- Mode Selection ---
    mode = st.radio("Select Training Mode", ["Mode 1: Train Single Model", "Mode 2: Compare All Models"], horizontal=True)
    
    # Define available algorithms
    if task_type == "Regression":
        # Conditionally show Single vs Multiple based on feature count
        lin_reg_name = "Linear Regression" if len(features) == 1 else "Multiple Linear Regression"
        algorithms = [lin_reg_name, "Polynomial Regression"]
    else:
        algorithms = ["K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest"]

    if mode == "Mode 1: Train Single Model":
        st.subheader("Train a Single Model")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_model = st.selectbox("Select Algorithm", algorithms)
        
        with col2:
            st.markdown("**Hyperparameters**")
            use_default = st.checkbox("Use default parameters", value=True)
            kwargs = {}
            
            if not use_default:
                if selected_model == "Polynomial Regression":
                    kwargs["degree"] = st.slider("Degree", 2, 5, 2)
                elif selected_model == "K-Nearest Neighbors":
                    kwargs["n_neighbors"] = st.slider("n_neighbors", 1, 20, 5)
                elif selected_model == "Support Vector Machine":
                    kwargs["kernel"] = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
                    kwargs["C"] = st.number_input("C (Regularization)", 0.01, 100.0, 1.0, step=0.1)
                elif selected_model == "Decision Tree":
                    kwargs["max_depth"] = st.slider("Max Depth", 1, 50, 10)
                    kwargs["criterion"] = st.selectbox("Criterion", ["gini", "entropy"])
                elif selected_model == "Random Forest":
                    kwargs["n_estimators"] = st.slider("Number of Estimators", 10, 500, 100, step=10)
                    kwargs["max_depth"] = st.slider("Max Depth", 1, 50, 10)

        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                # 1. Train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=st.session_state.test_size, 
                    random_state=st.session_state.random_state, 
                    shuffle=st.session_state.shuffle
                )
                
                # 2. Build Preprocessor
                preprocessor = build_preprocessor(
                    numeric_features, categorical_features,
                    st.session_state.impute_strategy,
                    st.session_state.encode_strategy,
                    st.session_state.scale_strategy
                )
                
                # 3. Get Model
                if task_type == "Regression":
                    model_instance = get_regression_model(selected_model, **kwargs)
                else:
                    model_instance = get_classification_model(selected_model, **kwargs)
                
                # 4. Create Pipeline
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_instance)])
                
                # 5. Train with progress tracking
                progress_bar = st.progress(0)
                start_time = time.time()
                
                # Simulate progress for UX
                for percent_complete in range(0, 100, 20):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete + 20)
                    
                pipeline.fit(X_train, y_train)
                end_time = time.time()
                elapsed_time = end_time - start_time
                
                # 6. Evaluate
                y_pred = pipeline.predict(X_test)
                if task_type == "Regression":
                    metrics = evaluate_regression(y_test, y_pred)
                else:
                    metrics = evaluate_classification(y_test, y_pred)
                metrics["Training Time (s)"] = round(elapsed_time, 4)
                
                # 7. Save to session state
                if "trained_models" not in st.session_state:
                    st.session_state.trained_models = {}
                
                st.session_state.trained_models[selected_model] = {
                    "pipeline": pipeline,
                    "metrics": metrics,
                    "X_test": X_test, # Saved for visualiztions later
                    "y_test": y_test,
                    "y_pred": y_pred
                }
                
                st.success(f"✅ Training completed successfully in {elapsed_time:.2f} seconds!")
                
                # 8. Display Metrics
                st.subheader(f"{selected_model} Evaluation")
                cols = st.columns(len(metrics) - 1) # exclude time from cards
                
                for idx, (metric_name, metric_val) in enumerate(metrics.items()):
                    if metric_name != "Training Time (s)":
                        cols[idx].metric(metric_name, round(metric_val, 4))
                
    elif mode == "Mode 2: Compare All Models":
        st.subheader("Compare All Supported Models")
        st.info("This will train all available models using default hyperparameters and compare their performance.")
        
        if st.button("🚀 Train & Compare All", type="primary"):
            with st.spinner("Training all models... This may take a moment."):
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, 
                    test_size=st.session_state.test_size, 
                    random_state=st.session_state.random_state, 
                    shuffle=st.session_state.shuffle
                )
                
                preprocessor = build_preprocessor(
                    numeric_features, categorical_features,
                    st.session_state.impute_strategy,
                    st.session_state.encode_strategy,
                    st.session_state.scale_strategy
                )
                
                progress_bar = st.progress(0)
                total_models = len(algorithms)
                results = []
                
                for idx, algo in enumerate(algorithms):
                    start_time = time.time()
                    
                    if task_type == "Regression":
                        model_instance = get_regression_model(algo)
                    else:
                        model_instance = get_classification_model(algo)
                        
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_instance)])
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    
                    elapsed = time.time() - start_time
                    
                    if task_type == "Regression":
                        metrics = evaluate_regression(y_test, y_pred)
                    else:
                        metrics = evaluate_classification(y_test, y_pred)
                        
                    metrics["Model"] = algo
                    metrics["Training Time (s)"] = round(elapsed, 4)
                    results.append(metrics)
                    
                    # Save to session state
                    st.session_state.trained_models[algo] = {
                        "pipeline": pipeline,
                        "metrics": {k: v for k, v in metrics.items() if k != "Model"},
                        "X_test": X_test,
                        "y_test": y_test,
                        "y_pred": y_pred
                    }
                    
                    progress_bar.progress((idx + 1) / total_models)
                
                st.success("✅ All models trained successfully!")
                
                # Create Comparison Table
                results_df = pd.DataFrame(results)
                
                # Reorder columns to put 'Model' first
                cols = ['Model'] + [c for c in results_df.columns if c != 'Model']
                results_df = results_df[cols]
                
                st.subheader("🏆 Leaderboard")
                
                # Determine sort column (Accuracy for classification, R2 for regression)
                sort_col = "Accuracy" if task_type == "Classification" else "R² Score"
                ascending = False # Higher is better for both Accuracy and R2
                
                results_df = results_df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
                
                # Save the best model globally
                best_model_name = results_df.iloc[0]['Model']
                st.session_state.best_model = best_model_name
                
                st.markdown(f"**Best Model:** :trophy: `{best_model_name}`")
                
                # Display highlighted dataframe
                st.dataframe(
                    results_df.style.highlight_max(subset=[sort_col], color='lightgreen', axis=0),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.info("Head over to the 'Compare Models' and 'Visualizations' tabs in the sidebar for deep-dive charts!")

if __name__ == "__main__":
    show_model_training()