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
    st.write("")
    st.title("⚙️ Model Training Studio")
    st.markdown(
        "<p style='font-size: 1.1rem; color: #666;'>Train individual models with custom hyperparameters or run an automated benchmark to find the best performer.</p>", 
        unsafe_allow_html=True
    )
    st.write("")
    
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
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # --- Configuration Overview Dashboard ---
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Task Detected", task_type)
        col2.metric("Target Column", target_col)
        col3.metric("Features Selected", f"{len(features)} Features")
        
    st.write("")

    # Define available algorithms
    if task_type == "Regression":
        lin_reg_name = "Linear Regression" if len(features) == 1 else "Multiple Linear Regression"
        algorithms = [lin_reg_name, "Polynomial Regression"]
    else:
        algorithms = ["K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "Random Forest"]

    # --- Training Modes (Tabs instead of Radio) ---
    tab_single, tab_compare = st.tabs(["🎯 Train Single Model", "🏆 Compare All Models"])
    
    # ==========================================
    # MODE 1: SINGLE MODEL
    # ==========================================
    with tab_single:
        st.write("")
        
        # FIX: Changed columns to equal width (2 instead of [1, 2])
        col_algo, col_params = st.columns(2, gap="large")
        
        with col_algo:
            # FIX: Added fixed height to force uniform box size
            with st.container(border=True, height=320):
                st.subheader("1. Algorithm")
                st.write("")
                selected_model = st.selectbox("Select Model", algorithms, label_visibility="collapsed")
        
        with col_params:
            # FIX: Added fixed height to force uniform box size
            with st.container(border=True, height=320):
                st.subheader("2. Hyperparameters")
                use_default = st.checkbox("Use default parameters", value=True)
                kwargs = {}
                
                if not use_default:
                    st.write("")
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
                else:
                    st.info("Default parameters will be used for optimal baseline performance.")

        st.write("")
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                with st.spinner(f"Training {selected_model}..."):
                    
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
                    
                    if task_type == "Regression":
                        model_instance = get_regression_model(selected_model, **kwargs)
                    else:
                        model_instance = get_classification_model(selected_model, **kwargs)
                    
                    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_instance)])
                    
                    progress_bar = st.progress(0)
                    start_time = time.time()
                    
                    for percent_complete in range(0, 100, 20):
                        time.sleep(0.1)
                        progress_bar.progress(percent_complete + 20)
                        
                    pipeline.fit(X_train, y_train)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    y_pred = pipeline.predict(X_test)
                    if task_type == "Regression":
                        metrics = evaluate_regression(y_test, y_pred)
                    else:
                        metrics = evaluate_classification(y_test, y_pred)
                    metrics["Training Time (s)"] = round(elapsed_time, 4)
                    
                    if "trained_models" not in st.session_state:
                        st.session_state.trained_models = {}
                    
                    st.session_state.trained_models[selected_model] = {
                        "pipeline": pipeline,
                        "metrics": metrics,
                        "X_test": X_test,
                        "y_test": y_test,
                        "y_pred": y_pred
                    }
                    
                    progress_bar.empty()
                    st.success(f"✅ Model trained successfully in {elapsed_time:.2f} seconds!")
                    
                    # Evaluation Dashboard
                    st.write("")
                    with st.container(border=True):
                        st.subheader(f"📊 {selected_model} Performance")
                        st.write("")
                        cols = st.columns(len(metrics) - 1)
                        for idx, (metric_name, metric_val) in enumerate(metrics.items()):
                            if metric_name != "Training Time (s)":
                                cols[idx].metric(metric_name, round(metric_val, 4))
                                
    # ==========================================
    # MODE 2: COMPARE ALL MODELS
    # ==========================================
    with tab_compare:
        st.write("")
        with st.container(border=True):
            st.markdown(
                "### 🏆 Automated Benchmark", 
                unsafe_allow_html=True
            )
            st.markdown(
                "This mode will sequentially train **all available algorithms** using their default parameters. "
                "It is perfect for establishing baselines and finding the best performing model architecture for your dataset."
            )
            
            st.write("")
            cmp_col1, cmp_col2, cmp_col3 = st.columns([1, 2, 1])
            with cmp_col2:
                if st.button("🚀 Train & Compare All Models", type="primary", use_container_width=True):
                    with st.spinner("Executing batch training..."):
                        
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
                            
                            st.session_state.trained_models[algo] = {
                                "pipeline": pipeline,
                                "metrics": {k: v for k, v in metrics.items() if k != "Model"},
                                "X_test": X_test,
                                "y_test": y_test,
                                "y_pred": y_pred
                            }
                            
                            progress_bar.progress((idx + 1) / total_models)
                        
                        progress_bar.empty()
                        st.success("✅ All models successfully trained & evaluated!")
                        
                        # Leaderboard display
                        results_df = pd.DataFrame(results)
                        cols = ['Model'] + [c for c in results_df.columns if c != 'Model']
                        results_df = results_df[cols]
                        
                        sort_col = "Accuracy" if task_type == "Classification" else "R² Score"
                        ascending = False
                        
                        results_df = results_df.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
                        best_model_name = results_df.iloc[0]['Model']
                        st.session_state.best_model = best_model_name
                        
                        st.write("")
                        st.markdown(f"#### 🥇 Champion Model: `{best_model_name}`")
                        
                        st.dataframe(
                            results_df.style.highlight_max(subset=[sort_col], color='rgba(76, 175, 80, 0.2)', axis=0),
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.info("Head over to the **Compare Models** and **Visualizations** tabs in the sidebar for deep-dive charts!")

if __name__ == "__main__":
    show_model_training()