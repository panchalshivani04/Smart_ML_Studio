import streamlit as st
import pandas as pd
from utils.helper import get_dataframe_info
from utils.preprocessing import detect_task_type

def show_upload_data():
    st.title("📂 Data Upload & Preparation")
    st.markdown("Upload your dataset, explore it, and configure it for machine learning.")
    
    # --- File Uploader ---
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            if st.session_state.raw_data is None or st.session_state.get('uploaded_filename') != uploaded_file.name:
                df = pd.read_csv(uploaded_file)
                st.session_state.raw_data = df
                st.session_state.uploaded_filename = uploaded_file.name
                # Reset downstream session variables
                st.session_state.processed_data = None
                st.session_state.best_model = None
                
            df = st.session_state.raw_data
            info = get_dataframe_info(df)
            
            # --- Display Metric Cards ---
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", info["rows"])
            col2.metric("Columns", info["columns"])
            col3.metric("Missing Values", info["missing_values"])
            col4.metric("Memory Usage", f"{info['memory_mb']} MB")
            
            st.divider()
            
            # --- Detailed Data Exploration Tabs ---
            tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🏷️ Columns & Types", "⚠️ Missing & Duplicates"])
            
            with tab1:
                st.dataframe(df.head(100), use_container_width=True)
                
            with tab2:
                dtypes_df = pd.DataFrame({
                    "Column Name": list(info["dtypes"].keys()),
                    "Data Type": list(info["dtypes"].values())
                })
                st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
                
            with tab3:
                missing_df = pd.DataFrame({
                    "Column Name": list(info["missing_per_col"].keys()),
                    "Missing Values": list(info["missing_per_col"].values())
                })
                if info["missing_values"] > 0:
                    missing_df = missing_df[missing_df["Missing Values"] > 0]
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Missing Values**")
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
                with col_b:
                    st.markdown(f"**Duplicate Rows:** `{info['duplicates']}`")
                    if info['duplicates'] > 0:
                        if st.button("🗑️ Remove Duplicates"):
                            st.session_state.raw_data = df.drop_duplicates().reset_index(drop=True)
                            st.rerun()

            st.divider()
            
            # ==========================================
            # STEP 3 CONTENT: CONFIGURATION & PREPROCESSING
            # ==========================================
            st.header("⚙️ Feature Selection & Preprocessing")
            
            # Layout for Feature Selection
            fs_col1, fs_col2 = st.columns(2)
            
            with fs_col1:
                target_col = st.selectbox("Select Target Column (Y)", df.columns)
                
                # Auto-detect task type
                detected_task = detect_task_type(df, target_col)
                task_index = 0 if detected_task == "Classification" else 1
                task_type = st.radio("Task Type", ["Classification", "Regression"], index=task_index)
                
            with fs_col2:
                default_features = [c for c in df.columns if c != target_col]
                features = st.multiselect("Select Input Features (X)", df.columns, default=default_features)
                
            st.divider()
            
            # Layout for Preprocessing and Splitting
            prep_col1, prep_col2 = st.columns(2)
            
            with prep_col1:
                st.subheader("🛠️ Data Preprocessing")
                impute_strategy = st.selectbox("Handle Missing Values", ["Drop rows", "Mean", "Median", "Most frequent"])
                encode_strategy = st.selectbox("Categorical Encoding", ["One-hot encoding", "Label encoding"])
                scale_strategy = st.selectbox("Feature Scaling", ["None", "StandardScaler", "MinMaxScaler"])
                
            with prep_col2:
                st.subheader("✂️ Train-Test Split")
                split_ratio = st.selectbox("Training Data Ratio", ["60%", "70%", "80%", "90%"], index=2) # Default 80%
                random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
                shuffle_data = st.checkbox("Shuffle Data", value=True)
                
            # Submit Configuration
            if st.button("✅ Save Configuration & Proceed", type="primary"):
                if len(features) == 0:
                    st.error("Please select at least one input feature.")
                elif target_col in features:
                    st.error("Target column cannot be in the input features.")
                else:
                    # Handle 'Drop rows' logic directly on the dataframe
                    processed_df = df.copy()
                    if impute_strategy == "Drop rows":
                        processed_df = processed_df.dropna(subset=features + [target_col]).reset_index(drop=True)
                    
                    # Map split ratio string to test_size float
                    test_size_map = {"60%": 0.4, "70%": 0.3, "80%": 0.2, "90%": 0.1}
                    test_size = test_size_map[split_ratio]
                    
                    # Save configurations to session state
                    st.session_state.processed_data = processed_df
                    st.session_state.target_column = target_col
                    st.session_state.features = features
                    st.session_state.task_type = task_type
                    st.session_state.impute_strategy = impute_strategy
                    st.session_state.encode_strategy = encode_strategy
                    st.session_state.scale_strategy = scale_strategy
                    st.session_state.test_size = test_size
                    st.session_state.random_state = random_state
                    st.session_state.shuffle = shuffle_data
                    
                    st.success("Configuration saved! You can now proceed to Model Training.")
                    
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            
    else:
        st.info("Awaiting file upload...")

if __name__ == "__main__":
    show_upload_data()