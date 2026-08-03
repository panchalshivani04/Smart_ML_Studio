import streamlit as st
import pandas as pd
from utils.helper import get_dataframe_info
from utils.preprocessing import detect_task_type

def show_upload_data():
    st.write("")
    st.title("📂 Data Upload & Preparation")
    st.markdown(
        "<p style='font-size: 1.1rem; color: #666;'>Upload your dataset, explore its health, and configure the preprocessing pipeline.</p>", 
        unsafe_allow_html=True
    )
    st.write("")
    
    # --- File Uploader ---
    with st.container(border=True):
        uploaded_file = st.file_uploader("Drop your CSV file here", type=["csv"])
    
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
            
            st.write("")
            st.subheader("Dataset Overview")
            
            # --- Display Metric Cards ---
            with st.container(border=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rows", f"{info['rows']:,}")
                col2.metric("Columns", f"{info['columns']:,}")
                col3.metric("Missing Values", f"{info['missing_values']:,}")
                col4.metric("Memory Usage", f"{info['memory_mb']} MB")
            
            st.write("")
            
            # --- Detailed Data Exploration Tabs ---
            tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "🏷️ Columns & Types", "⚠️ Missing & Duplicates"])
            
            with tab1:
                st.write("")
                st.dataframe(df.head(100), use_container_width=True)
                st.caption("Displaying the top 100 rows of the dataset.")
                
            with tab2:
                st.write("")
                dtypes_df = pd.DataFrame({
                    "Column Name": list(info["dtypes"].keys()),
                    "Data Type": list(info["dtypes"].values())
                })
                st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
                
            with tab3:
                st.write("")
                missing_df = pd.DataFrame({
                    "Column Name": list(info["missing_per_col"].keys()),
                    "Missing Values": list(info["missing_per_col"].values())
                })
                if info["missing_values"] > 0:
                    missing_df = missing_df[missing_df["Missing Values"] > 0]
                
                col_a, col_b = st.columns(2, gap="large")
                with col_a:
                    st.markdown("**Missing Values by Column**")
                    st.dataframe(missing_df, use_container_width=True, hide_index=True)
                    
                with col_b:
                    st.markdown("**Dataset Duplicates**")
                    with st.container(border=True):
                        st.metric("Duplicate Rows Detected", f"{info['duplicates']:,}")
                        if info['duplicates'] > 0:
                            st.warning("Removing duplicates prevents data leakage and bias.")
                            if st.button("🗑️ Clean Duplicates", use_container_width=True):
                                st.session_state.raw_data = df.drop_duplicates().reset_index(drop=True)
                                st.rerun()

            st.divider()
            
            # ==========================================
            # CONFIGURATION & PREPROCESSING (SaaS Styling)
            # ==========================================
            st.header("⚙️ Configuration Studio")
            st.markdown("<p style='color: #666;'>Define your target, select features, and tune preprocessing steps.</p>", unsafe_allow_html=True)
            st.write("")
            
            # 1. Feature Selection Container
            with st.container(border=True):
                st.subheader("1. Task & Feature Selection")
                fs_col1, fs_col2 = st.columns(2, gap="large")
                
                with fs_col1:
                    target_col = st.selectbox("Target Column (Y)", df.columns)
                    
                    # Auto-detect task type
                    detected_task = detect_task_type(df, target_col)
                    task_index = 0 if detected_task == "Classification" else 1
                    task_type = st.radio("Task Type", ["Classification", "Regression"], index=task_index)
                    
                with fs_col2:
                    default_features = [c for c in df.columns if c != target_col]
                    features = st.multiselect("Input Features (X)", df.columns, default=default_features)
            
            st.write("")
            
            # 2. Preprocessing & Split Containers Side-by-Side
            prep_col, split_col = st.columns(2, gap="large")
            
            with prep_col:
                with st.container(border=True):
                    st.subheader("2. Data Preprocessing")
                    impute_strategy = st.selectbox("Missing Values", ["Drop rows", "Mean", "Median", "Most frequent"])
                    encode_strategy = st.selectbox("Categorical Encoding", ["One-hot encoding", "Label encoding"])
                    scale_strategy = st.selectbox("Feature Scaling", ["None", "StandardScaler", "MinMaxScaler"])
                    
            with split_col:
                with st.container(border=True):
                    st.subheader("3. Train-Test Split")
                    split_ratio = st.selectbox("Training Data Ratio", ["60%", "70%", "80%", "90%"], index=2)
                    random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
                    st.write("") # Alignment spacing
                    shuffle_data = st.checkbox("Shuffle Data before splitting", value=True)
                    
            st.write("")
            
            # 3. Submit Configuration
            submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
            with submit_col2:
                if st.button("✅ Save Configuration & Proceed", type="primary", use_container_width=True):
                    if len(features) == 0:
                        st.error("Please select at least one input feature.")
                    elif target_col in features:
                        st.error("Target column cannot be in the input features.")
                    else:
                        processed_df = df.copy()
                        if impute_strategy == "Drop rows":
                            processed_df = processed_df.dropna(subset=features + [target_col]).reset_index(drop=True)
                        
                        test_size_map = {"60%": 0.4, "70%": 0.3, "80%": 0.2, "90%": 0.1}
                        test_size = test_size_map[split_ratio]
                        
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
                        
                        st.success("Configuration saved! You can now proceed to the Model Training page.")
                    
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty. Please upload a valid dataset.")
        except Exception as e:
            st.error(f"An error occurred while reading the file. Details: {e}")
            
    else:
        st.info("Awaiting file upload... Please browse or drag and drop a CSV file above.")

if __name__ == "__main__":
    show_upload_data()