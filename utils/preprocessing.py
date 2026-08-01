import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder

def detect_task_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Automatically detects if the task is Regression or Classification.
    """
    if df[target_col].dtype == 'object' or df[target_col].dtype == 'bool':
        return "Classification"
    elif df[target_col].nunique() < 20:  # Heuristic for classification
        return "Classification"
    else:
        return "Regression"

def build_preprocessor(numeric_features: list, categorical_features: list, 
                       impute_strategy: str, encode_strategy: str, scale_strategy: str):
    """
    Builds a Scikit-Learn ColumnTransformer for preprocessing data.
    """
    # --- Numeric Pipeline ---
    num_steps = []
    if impute_strategy in ['Mean', 'Median', 'Most frequent']:
        # Map UI string to sklearn strategy
        strategy = impute_strategy.lower().replace(' ', '_')
        num_steps.append(('imputer', SimpleImputer(strategy=strategy)))
        
    if scale_strategy == 'StandardScaler':
        num_steps.append(('scaler', StandardScaler()))
    elif scale_strategy == 'MinMaxScaler':
        num_steps.append(('scaler', MinMaxScaler()))
        
    num_pipeline = Pipeline(num_steps) if num_steps else 'passthrough'
    
    # --- Categorical Pipeline ---
    cat_steps = []
    if impute_strategy in ['Mean', 'Median', 'Most frequent']:
        # For categorical columns, we always impute with most_frequent regardless of numeric choice
        cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
    if encode_strategy == 'One-hot encoding':
        # sparse_output=False is required for compatibility with many models/pandas integration
        cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
    elif encode_strategy == 'Label encoding':
        # Using OrdinalEncoder for features (LabelEncoder is strictly for targets in sklearn)
        cat_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
        
    cat_pipeline = Pipeline(cat_steps) if cat_steps else 'passthrough'
    
    # --- Combine into ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_features),
            ('cat', cat_pipeline, categorical_features)
        ],
        remainder='drop' # Drop columns not specified in features
    )
    
    return preprocessor