import pandas as pd

def get_dataframe_info(df: pd.DataFrame) -> dict:
    """
    Extracts basic EDA information and statistics from a pandas DataFrame.
    
    Args:
        df (pd.DataFrame): The dataset.
        
    Returns:
        dict: A dictionary containing rows, columns, memory usage, etc.
    """
    # Calculate memory usage in Megabytes
    memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
    
    info = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(memory_usage, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_per_col": df.isnull().sum().to_dict()
    }
    
    return info