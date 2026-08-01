from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score
)
import numpy as np

def evaluate_regression(y_true, y_pred) -> dict:
    """Calculates regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R² Score": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse)
    }

def evaluate_classification(y_true, y_pred) -> dict:
    """Calculates classification metrics (using macro average for multiclass support)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_true, y_pred, average='macro', zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average='macro', zero_division=0)
    }