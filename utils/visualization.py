import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# --- 1. General EDA Visualizations ---

def plot_correlation_heatmap(df: pd.DataFrame):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.empty:
        return None
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Heatmap")
    return fig

def plot_target_distribution(df: pd.DataFrame, target_col: str, task_type: str):
    if task_type == "Classification":
        fig = px.histogram(df, x=target_col, color=target_col, title=f"Class Distribution: {target_col}")
    else:
        fig = px.histogram(df, x=target_col, marginal="box", title=f"Target Distribution: {target_col}")
    return fig

def plot_feature_distribution(df: pd.DataFrame, feature: str, target_col: str = None, task_type: str = None):
    if pd.api.types.is_numeric_dtype(df[feature]):
        if task_type == "Classification" and target_col:
            fig = px.box(df, x=target_col, y=feature, color=target_col, title=f"{feature} grouped by {target_col}")
        else:
            fig = px.histogram(df, x=feature, marginal="box", title=f"Distribution of {feature}")
    else:
        fig = px.histogram(df, x=feature, color=target_col if task_type == "Classification" else None, title=f"Count of {feature}")
    return fig

# --- 2. Regression Visualizations ---

def plot_actual_vs_predicted(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions', marker=dict(opacity=0.7)))
    # Add identity line (Perfect Prediction)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Fit', line=dict(color='red', dash='dash')))
    fig.update_layout(title="Actual vs Predicted Values", xaxis_title="Actual Values", yaxis_title="Predicted Values")
    return fig

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals, opacity=0.7, title="Residual Plot", labels={'x': 'Predicted Values', 'y': 'Residuals'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig

# --- 3. Classification Visualizations ---

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true)))
    fig = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale='Blues', title="Confusion Matrix")
    fig.update_layout(xaxis_title="Predicted Class", yaxis_title="Actual Class")
    return fig

def plot_roc_curve_binary(y_true, y_prob):
    # Only applicable for binary classification
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={roc_auc:.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    return fig

# --- 4. Feature Importance ---

def plot_feature_importance(pipeline, original_features):
    try:
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']
        
        # Try to extract transformed feature names from the preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
            # Clean up the names (remove the step prefix like 'num__' or 'cat__')
            feature_names = [name.split('__')[-1] for name in feature_names]
        else:
            feature_names = original_features

        importances = None
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_

        if importances is not None and len(importances) == len(feature_names):
            df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            df_imp['Abs_Importance'] = df_imp['Importance'].abs()
            df_imp = df_imp.sort_values(by='Abs_Importance', ascending=True).tail(20) # Top 20 features
            
            fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', title="Feature Importance / Coefficients")
            return fig
        return None
    except Exception as e:
        return None