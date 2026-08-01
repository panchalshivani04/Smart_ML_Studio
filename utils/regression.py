from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def get_regression_model(model_name: str, **kwargs):
    """
    Returns a configured scikit-learn regression model based on the provided name and parameters.
    """
    if model_name in ["Linear Regression", "Multiple Linear Regression"]:
        return LinearRegression()
    
    elif model_name == "Polynomial Regression":
        degree = kwargs.get("degree", 2)
        # Using a pipeline to apply polynomial features before linear regression
        return Pipeline([
            ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
            ("lin_reg", LinearRegression())
        ])
        
    else:
        raise ValueError(f"Unsupported regression model: {model_name}")