from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_classification_model(model_name: str, **kwargs):
    """
    Returns a configured scikit-learn classification model based on the provided name and parameters.
    """
    if model_name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=kwargs.get("n_neighbors", 5))
        
    elif model_name == "Support Vector Machine":
        return SVC(
            kernel=kwargs.get("kernel", "rbf"), 
            C=kwargs.get("C", 1.0), 
            probability=True # Required for ROC curves later
        )
        
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(
            max_depth=kwargs.get("max_depth", None), 
            criterion=kwargs.get("criterion", "gini"),
            random_state=42
        )
        
    elif model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100), 
            max_depth=kwargs.get("max_depth", None),
            random_state=42
        )
        
    else:
        raise ValueError(f"Unsupported classification model: {model_name}")