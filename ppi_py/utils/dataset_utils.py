import pandas as pd

def dataframe_decorator(func):
    def wrapper(data=None, X=None, y=None, yhat=None, X_unlabeled=None, yhat_unlabeled=None):
        # Check if data is a DataFrame and the other arguments are strings
        if isinstance(data, pd.DataFrame) and all(isinstance(arg, str) for arg in [X, y, yhat, X_unlabeled, yhat_unlabeled]):
            # Extract columns from DataFrame
            X_vals = data[X].values
            y_vals = data[y].values
            yhat_vals = data[yhat].values
            X_unlabeled_vals = data[X_unlabeled].values
            yhat_unlabeled_vals = data[yhat_unlabeled].values
            return func(X_vals, y_vals, yhat_vals, X_unlabeled_vals, yhat_unlabeled_vals)
        else:
            # Otherwise, call the function normally
            return func(X, y, yhat, X_unlabeled, yhat_unlabeled)
    return wrapper
