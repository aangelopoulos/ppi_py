import pandas as pd


def dataframe_decorator(func):
    def wrapper(
        data=None,
        X=None,
        Y=None,
        Yhat=None,
        X_unlabeled=None,
        Yhat_unlabeled=None,
    ):
        # Check if data is a DataFrame and the other arguments are strings
        if isinstance(data, pd.DataFrame) and all(
            isinstance(arg, str)
            for arg in [X, Y, Yhat, X_unlabeled, Yhat_unlabeled]
        ):
            # Extract columns from DataFrame
            X_vals = data[X].values
            Y_vals = data[Y].values
            Yhat_vals = data[Yhat].values
            X_unlabeled_vals = data[X_unlabeled].values
            Yhat_unlabeled_vals = data[Yhat_unlabeled].values
            return func(
                X_vals,
                Y_vals,
                Yhat_vals,
                X_unlabeled_vals,
                Yhat_unlabeled_vals,
            )
        else:
            # Otherwise, call the function normally
            return func(X, Y, Yhat, X_unlabeled, Yhat_unlabeled)

    return wrapper
