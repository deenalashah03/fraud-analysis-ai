import pandas as pd


def prepare_inference_data(
        transaction: dict
) -> pd.DataFrame:
    """
    Convert an incoming transaction dictionary into
    a DataFrame and prepare categorical columns for
    model inference.
    """

    X = pd.DataFrame([transaction])

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:
        X[column] = (
            X[column]
            .fillna("MISSING")
            .astype("category")
        )

    return X