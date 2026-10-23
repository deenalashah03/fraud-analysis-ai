import json

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "models" / "xgboost" / "fraud_xgboost_config.json"

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

    # Load selected features
    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    selected_features = config["selected_features"]

    # Make sure all required features exist
    missing_features = [
        feature
        for feature in selected_features
        if feature not in X.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Keep only features used by the trained model
    X = X[selected_features]

    return X