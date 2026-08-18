from preprocess import preprocess_data
from xgboost import XGBClassifier
import pandas as pd


def analyze_features():

    # Load and preprocess data
    X_train, X_validation, X_test, y_train, y_validation, y_test = preprocess_data()

    print("Training data shape:", X_train.shape)

    # Load trained model
    model = XGBClassifier(
        enable_categorical=True
    )

    model.load_model("../model/fraud_xgboost.json")

    print("Model loaded successfully.")

    # Get feature importance
    importance = model.feature_importances_

    # Create DataFrame
    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importance
    })

    # Sort from highest to lowest
    feature_importance = feature_importance.sort_values(
        by="importance",
        ascending=False
    )

    print("\n========== TOP 30 FEATURES ==========")
    print(feature_importance.head(30).to_string(index=False))

    print("\n========== BOTTOM 30 FEATURES ==========")
    print(feature_importance.tail(30).to_string(index=False))

    # Save results
    feature_importance.to_csv(
        "../analysis/xgboost_feature_importance.csv",
        index=False
    )

    print("\nFeature importance saved successfully.")


if __name__ == "__main__":
    analyze_features()