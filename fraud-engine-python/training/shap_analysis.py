import shap
import pandas as pd


def analyze_shap(
        model,
        X_validation,
        model_name
):
    """
    Perform global SHAP analysis for a trained model.

    SHAP analysis is performed ONLY on the validation dataset.

    The purpose of this function is to:
        1. Calculate SHAP values
        2. Calculate global feature importance
        3. Save feature importance results
        4. Display SHAP summary information

    This function does NOT:
        - Train the model
        - Select features
        - Drop features
        - Modify train/validation/test datasets
        - Use the test dataset
        - Perform hyperparameter tuning
        - Select a classification threshold
    """

    # --------------------------------------------------
    # Validate inputs
    # --------------------------------------------------

    if model is None:
        raise ValueError("A trained model must be provided.")

    if X_validation is None:
        raise ValueError("Validation data must be provided.")

    if not model_name:
        raise ValueError("model_name must be provided.")

    print("\n==========================================")
    print(f"SHAP ANALYSIS Started for : {model_name}")
    print("==========================================")

    print(
        "Validation data shape:",
        X_validation.shape
    )

    # --------------------------------------------------
    # Create SHAP Tree Explainer
    # --------------------------------------------------

    # SHAP explains the predictions made by an already
    # trained tree-based model.
    #
    # The model is supplied by the caller.
    #
    # This makes the SHAP logic reusable for:
    #
    #     XGBoost
    #     Random Forest
    #     Other compatible tree-based models

    explainer = shap.TreeExplainer(model)

    # --------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------

    # IMPORTANT:
    #
    # SHAP values are calculated ONLY on validation data.
    #
    # We deliberately do NOT calculate SHAP values on
    # X_test because the test set must remain untouched
    # until final model evaluation.

    shap_values = explainer.shap_values(X_validation)

    print("SHAP values calculated on validation data.")

    # --------------------------------------------------
    # Global feature importance
    # --------------------------------------------------

    # Mean absolute SHAP value represents the average
    # magnitude of a feature's contribution to the model's
    # predictions across the validation dataset.
    #
    # Higher value = greater overall contribution.

    shap_importance = pd.DataFrame({
        "feature": X_validation.columns,
        "importance": abs(shap_values).mean(axis=0)
    })

    shap_importance = shap_importance.sort_values(
        by="importance",
        ascending=False
    )

    # --------------------------------------------------
    # Display top features
    # --------------------------------------------------

    print("\n========== TOP 30 SHAP FEATURES ==========")

    print(
        shap_importance.head(30).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # Display bottom features
    # --------------------------------------------------

    print("\n========== BOTTOM 30 SHAP FEATURES ==========")

    print(
        shap_importance.tail(30).to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # SHAP importance distribution
    # --------------------------------------------------

    print(
        "\n========== SHAP IMPORTANCE DISTRIBUTION =========="
    )

    print(
        "Total features:",
        len(shap_importance)
    )

    print(
        "Importance > 1.0:",
        (shap_importance["importance"] > 1.0).sum()
    )

    print(
        "Importance > 0.5:",
        (shap_importance["importance"] > 0.5).sum()
    )

    print(
        "Importance > 0.1:",
        (shap_importance["importance"] > 0.1).sum()
    )

    print(
        "Importance > 0.01:",
        (shap_importance["importance"] > 0.01).sum()
    )

    print(
        "Importance > 0.001:",
        (shap_importance["importance"] > 0.001).sum()
    )

    print(
        "Importance <= 0.001:",
        (shap_importance["importance"] <= 0.001).sum()
    )

    # --------------------------------------------------
    # Save SHAP feature importance
    # --------------------------------------------------

    # Each model gets its own SHAP importance file.
    #
    # Example:
    #
    #     xgboost_shap_feature_importance.csv
    #     random_forest_shap_feature_importance.csv
    #
    # This is important because SHAP importance is
    # model-specific.

    output_path = (
        f"../analysis/{model_name}_shap_feature_importance.csv"
    )

    shap_importance.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSHAP feature importance saved to: {output_path}"
    )

    # --------------------------------------------------
    # SHAP summary plot
    # --------------------------------------------------

    # The SHAP values and feature data must represent
    # the same observations.
    #
    # Since SHAP was calculated on X_validation,
    # X_validation is used here as well.

    shap.summary_plot(
        shap_values,
        X_validation,
        show=True
    )

    print(
        f"\nSHAP analysis completed for {model_name}."
    )

    return shap_importance