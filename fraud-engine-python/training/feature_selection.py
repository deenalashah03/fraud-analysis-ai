import pandas as pd


def select_features(
        X_train,
        X_validation,
        X_test,
        shap_importance_path,
        threshold
):
    """
    Select features using global SHAP importance.

    The SHAP importance file must have been generated using
    validation data.

    The selected feature list is then applied identically to:

        X_train
        X_validation
        X_test

    Important:
        The test dataset is NOT used to determine which features
        are selected.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training features.

    X_validation : pandas.DataFrame
        Validation features.

    X_test : pandas.DataFrame
        Test features.

    shap_importance_path : str
        Path to the SHAP feature importance CSV.

    threshold : float
        Features with SHAP importance <= threshold are removed.

    Returns
    -------
    X_train_selected : pandas.DataFrame
    X_validation_selected : pandas.DataFrame
    X_test_selected : pandas.DataFrame
    selected_features : list
        List of features retained after selection.
    """

    # --------------------------------------------------
    # Load SHAP feature importance
    # --------------------------------------------------

    shap_importance = pd.read_csv(
        shap_importance_path
    )

    # --------------------------------------------------
    # Validate SHAP importance file
    # --------------------------------------------------

    required_columns = {
        "feature",
        "importance"
    }

    if not required_columns.issubset(
            shap_importance.columns
    ):
        raise ValueError(
            "SHAP importance file must contain "
            "'feature' and 'importance' columns."
        )

    # --------------------------------------------------
    # Determine features to remove
    # --------------------------------------------------

    features_to_remove = shap_importance[
        shap_importance["importance"] <= threshold
        ]["feature"].tolist()

    # --------------------------------------------------
    # Determine features to keep
    # --------------------------------------------------

    selected_features = [
        feature
        for feature in X_train.columns
        if feature not in features_to_remove
    ]

    # --------------------------------------------------
    # Apply SAME feature list to all datasets
    # --------------------------------------------------

    X_train_selected = X_train[
        selected_features
    ].copy()

    X_validation_selected = X_validation[
        selected_features
    ].copy()

    X_test_selected = X_test[
        selected_features
    ].copy()

    # --------------------------------------------------
    # Print feature-selection results
    # --------------------------------------------------

    print("\n========== FEATURE SELECTION ==========")

    print(
        "Original feature count:",
        X_train.shape[1]
    )

    print(
        "Features removed:",
        len(features_to_remove)
    )

    print(
        "SHAP threshold:",
        threshold
    )

    print(
        "Selected feature count:",
        len(selected_features)
    )

    print(
        "Training shape after selection:",
        X_train_selected.shape
    )

    print(
        "Validation shape after selection:",
        X_validation_selected.shape
    )

    print(
        "Test shape after selection:",
        X_test_selected.shape
    )

    return (
        X_train_selected,
        X_validation_selected,
        X_test_selected,
        selected_features
    )


if __name__ == "__main__":
    print(
        "feature_selection.py provides reusable "
        "feature-selection functionality."
    )