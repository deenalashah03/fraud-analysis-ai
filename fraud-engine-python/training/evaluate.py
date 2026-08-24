from preprocess import preprocess_data
from xgboost import XGBClassifier
import os
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def save_final_evaluation_result(
        threshold,
        auc_roc,
        auc_pr,
        precision,
        recall,
        f1,
        cm,
        feature_count,
        feature_experiment
):
    """
    Save final XGBoost test-set evaluation result.
    """

    analysis_dir = "../analysis"
    os.makedirs(analysis_dir, exist_ok=True)

    file_path = os.path.join(
        analysis_dir,
        "xgboost_model_evaluation.csv"
    )

    result = {
        "model": "XGBoost",
        "feature_count": feature_count,
        "feature_experiment": feature_experiment,
        "evaluation_stage": "test_final",
        "threshold": threshold,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_negative": cm[0][0],
        "false_positive": cm[0][1],
        "false_negative": cm[1][0],
        "true_positive": cm[1][1]
    }

    result_df = pd.DataFrame([result])

    # Append to existing CSV
    if os.path.exists(file_path):

        result_df.to_csv(
            file_path,
            mode="a",
            header=False,
            index=False
        )

    else:

        result_df.to_csv(
            file_path,
            mode="w",
            header=True,
            index=False
        )

    print(
        f"Final test evaluation saved to: {file_path}"
    )


def evaluate_model():

    # --------------------------------------------------
    # Load preprocessed data
    # --------------------------------------------------

    X_train, X_validation, X_test, \
        y_train, y_validation, y_test = preprocess_data()

    print("Test data shape:", X_test.shape)

    # --------------------------------------------------
    # Feature information
    # --------------------------------------------------

    feature_count = X_test.shape[1]

    feature_experiment = "final_xgboost"

    print(
        f"Features used by final XGBoost: {feature_count}"
    )

    # --------------------------------------------------
    # Load FINAL tuned XGBoost model
    # --------------------------------------------------

    model = XGBClassifier(
        enable_categorical=True
    )

    model.load_model(
        "../model/fraud_xgboost.json"
    )

    print("Final XGBoost model loaded successfully.")

    # --------------------------------------------------
    # Generate TEST probabilities
    # --------------------------------------------------

    y_test_proba = (
        model.predict_proba(X_test)[:, 1]
    )

    # --------------------------------------------------
    # LOCKED THRESHOLD
    #
    # Selected using validation-set F1.
    # --------------------------------------------------

    threshold = 0.4

    y_test_pred = (
            y_test_proba >= threshold
    ).astype(int)

    print(
        f"Final threshold: {threshold}"
    )

    # --------------------------------------------------
    # Probability-based metrics
    # --------------------------------------------------

    auc_roc = roc_auc_score(
        y_test,
        y_test_proba
    )

    auc_pr = average_precision_score(
        y_test,
        y_test_proba
    )

    # --------------------------------------------------
    # Threshold-based metrics
    # --------------------------------------------------

    precision = precision_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_test_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_test_pred
    )

    # --------------------------------------------------
    # FINAL TEST RESULTS
    # --------------------------------------------------

    print(
        "\n========== FINAL XGBOOST TEST EVALUATION =========="
    )

    print(f"Features: {feature_count}")
    print(f"Threshold: {threshold}")
    print(f"ROC-AUC: {auc_roc:.6f}")
    print(f"PR-AUC: {auc_pr:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 Score: {f1:.6f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTrue Negative:", cm[0][0])
    print("False Positive:", cm[0][1])
    print("False Negative:", cm[1][0])
    print("True Positive:", cm[1][1])

    # --------------------------------------------------
    # Save FINAL evaluation
    # --------------------------------------------------

    save_final_evaluation_result(
        threshold=threshold,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        precision=precision,
        recall=recall,
        f1=f1,
        cm=cm,
        feature_count=feature_count,
        feature_experiment=feature_experiment
    )


if __name__ == "__main__":
    evaluate_model()