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
    confusion_matrix,
    classification_report
)

def save_evaluation_result(
        model_name,
        threshold,
        auc_roc,
        auc_pr,
        precision,
        recall,
        f1,
        cm
):
    analysis_dir = "../analysis"
    os.makedirs(analysis_dir, exist_ok=True)

    file_path = os.path.join(
        analysis_dir,
        "xgboost_model_evaluation.csv"
    )

    result = {
        "model": model_name,
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

    # Append if file already exists
    # Otherwise create a new file with headers
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

    print(f"Evaluation result saved to: {file_path}")
def evaluate_model():

    # Load and preprocess data
    X_train, X_validation, X_test, y_train, y_validation, y_test = preprocess_data()

    print("Test data shape:", X_test.shape)

    # Load trained model
    model = XGBClassifier(
        enable_categorical=True
    )

    model.load_model("../model/fraud_xgboost.json")

    print("Model loaded successfully.")

    # Generate probability predictions
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to class predictions
    threshold = 0.9

    y_test_pred = (y_test_proba >= threshold).astype(int)
    # --------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------

    auc_roc = roc_auc_score(y_test, y_test_proba)

    auc_pr = average_precision_score(y_test, y_test_proba)

    precision = precision_score(y_test, y_test_pred)

    recall = recall_score(y_test, y_test_pred)

    f1 = f1_score(y_test, y_test_pred)

    cm = confusion_matrix(y_test, y_test_pred)

    save_evaluation_result(
        model_name="XGBoost",
        threshold=threshold,
        auc_roc=auc_roc,
        auc_pr=auc_pr,
        precision=precision,
        recall=recall,
        f1=f1,
        cm=cm
    )

if __name__ == "__main__":
    evaluate_model()