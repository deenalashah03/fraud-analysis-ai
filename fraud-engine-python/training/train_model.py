from preprocess import preprocess_data
from xgboost import XGBClassifier
def train_model():
    X_train, X_validation, X_test, y_train, y_validation, y_test = preprocess_data()
    print(X_train.shape)
    # Calculate class imbalance(Imbalance because legitimate cases are so much more than fraud. telling model to pay more )
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()
    scale_pos_weight = negative_count / positive_count

    print(f"Negative samples: {negative_count}")
    print(f"Positive samples: {positive_count}")
    print(f"Scale Pos Weight: {scale_pos_weight}")

    # Create XGBoost model
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight, #this means that positive sample be given scale_pos_weigh times weightage than legitimate/negative
        enable_categorical=True,
        tree_method="hist",
        n_estimators=4000,
        early_stopping_rounds=100,
        random_state=42
    )
    # Train model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=True
    )
    print("Best iteration:", model.best_iteration)
    print("Best AUC-PR:", model.best_score)

    model.save_model("../model/fraud_xgboost.json")
    print("Model saved successfully.")
if __name__ == "__main__":
    train_model()