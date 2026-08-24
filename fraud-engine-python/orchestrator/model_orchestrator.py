import json
import os
from typing import Any, TypedDict

import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from training.preprocess import preprocess_data
from training.shap_analysis import analyze_shap
from training.feature_selection import select_features

from models.xgboost.xgboost_model import XGBoostModel


# ==================================================
# MODEL REGISTRY CONFIGURATION
# ==================================================


class ModelConfig(TypedDict):
    model_class: type
    baseline_params: dict[str, Any]
    model_path: str
    config_path: str
    evaluation_file: str


# ==================================================
# MODEL REGISTRY
# ==================================================
#
# Each model has its own:
#
#     - model implementation
#     - baseline parameters
#     - final model artifact path
#     - inference configuration path
#     - experiment/evaluation CSV
#
# Random Forest can be added later as another entry.
# ==================================================

MODEL_REGISTRY: dict[str, ModelConfig] = {

    "xgboost": {

        "model_class": XGBoostModel,

        # --------------------------------------------------
        # TRUE BASELINE PARAMETERS
        # --------------------------------------------------
        #
        # These are the baseline/default parameters.
        # They are NOT the tuned final parameters.
        #
        "baseline_params": {
            "max_depth": 6,
            "learning_rate": 0.3,
            "min_child_weight": 1,
            "n_estimators": 4000,
            "early_stopping_rounds": 100
        },

        # --------------------------------------------------
        # Final model artifact
        # --------------------------------------------------

        "model_path": (
            "../models/xgboost/"
            "fraud_xgboost.json"
        ),

        # --------------------------------------------------
        # Final inference configuration
        # --------------------------------------------------

        "config_path": (
            "../models/xgboost/"
            "fraud_xgboost_config.json"
        ),

        # --------------------------------------------------
        # XGBoost experiment/evaluation ledger
        # --------------------------------------------------

        "evaluation_file": (
            "../analysis/"
            "xgboost_model_evaluation.csv"
        )
    }
}


class ModelOrchestrator:

    # ==================================================
    # MODEL LOOKUP
    # ==================================================

    def _get_model_config(
            self,
            model_name: str
    ) -> ModelConfig:
        """
        Return the configuration associated with a model name.
        """

        if model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        return MODEL_REGISTRY[model_name]

    def _get_model_class(
            self,
            model_name: str
    ) -> type:
        """
        Return the model implementation associated with
        the supplied model name.
        """

        model_config = self._get_model_config(
            model_name
        )

        return model_config["model_class"]

    # ==================================================
    # RECORD EXPERIMENT
    # ==================================================

    def record_experiment(
            self,
            model_name: str,
            model: Any,
            experiment_type: str,
            feature_count: int,
            feature_experiment: str,
            evaluation_stage: str,
            shap_threshold: float | None = None,
            threshold: float | None = None,
            auc_roc: float | None = None,
            auc_pr: float | None = None,
            precision: float | None = None,
            recall: float | None = None,
            f1_score: float | None = None,
            confusion_matrix: Any | None = None
    ) -> None:
        """
        Record one non-SHAP experiment/evaluation result.

        SHAP has its own feature-importance CSV.

        One execution = one row in the model-specific
        experiment/evaluation CSV.
        """

        model_config = self._get_model_config(
            model_name
        )

        file_path = model_config[
            "evaluation_file"
        ]

        result = {
            "model": model_name,

            "experiment_type": experiment_type,

            "feature_count": feature_count,

            "feature_experiment": feature_experiment,

            "evaluation_stage": evaluation_stage,

            # Feature-selection information
            "shap_threshold": shap_threshold,

            # --------------------------------------------------
            # XGBoost-specific parameters
            # --------------------------------------------------
            #
            # When Random Forest is added, its own recording
            # schema can be introduced without forcing RF into
            # these XGBoost-specific columns.
            #
            "max_depth": getattr(
                model,
                "max_depth",
                None
            ),

            "learning_rate": getattr(
                model,
                "learning_rate",
                None
            ),

            "min_child_weight": getattr(
                model,
                "min_child_weight",
                None
            ),

            "n_estimators": getattr(
                model,
                "n_estimators",
                None
            ),

            "early_stopping_rounds": getattr(
                model,
                "early_stopping_rounds",
                None
            ),

            "scale_pos_weight": getattr(
                model,
                "scale_pos_weight",
                None
            ),

            "best_iteration": getattr(
                model,
                "best_iteration",
                None
            ),

            # Classification threshold
            "threshold": threshold,

            # Metrics
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,

            # Confusion matrix
            "true_negative": None,
            "false_positive": None,
            "false_negative": None,
            "true_positive": None
        }

        # --------------------------------------------------
        # Add confusion matrix when available
        # --------------------------------------------------

        if confusion_matrix is not None:

            result["true_negative"] = (
                confusion_matrix[0][0]
            )

            result["false_positive"] = (
                confusion_matrix[0][1]
            )

            result["false_negative"] = (
                confusion_matrix[1][0]
            )

            result["true_positive"] = (
                confusion_matrix[1][1]
            )

        result_df = pd.DataFrame([result])

        # --------------------------------------------------
        # Ensure analysis directory exists
        # --------------------------------------------------

        os.makedirs(
            "../analysis",
            exist_ok=True
        )

        # --------------------------------------------------
        # Save / append
        # --------------------------------------------------

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
            f"Experiment result saved to: {file_path}"
        )

    # ==================================================
    # 1. BASELINE
    # ==================================================

    def run_baseline(
            self,
            model_name: str
    ) -> dict[str, Any]:
        """
        Train the baseline model.

        Baseline parameters come from MODEL_REGISTRY.

        The trained model remains in memory so SHAP can
        consume the same model instance.
        """

        # --------------------------------------------------
        # Get model configuration
        # --------------------------------------------------

        model_config = self._get_model_config(
            model_name
        )

        model_class = model_config[
            "model_class"
        ]

        baseline_params = model_config[
            "baseline_params"
        ]

        # --------------------------------------------------
        # Load original data
        # --------------------------------------------------

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test
        ) = preprocess_data()

        print(
            "Training data shape:",
            X_train.shape
        )

        print(
            "Validation data shape:",
            X_validation.shape
        )

        print(
            "Test data shape:",
            X_test.shape
        )

        # --------------------------------------------------
        # Create model
        # --------------------------------------------------

        model = model_class(
            **baseline_params
        )

        # --------------------------------------------------
        # Train
        # --------------------------------------------------

        training_result = model.train(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation
        )

        # --------------------------------------------------
        # Baseline validation probabilities
        # --------------------------------------------------

        y_validation_proba = (
            model.predict_proba(
                X_validation
            )
        )

        # --------------------------------------------------
        # Probability-based metrics
        # --------------------------------------------------

        auc_roc = roc_auc_score(
            y_validation,
            y_validation_proba
        )

        auc_pr = average_precision_score(
            y_validation,
            y_validation_proba
        )

        # --------------------------------------------------
        # Baseline reference threshold
        # --------------------------------------------------
        #
        # 0.5 is only a reference threshold for descriptive
        # baseline classification metrics.
        #
        # It is NOT the final locked threshold.
        #

        baseline_threshold = 0.5

        y_validation_pred = (
                y_validation_proba >= baseline_threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            y_validation_pred,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            y_validation_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            y_validation_pred,
            zero_division=0
        )

        cm = confusion_matrix(
            y_validation,
            y_validation_pred
        )

        # --------------------------------------------------
        # Record baseline
        # --------------------------------------------------

        self.record_experiment(
            model_name=model_name,
            model=model,
            experiment_type="baseline",
            feature_count=X_train.shape[1],
            feature_experiment="baseline",
            evaluation_stage="validation",
            threshold=baseline_threshold,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm
        )

        return {
            "model": model,

            "training_result": training_result,

            "X_train": X_train,
            "X_validation": X_validation,
            "X_test": X_test,

            "y_train": y_train,
            "y_validation": y_validation,
            "y_test": y_test
        }

    # ==================================================
    # 2. SHAP
    # ==================================================

    def run_shap(
            self,
            model_name: str,
            model: Any,
            X_validation: pd.DataFrame
    ):
        """
        Run SHAP on an already-trained model.

        SHAP uses validation data only.

        SHAP writes its own feature-importance file.
        """

        # --------------------------------------------------
        # Resolve registered model class
        # --------------------------------------------------

        model_class = self._get_model_class(
            model_name
        )

        # --------------------------------------------------
        # Validate supplied model
        # --------------------------------------------------

        if not isinstance(
                model,
                model_class
        ):
            raise TypeError(
                f"Expected {model_class.__name__}, "
                f"but received {type(model).__name__}"
            )

        if model.model is None:

            raise RuntimeError(
                "The supplied model has not been trained "
                "or loaded."
            )

        # --------------------------------------------------
        # Run SHAP
        # --------------------------------------------------
        #
        # model       = XGBoostModel wrapper
        # model.model = actual XGBClassifier
        #
        # TreeExplainer needs the actual classifier.

        return analyze_shap(
            model=model.model,
            X_validation=X_validation,
            model_name=model_name
        )

    # ==================================================
    # 3. FEATURE-SELECTION EXPERIMENT
    # ==================================================

    def run_feature_selection_experiment(
            self,
            model_name: str,
            threshold: float
    ) -> dict[str, Any]:
        """
        Run ONE feature-selection experiment.

        Original data
            ↓
        candidate SHAP threshold
            ↓
        selected features
            ↓
        NEW baseline-configured model
            ↓
        train from scratch
            ↓
        validation AUC-PR
            ↓
        record experiment
        """

        # --------------------------------------------------
        # Get model configuration
        # --------------------------------------------------

        model_config = self._get_model_config(
            model_name
        )

        model_class = model_config[
            "model_class"
        ]

        baseline_params = model_config[
            "baseline_params"
        ]

        # --------------------------------------------------
        # Load original data
        # --------------------------------------------------

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test
        ) = preprocess_data()

        # --------------------------------------------------
        # SHAP importance file
        # --------------------------------------------------

        shap_importance_path = (
            f"../analysis/"
            f"{model_name}_shap_feature_importance.csv"
        )

        # --------------------------------------------------
        # Apply candidate threshold
        # --------------------------------------------------

        (
            X_train_selected,
            X_validation_selected,
            X_test_selected,
            selected_features
        ) = select_features(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            shap_importance_path=shap_importance_path,
            threshold=threshold
        )

        print(
            "\n========== FEATURE SELECTION EXPERIMENT =========="
        )

        print(
            "Original feature count:",
            X_train.shape[1]
        )

        print(
            "Selected feature count:",
            len(selected_features)
        )

        print(
            "SHAP threshold:",
            threshold
        )

        # --------------------------------------------------
        # Create NEW model
        # --------------------------------------------------

        model = model_class(
            **baseline_params
        )

        # --------------------------------------------------
        # Train from scratch
        # --------------------------------------------------

        training_result = model.train(
            X_train=X_train_selected,
            y_train=y_train,
            X_validation=X_validation_selected,
            y_validation=y_validation
        )

        # --------------------------------------------------
        # Validation PR-AUC
        # --------------------------------------------------

        y_validation_proba = (
            model.predict_proba(
                X_validation_selected
            )
        )

        auc_pr = average_precision_score(
            y_validation,
            y_validation_proba
        )

        # --------------------------------------------------
        # Record
        # --------------------------------------------------

        self.record_experiment(
            model_name=model_name,
            model=model,
            experiment_type="feature_selection",
            feature_count=len(selected_features),
            feature_experiment=(
                f"shap_threshold_{threshold}"
            ),
            evaluation_stage="validation",
            shap_threshold=threshold,
            auc_pr=auc_pr
        )

        return {
            "model": model,
            "selected_features": selected_features,
            "feature_count": len(selected_features),
            "threshold": threshold,
            "training_result": training_result,
            "validation_auc_pr": auc_pr,
            "X_train": X_train_selected,
            "X_validation": X_validation_selected,
            "X_test": X_test_selected,
            "y_train": y_train,
            "y_validation": y_validation,
            "y_test": y_test
        }

    # ==================================================
    # 4. HYPERPARAMETER EXPERIMENT
    # ==================================================

    def run_hyperparameter_experiment(
            self,
            model_name: str,
            shap_threshold: float,
            params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Run ONE hyperparameter experiment.

        Original data
            ↓
        locked feature-selection threshold
            ↓
        selected feature set
            ↓
        NEW model with candidate parameters
            ↓
        train from scratch
            ↓
        validation AUC-PR
            ↓
        record experiment
        """

        model_class = self._get_model_class(
            model_name
        )

        # --------------------------------------------------
        # Load original data
        # --------------------------------------------------

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test
        ) = preprocess_data()

        # --------------------------------------------------
        # Apply locked feature-selection decision
        # --------------------------------------------------

        shap_importance_path = (
            f"../analysis/"
            f"{model_name}_shap_feature_importance.csv"
        )

        (
            X_train_selected,
            X_validation_selected,
            X_test_selected,
            selected_features
        ) = select_features(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            shap_importance_path=shap_importance_path,
            threshold=shap_threshold
        )

        print(
            "\n========== HYPERPARAMETER EXPERIMENT =========="
        )

        print(
            "Selected feature count:",
            len(selected_features)
        )

        print(
            "Locked SHAP threshold:",
            shap_threshold
        )

        print(
            "Candidate parameters:",
            params
        )

        # --------------------------------------------------
        # Create NEW model
        # --------------------------------------------------

        model = model_class(
            **params
        )

        # --------------------------------------------------
        # Train from scratch
        # --------------------------------------------------

        training_result = model.train(
            X_train=X_train_selected,
            y_train=y_train,
            X_validation=X_validation_selected,
            y_validation=y_validation
        )

        # --------------------------------------------------
        # Validation PR-AUC
        # --------------------------------------------------

        y_validation_proba = (
            model.predict_proba(
                X_validation_selected
            )
        )

        auc_pr = average_precision_score(
            y_validation,
            y_validation_proba
        )

        print(
            "Validation AUC-PR:",
            auc_pr
        )

        # --------------------------------------------------
        # Record
        # --------------------------------------------------

        self.record_experiment(
            model_name=model_name,
            model=model,
            experiment_type="hyperparameter",
            feature_count=len(selected_features),
            feature_experiment=(
                f"shap_threshold_{shap_threshold}"
            ),
            evaluation_stage="validation",
            shap_threshold=shap_threshold,
            auc_pr=auc_pr
        )

        return {
            "model": model,
            "selected_features": selected_features,
            "feature_count": len(selected_features),
            "shap_threshold": shap_threshold,
            "params": params,
            "training_result": training_result,
            "validation_auc_pr": auc_pr,
            "X_train": X_train_selected,
            "X_validation": X_validation_selected,
            "X_test": X_test_selected,
            "y_train": y_train,
            "y_validation": y_validation,
            "y_test": y_test
        }

    # ==================================================
    # 5. FINAL MODEL TRAINING
    # ==================================================

    def train_final(
            self,
            model_name: str,
            shap_threshold: float,
            params: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Train the final model using locked feature-selection
        and locked hyperparameters.

        This is a fresh training run.

        The final model is saved to the model path defined
        in MODEL_REGISTRY.

        Inference configuration is saved separately AFTER
        the classification threshold has been locked.
        """

        model_config = self._get_model_config(
            model_name
        )

        model_class = model_config[
            "model_class"
        ]

        model_path = model_config[
            "model_path"
        ]

        # --------------------------------------------------
        # Load original data
        # --------------------------------------------------

        (
            X_train,
            X_validation,
            X_test,
            y_train,
            y_validation,
            y_test
        ) = preprocess_data()

        # --------------------------------------------------
        # Apply locked feature selection
        # --------------------------------------------------

        shap_importance_path = (
            f"../analysis/"
            f"{model_name}_shap_feature_importance.csv"
        )

        (
            X_train_selected,
            X_validation_selected,
            X_test_selected,
            selected_features
        ) = select_features(
            X_train=X_train,
            X_validation=X_validation,
            X_test=X_test,
            shap_importance_path=shap_importance_path,
            threshold=shap_threshold
        )

        print(
            "\n========== FINAL MODEL TRAINING =========="
        )

        print(
            "Feature count:",
            len(selected_features)
        )

        print(
            "SHAP threshold:",
            shap_threshold
        )

        print(
            "Final parameters:",
            params
        )

        # --------------------------------------------------
        # Create NEW final model
        # --------------------------------------------------

        model = model_class(
            **params
        )

        # --------------------------------------------------
        # Train from scratch
        # --------------------------------------------------

        training_result = model.train(
            X_train=X_train_selected,
            y_train=y_train,
            X_validation=X_validation_selected,
            y_validation=y_validation
        )

        # --------------------------------------------------
        # Record final training
        # --------------------------------------------------

        self.record_experiment(
            model_name=model_name,
            model=model,
            experiment_type="final_training",
            feature_count=len(selected_features),
            feature_experiment=(
                f"shap_threshold_{shap_threshold}"
            ),
            evaluation_stage="validation",
            shap_threshold=shap_threshold,
            auc_pr=model.validation_auc_pr
        )

        # --------------------------------------------------
        # Save final model
        # --------------------------------------------------

        model.save(
            model_path
        )

        print(
            f"Final {model_name} model saved to: "
            f"{model_path}"
        )

        return {
            "model": model,
            "selected_features": selected_features,
            "feature_count": len(selected_features),
            "shap_threshold": shap_threshold,
            "params": params,
            "training_result": training_result,
            "X_train": X_train_selected,
            "X_validation": X_validation_selected,
            "X_test": X_test_selected,
            "y_train": y_train,
            "y_validation": y_validation,
            "y_test": y_test
        }

    # ==================================================
    # 6. THRESHOLD EXPERIMENT
    # ==================================================

    def run_threshold_experiment(
            self,
            model_name: str,
            model: Any,
            X_validation: pd.DataFrame,
            y_validation: pd.Series,
            threshold: float
    ) -> dict[str, Any]:
        """
        Run ONE classification-threshold experiment.

        The final model is NOT retrained.

        Threshold selection uses validation data only.
        """

        model_class = self._get_model_class(
            model_name
        )

        # --------------------------------------------------
        # Validate supplied model
        # --------------------------------------------------

        if not isinstance(
                model,
                model_class
        ):
            raise TypeError(
                f"Expected {model_class.__name__}, "
                f"but received {type(model).__name__}"
            )

        if model.model is None:

            raise RuntimeError(
                "The supplied model has not been trained "
                "or loaded."
            )

        # --------------------------------------------------
        # Validation probabilities
        # --------------------------------------------------

        y_validation_proba = (
            model.predict_proba(
                X_validation
            )
        )

        # --------------------------------------------------
        # Probability metrics
        # --------------------------------------------------

        auc_roc = roc_auc_score(
            y_validation,
            y_validation_proba
        )

        auc_pr = average_precision_score(
            y_validation,
            y_validation_proba
        )

        # --------------------------------------------------
        # Apply candidate threshold
        # --------------------------------------------------

        y_validation_pred = (
                y_validation_proba >= threshold
        ).astype(int)

        # --------------------------------------------------
        # Threshold metrics
        # --------------------------------------------------

        precision = precision_score(
            y_validation,
            y_validation_pred,
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            y_validation_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            y_validation_pred,
            zero_division=0
        )

        cm = confusion_matrix(
            y_validation,
            y_validation_pred
        )

        # --------------------------------------------------
        # Print results
        # --------------------------------------------------

        print(
            "\n========== THRESHOLD EXPERIMENT =========="
        )

        print(
            "Threshold:",
            threshold
        )

        print(
            f"ROC-AUC: {auc_roc:.6f}"
        )

        print(
            f"PR-AUC: {auc_pr:.6f}"
        )

        print(
            f"Precision: {precision:.6f}"
        )

        print(
            f"Recall: {recall:.6f}"
        )

        print(
            f"F1 Score: {f1:.6f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(cm)

        # --------------------------------------------------
        # Record threshold experiment
        # --------------------------------------------------

        self.record_experiment(
            model_name=model_name,
            model=model,
            experiment_type="threshold",
            feature_count=X_validation.shape[1],
            feature_experiment="final_xgboost",
            evaluation_stage="validation_threshold",
            threshold=threshold,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm
        )

        return {
            "threshold": threshold,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }

    # ==================================================
    # 7. SAVE INFERENCE CONFIGURATION
    # ==================================================

    def save_inference_config(
            self,
            model_name: str,
            selected_features: list[str],
            feature_selection_threshold: float,
            classification_threshold: float
    ) -> None:
        """
        Save the locked configuration required for runtime
        inference.

        This should be called only after:
            - feature selection is locked
            - hyperparameters are locked
            - classification threshold is locked
        """

        model_config = self._get_model_config(
            model_name
        )

        config_path = model_config[
            "config_path"
        ]

        config = {
            "model_name": model_name,

            "feature_selection_threshold": (
                feature_selection_threshold
            ),

            "classification_threshold": (
                classification_threshold
            ),

            "feature_count": (
                len(selected_features)
            ),

            "selected_features": selected_features
        }

        os.makedirs(
            os.path.dirname(config_path),
            exist_ok=True
        )

        with open(
                config_path,
                "w",
                encoding="utf-8"
        ) as file:

            json.dump(
                config,
                file,
                indent=4
            )

        print(
            f"Inference configuration saved to: "
            f"{config_path}"
        )

    # ==================================================
    # 8. FINAL TEST EVALUATION
    # ==================================================

    def evaluate_final_model(
            self,
            model_name: str,
            model: Any,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            threshold: float,
            feature_experiment: str
    ) -> dict[str, Any]:
        """
        Perform final evaluation on the untouched test set.

        At this point:
            - feature selection is locked
            - hyperparameters are locked
            - threshold is locked

        No training or tuning occurs here.
        """

        model_class = self._get_model_class(
            model_name
        )

        # --------------------------------------------------
        # Validate model
        # --------------------------------------------------

        if not isinstance(
                model,
                model_class
        ):
            raise TypeError(
                f"Expected {model_class.__name__}, "
                f"but received {type(model).__name__}"
            )

        if model.model is None:

            raise RuntimeError(
                "The supplied model has not been trained "
                "or loaded."
            )

        # --------------------------------------------------
        # Test probabilities
        # --------------------------------------------------

        y_test_proba = (
            model.predict_proba(
                X_test
            )
        )

        # --------------------------------------------------
        # Probability metrics
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
        # Apply locked threshold
        # --------------------------------------------------

        y_test_pred = (
                y_test_proba >= threshold
        ).astype(int)

        # --------------------------------------------------
        # Classification metrics
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

        print(
            "\n========== FINAL TEST EVALUATION =========="
        )

        print(
            "Feature count:",
            X_test.shape[1]
        )

        print(
            "Locked threshold:",
            threshold
        )

        print(
            f"ROC-AUC: {auc_roc:.6f}"
        )

        print(
            f"PR-AUC: {auc_pr:.6f}"
        )

        print(
            f"Precision: {precision:.6f}"
        )

        print(
            f"Recall: {recall:.6f}"
        )

        print(
            f"F1 Score: {f1:.6f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(cm)

        # --------------------------------------------------
        # Record final test evaluation
        # --------------------------------------------------

        self.record_experiment(
            model_name=model_name,
            model=model,
            experiment_type="final_evaluation",
            feature_count=X_test.shape[1],
            feature_experiment=feature_experiment,
            evaluation_stage="test_final",
            threshold=threshold,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=cm
        )

        return {
            "threshold": threshold,
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }

    # ==================================================
    # 9. RUNTIME PREDICTION
    # ==================================================

    def predict(
            self,
            model_name: str,
            X: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Run inference using a finalized model.

        Runtime inference does NOT:
            - train
            - run SHAP
            - run feature-selection experiments
            - tune hyperparameters
            - select a threshold

        The selected feature list and classification threshold
        come from the finalized inference configuration.
        """

        # --------------------------------------------------
        # Get model configuration
        # --------------------------------------------------

        model_config = self._get_model_config(
            model_name
        )

        model_class = model_config[
            "model_class"
        ]

        model_path = model_config[
            "model_path"
        ]

        config_path = model_config[
            "config_path"
        ]

        # --------------------------------------------------
        # Validate config file
        # --------------------------------------------------

        if not os.path.exists(
                config_path
        ):
            raise FileNotFoundError(
                "Inference configuration not found: "
                f"{config_path}"
            )

        # --------------------------------------------------
        # Load inference configuration
        # --------------------------------------------------

        with open(
                config_path,
                "r",
                encoding="utf-8"
        ) as file:

            inference_config = json.load(
                file
            )

        selected_features = inference_config[
            "selected_features"
        ]

        classification_threshold = (
            inference_config[
                "classification_threshold"
            ]
        )

        # --------------------------------------------------
        # Validate input
        # --------------------------------------------------

        if X is None:

            raise ValueError(
                "Inference input cannot be None."
            )

        # --------------------------------------------------
        # Check required features
        # --------------------------------------------------

        missing_features = [
            feature
            for feature in selected_features
            if feature not in X.columns
        ]

        if missing_features:

            raise ValueError(
                "Inference input is missing required "
                f"features: {missing_features}"
            )

        # --------------------------------------------------
        # Select features in exact training order
        # --------------------------------------------------

        X_selected = X[
            selected_features
        ].copy()

        # --------------------------------------------------
        # Create model wrapper
        # --------------------------------------------------
        #
        # The constructor parameters are needed only to
        # instantiate the wrapper. The actual learned model
        # is loaded immediately afterward from the final
        # model artifact.

        model = model_class(
            **model_config["baseline_params"]
        )

        # --------------------------------------------------
        # Validate model artifact
        # --------------------------------------------------

        if not os.path.exists(
                model_path
        ):
            raise FileNotFoundError(
                f"Final model not found: {model_path}"
            )

        # --------------------------------------------------
        # Load finalized model
        # --------------------------------------------------

        model.load(
            model_path
        )

        # --------------------------------------------------
        # Generate probability
        # --------------------------------------------------

        probabilities = (
            model.predict_proba(
                X_selected
            )
        )

        # --------------------------------------------------
        # Apply locked threshold
        # --------------------------------------------------

        predictions = (
                probabilities >= classification_threshold
        ).astype(int)

        # --------------------------------------------------
        # Return inference result
        # --------------------------------------------------

        return {
            "model": model_name,
            "probability": probabilities,
            "prediction": predictions,
            "threshold": classification_threshold
        }


# ======================================================
# MANUAL DEVELOPMENT ENTRY POINT
# ======================================================

def main():

    orchestrator = ModelOrchestrator()

    # ==================================================
    # 1. BASELINE
    # ==================================================
    #
    # Uncomment this section only when you want to rerun
    # the baseline.
    #
    # 432 original features
    # + baseline parameters
    # → baseline model
    # → validation metrics
    # → record CSV
    #
    # --------------------------------------------------

    # baseline_result = orchestrator.run_baseline(
    #     model_name="xgboost"
    # )

    # print(
    #     "\n========== BASELINE COMPLETE =========="
    # )

    # print(
    #     "Best iteration:",
    #     baseline_result["training_result"][
    #         "best_iteration"
    #     ]
    # )

    # print(
    #     "Validation AUC-PR:",
    #     baseline_result["training_result"][
    #         "validation_auc_pr"
    #     ]
    # )


    # ==================================================
    # 2. SHAP
    # ==================================================
    #
    # SHAP is performed on the baseline model.
    #
    # SHAP writes its own:
    #
    #     xgboost_shap_feature_importance.csv
    #
    # --------------------------------------------------

    # shap_importance = orchestrator.run_shap(
    #     model_name="xgboost",
    #     model=baseline_result["model"],
    #     X_validation=baseline_result["X_validation"]
    # )

    # print(
    #     "\n========== SHAP COMPLETE =========="
    # )

    # print(
    #     "Features analyzed:",
    #     len(shap_importance)
    # )


    # ==================================================
    # 3. FEATURE-SELECTION EXPERIMENT
    # ==================================================
    #
    # Run ONE threshold at a time.
    #
    # Example:
    #
    #     0.001
    #     0.005
    #     0.010
    #     0.020
    #
    # Each experiment starts from the original 432
    # features and trains a NEW model from scratch.
    #
    # --------------------------------------------------

    # feature_selection_result = (
    #     orchestrator.run_feature_selection_experiment(
    #         model_name="xgboost",
    #         threshold=0.01
    #     )
    # )

    # print(
    #     "\n========== FEATURE SELECTION COMPLETE =========="
    # )

    # print(
    #     "Feature count:",
    #     feature_selection_result["feature_count"]
    # )

    # print(
    #     "Validation AUC-PR:",
    #     feature_selection_result[
    #         "validation_auc_pr"
    #     ]
    # )


    # ==================================================
    # 4. HYPERPARAMETER EXPERIMENT
    # ==================================================
    #
    # Feature-selection decision is locked:
    #
    #     SHAP threshold = 0.01
    #     Features = 202
    #
    # Run ONE candidate configuration at a time.
    #
    # --------------------------------------------------

    # hyperparameter_result = (
    #     orchestrator.run_hyperparameter_experiment(
    #         model_name="xgboost",
    #         shap_threshold=0.01,
    #         params={
    #             "max_depth": 10,
    #             "learning_rate": 0.3,
    #             "min_child_weight": 1,
    #             "n_estimators": 4000,
    #             "early_stopping_rounds": 100
    #         }
    #     )
    # )

    # print(
    #     "\n========== HYPERPARAMETER COMPLETE =========="
    # )

    # print(
    #     "Feature count:",
    #     hyperparameter_result["feature_count"]
    # )

    # print(
    #     "Best iteration:",
    #     hyperparameter_result["training_result"][
    #         "best_iteration"
    #     ]
    # )

    # print(
    #     "Validation AUC-PR:",
    #     hyperparameter_result[
    #         "validation_auc_pr"
    #     ]
    # )


    # ==================================================
    # 5. FINAL MODEL TRAINING
    # ==================================================
    #
    # Feature selection and hyperparameters are locked.
    #
    # This creates a FRESH final model and saves:
    #
    #     fraud_xgboost.json
    #
    # --------------------------------------------------

    final_result = orchestrator.train_final(
        model_name="xgboost",
        shap_threshold=0.01,
        params={
            "max_depth": 10,
            "learning_rate": 0.3,
            "min_child_weight": 1,
            "n_estimators": 4000,
            "early_stopping_rounds": 100
        }
    )

    print(
        "\n========== FINAL MODEL COMPLETE =========="
    )

    print(
        "Feature count:",
        final_result["feature_count"]
    )

    print(
        "Best iteration:",
        final_result["training_result"][
            "best_iteration"
        ]
    )

    print(
        "Validation AUC-PR:",
        final_result["training_result"][
            "validation_auc_pr"
        ]
    )


    # ==================================================
    # 6. THRESHOLD EXPERIMENT
    # ==================================================
    #
    # The final model is already trained.
    #
    # No model retraining happens here.
    #
    # Run ONE candidate threshold at a time.
    #
    # --------------------------------------------------

    threshold_result = (
        orchestrator.run_threshold_experiment(
            model_name="xgboost",
            model=final_result["model"],
            X_validation=final_result["X_validation"],
            y_validation=final_result["y_validation"],
            threshold=0.4
        )
    )

    print(
        "\n========== THRESHOLD COMPLETE =========="
    )

    print(
        "Threshold:",
        threshold_result["threshold"]
    )

    print(
        "Precision:",
        threshold_result["precision"]
    )

    print(
        "Recall:",
        threshold_result["recall"]
    )

    print(
        "F1:",
        threshold_result["f1_score"]
    )


    # ==================================================
    # 7. SAVE INFERENCE CONFIGURATION
    # ==================================================
    #
    # Threshold is now locked.
    #
    # The finalized model + locked runtime configuration
    # are now persisted for FastAPI/runtime use.
    #
    # --------------------------------------------------

    orchestrator.save_inference_config(
        model_name="xgboost",
        selected_features=(
            final_result["selected_features"]
        ),
        feature_selection_threshold=0.01,
        classification_threshold=0.4
    )

    # ==================================================
    # 8. FINAL TEST EVALUATION
    # ==================================================
    #
    # Test data is used ONLY here.
    #
    # No training or tuning occurs here.
    #
    # --------------------------------------------------

    final_test_result = (
        orchestrator.evaluate_final_model(
            model_name="xgboost",
            model=final_result["model"],
            X_test=final_result["X_test"],
            y_test=final_result["y_test"],
            threshold=0.4,
            feature_experiment=(
                "shap_threshold_0.01"
            )
        )
    )

    print(
        "\n========== FINAL TEST COMPLETE =========="
    )

    print(
        "ROC-AUC:",
        final_test_result["auc_roc"]
    )

    print(
        "PR-AUC:",
        final_test_result["auc_pr"]
    )

    print(
        "Precision:",
        final_test_result["precision"]
    )

    print(
        "Recall:",
        final_test_result["recall"]
    )

    print(
        "F1:",
        final_test_result["f1_score"]
    )


    # ==================================================
    # 9. RUNTIME PREDICTION
    # ==================================================
    #
    # FastAPI will call orchestrator.predict() later.
    #
    # This is intentionally NOT called from main().
    #
    # The runtime path will be:
    #
    #     FastAPI
    #         ↓
    #     ModelOrchestrator.predict()
    #         ↓
    #     finalized model + inference config
    #         ↓
    #     probability + prediction
    #
    # --------------------------------------------------

    # prediction_result = orchestrator.predict(
    #     model_name="xgboost",
    #     X=prepared_inference_data
    # )
    #
    # print(
    #     "\n========== PREDICTION =========="
    # )
    #
    # print(
    #     "Probability:",
    #     prediction_result["probability"]
    # )
    #
    # print(
    #     "Prediction:",
    #     prediction_result["prediction"]
    # )


if __name__ == "__main__":
    main()