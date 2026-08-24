from xgboost import XGBClassifier


class XGBoostModel:

    def __init__(
            self,
            max_depth,
            learning_rate,
            min_child_weight,
            n_estimators=4000,
            early_stopping_rounds=100,
            random_state=42
    ):
        """
        XGBoost model wrapper.

        This class is responsible only for the XGBoost model
        and its training / prediction lifecycle.

        It does NOT:
            - preprocess data
            - perform SHAP analysis
            - select features
            - run experiments
            - decide which parameters are best
            - select a classification threshold
            - orchestrate other models
        """

        # --------------------------------------------------
        # Store model configuration
        # --------------------------------------------------

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        # --------------------------------------------------
        # Runtime model state
        # --------------------------------------------------

        self.model = None
        self.scale_pos_weight = None

        # Training results
        self.best_iteration = None
        self.validation_auc_pr = None

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    def train(
            self,
            X_train,
            y_train,
            X_validation,
            y_validation
    ):
        """
        Train the XGBoost model.

        Training data:
            Used to fit the model.

        Validation data:
            Used for validation AUC-PR and early stopping.

        Test data:
            Never supplied to this method.
        """

        # --------------------------------------------------
        # Calculate class imbalance
        # --------------------------------------------------

        # Class imbalance is calculated using TRAINING data only.

        negative_count = (y_train == 0).sum()
        positive_count = (y_train == 1).sum()

        self.scale_pos_weight = (
                negative_count / positive_count
        )

        print(
            f"Negative samples: {negative_count}"
        )

        print(
            f"Positive samples: {positive_count}"
        )

        print(
            f"Scale Pos Weight: {self.scale_pos_weight}"
        )

        # --------------------------------------------------
        # Create XGBoost model
        # --------------------------------------------------

        self.model = XGBClassifier(
            objective="binary:logistic",

            # Primary model-development metric
            eval_metric="aucpr",

            # Handle class imbalance
            scale_pos_weight=self.scale_pos_weight,

            # Support pandas categorical columns
            enable_categorical=True,

            # Histogram-based tree construction
            tree_method="hist",

            # Model configuration
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            n_estimators=self.n_estimators,

            # Early stopping uses validation performance
            early_stopping_rounds=self.early_stopping_rounds,

            random_state=self.random_state
        )

        # --------------------------------------------------
        # Train model
        # --------------------------------------------------

        self.model.fit(
            X_train,
            y_train,
            eval_set=[
                (X_validation, y_validation)
            ],
            verbose=True
        )

        # --------------------------------------------------
        # Store training results
        # --------------------------------------------------

        self.best_iteration = self.model.best_iteration
        self.validation_auc_pr = self.model.best_score

        print(
            "Best iteration:",
            self.best_iteration
        )

        print(
            "Validation AUC-PR:",
            self.validation_auc_pr
        )

        return {
            "best_iteration": self.best_iteration,
            "validation_auc_pr": self.validation_auc_pr,
            "scale_pos_weight": self.scale_pos_weight
        }

    # --------------------------------------------------
    # Predict probability
    # --------------------------------------------------

    def predict_proba(self, X):
        """
        Return the model's fraud probability.

        This method does NOT apply a classification threshold.
        """

        if self.model is None:
            raise RuntimeError(
                "XGBoost model has not been trained or loaded."
            )

        return self.model.predict_proba(X)[:, 1]

    # --------------------------------------------------
    # Predict class
    # --------------------------------------------------

    def predict(
            self,
            X,
            threshold
    ):
        """
        Convert fraud probabilities into binary predictions
        using the supplied threshold.

        Note:
            Threshold is supplied by the caller. The model class
            does not decide what the threshold should be.
        """

        probabilities = self.predict_proba(X)

        predictions = (
                probabilities >= threshold
        ).astype(int)

        return probabilities, predictions

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    def save(self, model_path):
        """
        Save the trained XGBoost model.
        """

        if self.model is None:
            raise RuntimeError(
                "Cannot save an XGBoost model before training."
            )

        self.model.save_model(model_path)

        print(
            f"XGBoost model saved to: {model_path}"
        )

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    def load(self, model_path):
        """
        Load an existing finalized XGBoost model.

        The model configuration is loaded from the XGBoost
        artifact itself.
        """

        self.model = XGBClassifier(
            enable_categorical=True
        )

        self.model.load_model(model_path)

        print(
            f"XGBoost model loaded from: {model_path}"
        )