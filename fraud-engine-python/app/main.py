from typing import Any

from fastapi import FastAPI, HTTPException

from inference.inference_preprocess import (
    prepare_inference_data
)
from orchestrator.model_orchestrator import (
    ModelOrchestrator
)
from rules.rule_engine import RuleEngine


app = FastAPI(
    title="Fraud Analysis AI API"
)

orchestrator = ModelOrchestrator()
rule_engine = RuleEngine()

@app.get("/health")
def health():
    """
    Check whether the service is running.
    """

    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(
        transaction: dict[str, Any]
):
    """
    Predict fraud probability for one transaction.
    """

    try:

        # Prepare incoming transaction
        prepared_data = prepare_inference_data(
            transaction
        )

        # Run model inference
        result = orchestrator.predict(
            model_name="xgboost",
            X=prepared_data
        )
        matched_rules = rule_engine.evaluate_rules(
            transaction
        )

        probability = float(result["fraud_probability"])
        prediction = int(result["prediction"])
        return {
            "fraud_probability": f"{probability * 100:.2f}%",
            "prediction": (
                "FRAUD"
                if prediction == 1
                else "NOT_FRAUD"
            ),
            "matched_rules": matched_rules

        }

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except FileNotFoundError as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )